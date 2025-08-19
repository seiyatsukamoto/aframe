import torch
import h5py
from train.data.supervised.supervised import SupervisedAframeDataset
import os
from utils.preprocessing import PsdEstimator
from train import augmentations as aug
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels
import torchaudio.transforms as T
from train.metrics import get_timeslides
from train.waveform_sampler import WaveformSampler
from ml4gw.utils.slicing import unfold_windows
import torch.nn.functional as F
import numpy as np
from ml4gw.transforms import SpectralDensity

Tensor = torch.Tensor

def nonresampler(X):
    return X

class ResampledAframeDataset(SupervisedAframeDataset):
    def __init__(self, resample_rate: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hparams.resample_rate = resample_rate
        if self.hparams.resample_rate != self.hparams.sample_rate:
            self.resampler = T.Resample(self.hparams.sample_rate, self.hparams.resample_rate)
        else:
            self.resampler = nonresampler
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [X], waveforms = batch
            X, y = self.augment(X, waveforms)
            batch = (self.resampler(X), y)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            fraction = self.hparams.resample_rate/self.hparams.sample_rate
            X_bg, X_fg = self.build_val_batches(background, signals)
            X_bg = self.resampler(X_bg.reshape(X_bg.shape[0]*X_bg.shape[1], X_bg.shape[2])).reshape(X_bg.shape[0], X_bg.shape[1], int(fraction*X_bg.shape[2]))
            X_fg = self.resampler(X_fg.reshape(X_fg.shape[0]*X_fg.shape[1]*X_fg.shape[2], X_fg.shape[3])).reshape(X_fg.shape[0], X_fg.shape[1], 
                                                                                                                  X_fg.shape[2], int(fraction*X_fg.shape[3]))
            batch = (int(shift*fraction), X_bg, X_fg)
        return batch

    def transforms_to_device(self):
        """
        Move all `torch.nn.Modules` to the local device
        """
        for item in self.__dict__.values():
            if isinstance(item, torch.nn.Module):
#                item.to(torch.device('cuda:0'))
                item.to(self.device)

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)
        return X, y


class FFTAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def build_transforms(self):
        """
        Helper utility in case we ever want to construct
        this dataset on its own.
        """
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length,
            self.hparams.sample_rate,
            fftlength,
            window=self.psd_window,
            fast=self.hparams.highpass is not None,
            average="median",
        )
        self.whitener = Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        self.projector = aug.WaveformProjector(
            self.hparams.ifos,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        self.psd_X = SpectralDensity(
            self.hparams.sample_rate, 
            fftlength/10, 
            None, 
            "median", 
            window=self.psd_window, 
            fast=self.hparams.highpass is not None
        )
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            freqs = torch.fft.rfftfreq(
                inj.shape[-1], d=1 / self.hparams.sample_rate
            )
            inj = torch.fft.rfft(inj)
            mask = freqs >= self.hparams.highpass
            mask *= freqs <= self.hparams.lowpass
            inj = inj[:, :, mask]
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        
        X_bg = self.whitener(X_bg, psds)
        freqs = torch.fft.rfftfreq(
                X_bg.shape[-1], d=1 / self.hparams.sample_rate
        )
        X_bg = torch.fft.rfft(X_bg)
        mask = freqs >= self.hparams.highpass
        mask *= freqs <= self.hparams.lowpass
        X_bg = X_bg[..., mask]
        
        freqs = np.linspace(0, self.hparams.sample_rate/2, psds.shape[-1])
        mask = freqs >= self.hparams.highpass
        mask *= freqs <= self.hparams.lowpass
        psds = psds[..., mask]
        asds = (psds**0.5 * 1e23).float()
        
        if asds.shape[-1] != X_fg.shape[-1]:
            asds = F.interpolate(asds, size=(X_fg.shape[-1],), mode="linear", align_corners=False)
        
        X_bg = torch.cat((abs(X_bg.real)**.5, abs(X_bg.imag)**.5, 1/asds), dim=1)
        asds = asds.unsqueeze(dim = 0).repeat(self.hparams.num_valid_views,1,1,1)
        X_fg = torch.cat((abs(X_fg.real)**.5, abs(X_fg.imag)**.5, 1/asds), dim=2)
        return X_bg, X_fg

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [X], waveforms = batch
            batch = self.augment(X, waveforms)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg = self.build_val_batches(background, signals)
            batch = (shift, X_bg, X_fg)
        return batch

    def transforms_to_device(self):
        """
        Move all `torch.nn.Modules` to the local device
        """
        for item in self.__dict__.values():
            if isinstance(item, torch.nn.Module):
                item.to(self.device)

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        
        X = self.whitener(X, psds)
        X_fft = torch.fft.rfft(X)
        freqs = torch.fft.rfftfreq(
            X.shape[-1], d=1 / self.hparams.sample_rate
        )
        mask = freqs >= self.hparams.highpass
        mask *= freqs <= self.hparams.lowpass
        X_fft = X_fft[:, :, mask]

        freqs = np.linspace(0, self.hparams.sample_rate/2, psds.shape[-1])
        mask = freqs >= self.hparams.highpass
        mask *= freqs <= self.hparams.lowpass
        psds = psds[:, :, mask]
        asds = (psds**0.5 * 1e23).float()
        if asds.shape[-1] != X_fft.shape[-1]:
            asds = F.interpolate(asds, size=(X_fft.shape[-1],), mode="linear", align_corners=False)
            
        return torch.cat((abs(X_fft.real)**.5, abs(X_fft.imag)**.5, 1/asds), dim=1), y