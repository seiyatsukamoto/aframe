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
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj_fft = torch.fft.rfft(inj, dim=-1)
            freqs = torch.fft.rfftfreq(
                inj.shape[-1], 1 / self.hparams.sample_rate, device=self.device
            )
            mask = freqs > self.hparams.highpass
            mask *= freqs < self.hparams.lowpass
            inj_fft = inj_fft[..., mask]
            inj_fft = (2/inj.shape[-1])*inj_fft.abs()
            X_fg.append(inj_fft)

        X_fg = torch.stack(X_fg)

        X_bg_fft = torch.fft.rfft(X_bg, dim=-1)
        freqs = torch.fft.rfftfreq(
                X_bg.shape[-1], 1 / self.hparams.sample_rate, device=self.device
            )
        mask = freqs > self.hparams.highpass
        mask *= freqs < self.hparams.lowpass
        X_bg_fft = X_bg_fft[..., mask]
        X_bg = (2/X_bg.shape[-1])*X_bg_fft.abs()

        asds = psds**0.5 * 1e23
        asds = asds.float()
        num_freqs = X_fg.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = F.interpolate(asds, size=(num_freqs,), mode="linear", align_corners=False)
        inv_asds = 1 / asds
        X_bg = torch.cat([X_bg, inv_asds], dim=1).float()
        inv_asds = inv_asds.unsqueeze(dim = 0).repeat(self.hparams.num_valid_views,1,1,1)
        X_fg = torch.cat([X_fg, inv_asds], dim=2).float()
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

        freqs = np.linspace(0, self.hparams.sample_rate/2, psds.shape[-1])
        mask = freqs >= self.hparams.highpass
        mask *= freqs <= self.hparams.lowpass
        psds = psds[:, :, mask]
        asds = psds**0.5 * 1e23
        asds = asds.float()

        X_fft = torch.fft.rfft(X, dim=-1)
        freqs = torch.fft.rfftfreq(
            X.shape[-1], 1 / self.hparams.sample_rate, device=self.device
        )
        mask = freqs > self.hparams.highpass
        mask *= freqs < self.hparams.lowpass
        X_fft = X_fft[..., mask]
        num_freqs = X_fft.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = F.interpolate(asds, size=(num_freqs,), mode="linear", align_corners=False)
        inv_asds = 1 / asds
        X_fft = (2/X.shape[-1])*X_fft.abs()
        X_fft = torch.cat((X_fft, inv_asds), dim=1)
        return X_fft, y