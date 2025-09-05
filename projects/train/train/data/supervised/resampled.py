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