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

class FFTAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
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
        
        X_bg = torch.cat((X_bg.real, X_bg.imag, 1/asds), dim=1)
        asds = asds.unsqueeze(dim = 0).repeat(self.hparams.num_valid_views,1,1,1)
        X_fg = torch.cat((X_fg.real, X_fg.imag, 1/asds), dim=2)
        return X_bg, X_fg

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
            
        return torch.cat((X_fft.real, X_fft.imag, 1/asds), dim=1), y