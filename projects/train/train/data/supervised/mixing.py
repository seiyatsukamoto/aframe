import torch
import h5py
from train.data.supervised.supervised import SupervisedAframeDataset
import os
from utils.preprocessing import PsdEstimator
from train import augmentations as aug
from ml4gw.transforms import Whiten
import torchaudio.transforms as T
from train.metrics import get_timeslides
from ml4gw.utils.slicing import unfold_windows
import torch.nn.functional as F
import numpy as np
from ml4gw.transforms import SpectralDensity
import random
Tensor = torch.Tensor
import ml4gw
from typing import Callable, Optional, Union
from collections.abc import Sequence

class MultimodalMultibandMixing(SupervisedAframeDataset):
    def __init__(self,
                 resample_rates: Sequence[float], 
                 high_passes: Sequence[float], 
                 low_passes: Sequence[float],
                 fft_kernel_size: float,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resample_rates = resample_rates #assumes that kernels read highest to lowest with last inde reserved for fft params
        self.high_passes = high_passes
        self.low_passes = low_passes
        self.fft_kernel_size = fft_kernel_size
        self.min_sample_rate = min(self.resample_rates)
        self.num_samples = self.hparams.kernel_length*self.hparams.sample_rate
        torch.serialization.add_safe_globals([ml4gw.distributions.PowerLaw])
        torch.serialization.add_safe_globals([torch.distributions.transforms.AffineTransform])
        torch.serialization.add_safe_globals([torch.distributions.transforms.PowerTransform])
        torch.serialization.add_safe_globals([torch.distributions.uniform.Uniform])
        torch.serialization.add_safe_globals([ml4gw.distributions.Cosine])

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
        whitener = []
        for band in range(len(self.resample_rates)):
            whitener.append(Whiten(
                self.hparams.fduration,
                self.hparams.sample_rate,
                self.hparams.high_passes[band],
                self.hparams.low_passes[band],
            ))
        self.whitener = torch.nn.ModuleList(whitener)
        resampler = []
        for band in range(len(self.resample_rates)):
            resampler.append(T.Resample(self.hparams.sample_rate, self.resample_rates[band]))
        self.resampler = torch.nn.ModuleList(resampler)
        self.projector = aug.WaveformProjector(
            self.hparams.ifos,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_fg_fft = []
        for inj in X_inj:
            inj = self.resampler[-1](self.whitener[-1](inj[..., -(self.hparams.fduration+self.fft_kernel_size)*self.hparams.sample_rate:], psds))
            freqs = torch.fft.rfftfreq(
                inj.shape[-1], d=1 / self.hparams.sample_rate
            )
            inj = torch.fft.rfft(inj)
            mask = freqs >= self.high_passes[-1]
            mask *= freqs <= self.low_passes[-1]
            inj = inj[:, :, mask]
            X_fg_fft.append(inj)
        
        X_fg_fft = torch.stack(X_fg_fft)
        
        X_bg_fft = self.resampler[-1](self.whitener[-1](X_bg[..., -(self.hparams.fduration+self.fft_kernel_size)*self.hparams.sample_rate:], psds))
        freqs = torch.fft.rfftfreq(
                X_bg_fft.shape[-1], d=1 / self.hparams.sample_rate
        )
        X_bg_fft = torch.fft.rfft(X_bg_fft)
        mask = freqs >= self.high_passes[-1]
        mask *= freqs <= self.low_passes[-1]
        X_bg_fft = X_bg_fft[..., mask]
        
        freqs = np.linspace(0, self.hparams.sample_rate/2, psds.shape[-1])
        mask = freqs >= self.high_passes[-1]
        mask *= freqs <= self.low_passes[-1]
        asds = (psds[:, :, mask]**0.5 * 1e23).float()
        
        if asds.shape[-1] != X_fg_fft.shape[-1]:
            asds = F.interpolate(asds, size=(X_fg_fft.shape[-1],), mode="linear", align_corners=False)
        
        X_bg_fft = torch.cat((X_bg_fft.real, X_bg_fft.imag, 1/asds), dim=1)
        asds = asds.unsqueeze(dim = 0).repeat(self.hparams.num_valid_views,1,1,1)
        X_fg_fft = torch.cat((X_fg_fft.real, X_fg_fft.imag, 1/asds), dim=2)
        
        bg = tuple()
        fg = tuple()
        for band, rr in enumerate(self.resample_rates[:-1]):
            fraction = rr/self.hparams.sample_rate
            sample_size = int(-(self.num_samples*self.min_sample_rate/rr)-(self.hparams.fduration*self.hparams.sample_rate))
            X_bg_bp = self.whitener[band](X_bg[..., sample_size:], psds)
            shape = X_bg_bp.shape
            X_bg_bp = self.resampler[band](X_bg_bp.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
            # whiten each view of injections
            X_fg_bp = []
            for inj in X_inj:
                inj = self.whitener[band](inj[..., sample_size:], psds)
                shape = inj.shape
                inj = self.resampler[band](inj.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
                X_fg_bp.append(inj)
                
            X_fg_bp = torch.stack(X_fg_bp)
            bg = bg + (X_bg_bp,)
            fg = fg + (X_fg_bp,)
        bg = bg + (X_bg_fft,)
        fg = fg + (X_fg_fft,)
        return bg, fg
    
    @torch.no_grad()
    def inject(self, X, waveforms):
        batch = super().inject(X, waveforms)
        X = self.resampler[-1](self.whitener[-1](batch[0][..., -(self.hparams.fduration+self.fft_kernel_size)*self.hparams.sample_rate:], batch[2]))
        X_fft = torch.fft.rfft(X)
        freqs = torch.fft.rfftfreq(
            X.shape[-1], d=1 / self.hparams.sample_rate
        )
        mask = freqs >= self.high_passes[-1]
        mask *= freqs <= self.low_passes[-1]
        X_fft = X_fft[:, :, mask]
        freqs = np.linspace(0, self.hparams.sample_rate/2, batch[2].shape[-1])
        mask = freqs >= self.high_passes[-1]
        mask *= freqs <= self.low_passes[-1]
        asds = (batch[2][:, :, mask]**0.5 * 1e23).float()
        if asds.shape[-1] != X_fft.shape[-1]:
            asds = F.interpolate(asds, size=(X_fft.shape[-1],), mode="linear", align_corners=False)
        X_fft = torch.cat((X_fft.real, X_fft.imag, 1/asds), dim=1)
        X = tuple()
        for band, rr in enumerate(self.resample_rates[:-1]):
            sample_size = int(-(self.num_samples*self.min_sample_rate/rr)-(self.hparams.fduration*self.hparams.sample_rate))
            X = X + (self.resampler[band](self.whitener[band](batch[0][..., sample_size:], batch[2])),)
        X = X + (X_fft,)
        return X, batch[1]
    
