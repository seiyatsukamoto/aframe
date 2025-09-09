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
import random
Tensor = torch.Tensor

from typing import Callable, Optional, Union
from collections.abc import Sequence

def nonresampler(X):
    return X

class MultimodalMultibandDataset(SupervisedAframeDataset):
    def __init__(self,
                 resample_rates: Sequence[float], 
                 kernel_lengths: Sequence[float], 
                 high_passes: Sequence[float], 
                 low_passes: Sequence[float], 
                 fft_kernel_length: float,
                 fft_high_pass: float,
                 fft_low_pass: float,
                 inference_sampling_rates: Sequence[float],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resample_rates = resample_rates
        self.kernel_lengths = kernel_lengths
        self.high_passes = high_passes
        self.low_passes = low_passes
        self.fft_kernel_length = fft_kernel_length
        self.fft_high_pass = fft_high_pass
        self.fft_low_pass = fft_low_pass
        self.inference_sampling_rates = inference_sampling_rates

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
        self.whitener = []
        for band in range(len(self.resample_rates)):
            self.whitener.append(Whiten(
                self.hparams.fduration,
                self.hparams.sample_rate,
                self.hparams.high_passes[band],
                self.hparams.low_passes[band],
            ))
        self.whitener.append(Whiten(
                self.hparams.fduration,
                self.hparams.sample_rate,
                self.hparams.fft_high_pass,
                self.hparams.fft_low_pass,
            ))
        self.resampler = []
        for band in range(len(self.resample_rates)):
            self.resampler.append(T.Resample(self.hparams.sample_rate, self.resample_rates[band]))
        self.projector = aug.WaveformProjector(
            self.hparams.ifos,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        templates = []
        template_shape = []
        for x, y in zip(self.inference_sampling_rates[:-1], self.inference_sampling_rates[1:]):
            template_shape.append(int(x/y))
        for i in range(template_shape[0]):
            for j in range(template_shape[1]):
                templates.append([])
                offset1 = ((i+1)*1/self.inference_sampling_rates[0])*self.hparams.sample_rate
                offset2 = ((j+1)*1/self.inference_sampling_rates[1])*self.hparams.sample_rate
                if offset1 == 0:
                    templates[-1].append(slice(int(-offset1-(self.kernel_lengths[1]+self.hparams.fduration)*self.hparams.sample_rate), None, 1))
                else:
                    templates[-1].append(slice(int(-offset1-(self.kernel_lengths[1]+self.hparams.fduration)*self.hparams.sample_rate), 
                                               int(-offset1), 1))
                templates[-1].append(slice(int(-offset1-offset2-(self.kernel_lengths[2]+self.hparams.fduration)*self.hparams.sample_rate), 
                                           int(-offset1-offset2), 1))
        self.templates = templates
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_fg_fft = []
        for inj in X_inj:
            inj = self.whitener[-1](inj[..., int(-(self.fft_kernel_length+self.hparams.fduration)*self.hparams.sample_rate):], psds)
            freqs = torch.fft.rfftfreq(
                inj.shape[-1], d=1 / self.hparams.sample_rate
            )
            inj = torch.fft.rfft(inj)
            mask = freqs >= self.fft_high_pass
            mask *= freqs <= self.fft_low_pass
            inj = inj[:, :, mask]
            X_fg_fft.append(inj)
        
        X_fg_fft = torch.stack(X_fg_fft)
        
        X_bg_fft = self.whitener[-1](X_bg[..., int(-(self.fft_kernel_length+self.hparams.fduration)*self.hparams.sample_rate):], psds)
        freqs = torch.fft.rfftfreq(
                X_bg_fft.shape[-1], d=1 / self.hparams.sample_rate
        )
        X_bg_fft = torch.fft.rfft(X_bg_fft)
        mask = freqs >= self.fft_high_pass
        mask *= freqs <= self.fft_low_pass
        X_bg_fft = X_bg_fft[..., mask]
        
        freqs = np.linspace(0, self.hparams.sample_rate/2, psds.shape[-1])
        mask = freqs >= self.fft_high_pass
        mask *= freqs <= self.fft_low_pass
        asds = (psds[:, :, mask]**0.5 * 1e23).float()
        
        if asds.shape[-1] != X_fg_fft.shape[-1]:
            asds = F.interpolate(asds, size=(X_fg_fft.shape[-1],), mode="linear", align_corners=False)
        
        X_bg_fft = torch.cat((X_bg_fft.real, X_bg_fft.imag, 1/asds), dim=1)
        asds = asds.unsqueeze(dim = 0).repeat(self.hparams.num_valid_views,1,1,1)
        X_fg_fft = torch.cat((X_fg_fft.real, X_fg_fft.imag, 1/asds), dim=2)
        
        bg = tuple()
        fg = tuple()
        fraction = self.resample_rates[0]/self.hparams.sample_rate
        X_bg_bp = self.whitener[0](X_bg[..., int(-(self.kernel_lengths[0]+self.hparams.fduration)*self.hparams.sample_rate):], psds)
        shape = X_bg_bp.shape
        X_bg_bp = self.resampler[0](X_bg_bp.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
        # whiten each view of injections
        X_fg_bp = []
        for inj in X_inj:
            inj = self.whitener[0](inj[..., int(-(self.kernel_lengths[0]+self.hparams.fduration)*self.hparams.sample_rate):], psds)
            shape = inj.shape
            inj = self.resampler[0](inj.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
            X_fg_bp.append(inj)
            
        X_fg_bp = torch.stack(X_fg_bp)
        bg = bg + (X_bg_bp,)
        fg = fg + (X_fg_bp,)
        template_samples = random.choices(self.templates, k = X_bg.shape[0])
        for band, kl in enumerate(self.kernel_lengths[1:]):
            fraction = self.resample_rates[band+1]/self.hparams.sample_rate
            X_bg_bp = torch.stack([X_bg[i, :, width] for i, width in enumerate([template[band] for template in template_samples])])
            X_bg_bp = self.whitener[band+1](X_bg_bp, psds)
            shape = X_bg_bp.shape
            X_bg_bp = self.resampler[band+1](X_bg_bp.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
            # whiten each view of injections
            X_fg_bp = []
            for inj in X_inj:
                inj = torch.stack([inj[i, :, width] for i, width in enumerate([template[band] for template in template_samples])])
                inj = self.whitener[band+1](inj, psds)
                shape = inj.shape
                inj = self.resampler[band+1](inj.reshape(shape[0]*shape[1], shape[2])).reshape(shape[0], shape[1], int(fraction*shape[2]))
                X_fg_bp.append(inj)
                
            X_fg_bp = torch.stack(X_fg_bp)
            bg = bg + (X_bg_bp,)
            fg = fg + (X_fg_bp,)
        bg = bg + (X_bg_fft,)
        fg = fg + (X_fg_fft,)
        return bg, fg
    
    def augment(self, X, waveforms):
        batch = super().augment(X, waveforms)
        X = self.whitener[-1](batch[0][..., int(-(self.fft_kernel_length+self.hparams.fduration)*self.hparams.sample_rate):], batch[2])
        X_fft = torch.fft.rfft(X)
        freqs = torch.fft.rfftfreq(
            X.shape[-1], d=1 / self.hparams.sample_rate
        )
        mask = freqs >= self.fft_high_pass
        mask *= freqs <= self.fft_low_pass
        X_fft = X_fft[:, :, mask]
        freqs = np.linspace(0, self.hparams.sample_rate/2, batch[2].shape[-1])
        mask = freqs >= self.fft_high_pass
        mask *= freqs <= self.fft_low_pass
        asds = (batch[2][:, :, mask]**0.5 * 1e23).float()
        if asds.shape[-1] != X_fft.shape[-1]:
            asds = F.interpolate(asds, size=(X_fft.shape[-1],), mode="linear", align_corners=False)
        X_fft = torch.cat((X_fft.real, X_fft.imag, 1/asds), dim=1)
        X = tuple()
        sliced_waveforms = batch[0][..., int(-(self.kernel_lengths[0]+self.hparams.fduration)*self.hparams.sample_rate):]
        X = X + (self.resampler[0](self.whitener[0](sliced_waveforms, batch[2])),)
        template_samples = random.choices(self.templates, k = self.hparams.batch_size)
        for band, kl in enumerate(self.kernel_lengths[1:]):
            sliced_waveforms = torch.stack([batch[0][i, :, width] for i, width in enumerate([template[band] for template in template_samples])])
            X = X + (self.resampler[band+1](self.whitener[band+1](sliced_waveforms, batch[2])),)
        X = X + (X_fft,)
        return X, batch[1]