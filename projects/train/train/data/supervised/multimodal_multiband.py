import torch
import h5py
from train.data.supervised.supervised import SupervisedAframeDataset
import os
from utils.preprocessing import PsdEstimator
from train import augmentations as aug
from ml4gw.transforms import Whiten
import torchaudio.transforms as T
from train.metrics import get_timeslides
from train.waveform_sampler import WaveformSampler
from ml4gw.utils.slicing import unfold_windows
import torch.nn.functional as F
import numpy as np
from ml4gw.transforms import SpectralDensity
import random
Tensor = torch.Tensor

from ml4gw.utils.slicing import sample_kernels
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
                 inference_sampling_rates: Sequence[float],
                 initial_offsets: Sequence[float],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resample_rates = resample_rates
        self.kernel_lengths = kernel_lengths
        self.high_passes = high_passes
        self.low_passes = low_passes
        self.inference_sampling_rates = inference_sampling_rates
        self.min_kernel_size = int(self.hparams.kernel_lengths[0]*self.hparams.sample_rate)
        self.initial_offsets = np.array(initial_offsets)
        
    def slice_waveforms(self, waveforms: torch.Tensor) -> torch.Tensor:
        signal_idx = int(self.signal_time * self.hparams.sample_rate)
        kernel_size = int(self.hparams.kernel_length * self.hparams.sample_rate)
        
        signal_start = signal_idx - kernel_size + self.right_pad_size - self.filter_size//2
        
        signal_stop = signal_idx + self.min_kernel_size - self.left_pad_size + self.filter_size//2
        
        # If signal_start is less than 0, add padding on the left
        if signal_start < 0:
            raise ValueError(
                f"signal_start ({signal_start}) cannot be less than 0 "
                f"padding not yet implemented in this model"
            )
        if signal_stop - waveforms.shape[-1] > 0:
            raise ValueError(
                f"signal_stop ({signal_stop}) cannot greater than waveform "
                f"padding not implemented in this model (maybe in future)"
            )
        
        waveforms = waveforms[..., signal_start:signal_stop]
        return waveforms
    
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
        templates = []
        template_shape = []
        for x, y in zip(self.inference_sampling_rates[:-2], self.inference_sampling_rates[1:-1]):
            template_shape.append(int(x/y))
        
        initial_offsets_rate = self.hparams.sample_rate * self.initial_offsets // max(self.inference_sampling_rates)
        grid = np.meshgrid(*(range(i) for i in template_shape))
        grid = [i.flatten() for i in grid]
        grid = np.dstack(grid)[0]
        sample_size = [int((self.kernel_lengths[i+1]+self.hparams.fduration)*self.hparams.sample_rate) for i in range(len(self.kernel_lengths)-2)]
        for x in grid:
            templates.append([])
            offsets = [off*self.hparams.sample_rate//self.inference_sampling_rates[i] + initial_offsets_rate[i] for i, off in enumerate(x)]
            current_offset = 0
            for off, size in zip(offsets,sample_size):
                if off == 0:
                    templates[-1].append(slice(int(-off-size), None, 1))
                else:
                    templates[-1].append(slice(int(-off-size-current_offset), int(-off-current_offset), 1))
                    current_offset += off
        
        self.templates = templates

    @torch.no_grad()
    def super_build_val_batches(
        self, background: Tensor, signals: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Unfold a timeseries of background data
        into a batch of kernels, then inject
        multiple views of the provided signals
        into these timeseries.

        Args:
            background: A tensor of background data
            signals: A tensor of signals to inject

        Returns:
            raw strain background kernels, injected kernels, and psds
        """

        # unfold the background data into kernels
        sample_size = int(self.sample_length * self.hparams.sample_rate)
        stride = int(self.hparams.valid_stride * self.hparams.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)

        # split data into kernel and psd data and estimate psd
        X, psd = self.psd_estimator(background)
        # sometimes at the end of a segment, there won't be
        # enough background kernels and so we'll have to inject
        # our signals on overlapping data and ditch some at the end
        step = int(len(X) / len(signals))
        if not step:
            signals = signals[: len(X)]
        else:
            X = X[::step][: len(signals)]
            psd = psd[::step][: len(signals)]

        # create `num_view` instances of the injection on top of
        # the background, each showing a different, overlapping
        # portion of the signal
        signal_idx = int(self.signal_time * self.hparams.sample_rate)
        kernel_size = int(self.hparams.kernel_length * self.hparams.sample_rate)
        
        min_start = int(
            signal_idx - kernel_size - self.filter_size//2 + self.right_pad_size
        )
        max_start = int(
            signal_idx - kernel_size - self.filter_size//2 + self.min_kernel_size - self.left_pad_size
        )
        max_stop = max_start + kernel_size
        pad = max_stop - signals.size(-1)
        if pad > 0:
            signals = torch.nn.functional.pad(signals, [0, pad])
        step = (max_start - min_start)/(self.hparams.num_valid_views - 1)
        X_inj = []
        for i in range(self.hparams.num_valid_views):
            start = max_start - int(i * step)
            stop = start + kernel_size + self.filter_size
            injected = X + signals[:, :, int(start) : int(stop)]
            X_inj.append(injected)
        X_inj = torch.stack(X_inj)

        return X, X_inj, psd
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = self.super_build_val_batches(background, signals)
        X_fg_fft = []
        for inj in X_inj:
            inj = self.resampler[-1](self.whitener[-1](inj[..., int(-(self.kernel_lengths[-1]+self.hparams.fduration)*self.hparams.sample_rate):], psds))
            freqs = torch.fft.rfftfreq(
                inj.shape[-1], d=1 / self.hparams.sample_rate
            )
            inj = torch.fft.rfft(inj)
            mask = freqs >= self.high_passes[-1]
            mask *= freqs <= self.low_passes[-1]
            inj = inj[:, :, mask]
            X_fg_fft.append(inj)
        
        X_fg_fft = torch.stack(X_fg_fft)
        
        X_bg_fft = self.whitener[-1](X_bg[..., int(-(self.kernel_lengths[-1]+self.hparams.fduration)*self.hparams.sample_rate):], psds)
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
        for band, kl in enumerate(self.kernel_lengths[1:-1]):
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
    
    @torch.no_grad()
    def super_augment(self, X, waveforms):
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        # sample enough waveforms to do true injections,
        # swapping, and muting

        *params, polarizations, mask = self.waveform_sampler(
            X, self.sample_prob, waveforms
        )
        N = len(params[0])
        snrs = self.snr_sampler.sample((N,)).to(X.device)
        responses = self.projector(*params, snrs, psds[mask], **polarizations)
        kernels = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True
        )

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        swap_indices = mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1

        return X, y, psds
    
    @torch.no_grad()
    def augment(self, X, waveforms):
        batch = self.super_augment(X, waveforms)
        X = self.resampler[-1](self.whitener[-1](batch[0][..., int(-(self.kernel_lengths[-1]+self.hparams.fduration)*self.hparams.sample_rate):], batch[2]))
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
        sliced_waveforms = batch[0][..., int(-(self.kernel_lengths[0]+self.hparams.fduration)*self.hparams.sample_rate):]
        X = X + (self.resampler[0](self.whitener[0](sliced_waveforms, batch[2])),)
        template_samples = random.choices(self.templates, k = self.hparams.batch_size)
        for band, kl in enumerate(self.kernel_lengths[1:-1]):
            sliced_waveforms = torch.stack([batch[0][i, :, width] for i, width in enumerate([template[band] for template in template_samples])])
            X = X + (self.resampler[band+1](self.whitener[band+1](sliced_waveforms, batch[2])),)
        X = X + (X_fft,)
        return X, batch[1]
