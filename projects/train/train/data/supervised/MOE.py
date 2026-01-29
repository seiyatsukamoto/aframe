import torch

from train.data.supervised.supervised import SupervisedAframeDataset
from typing import Optional

import torch
from ml4gw.utils.slicing import sample_kernels

from train import augmentations as aug
from ml4gw.transforms.qtransform import SingleQTransform
from train.data.base import ZippedDataset
from ml4gw.utils.slicing import unfold_windows
from train.data.waveforms import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
    WaveformSampler,
)
import h5py
from train.metrics import get_timeslides
from ml4gw.dataloading import Hdf5TimeSeriesDataset
import lightning.pytorch as pl
import ml4gw
from architectures.resnet_1d_autoencoder import ResNet1D_autoencoder
from architectures.resnet_2d_autoencoder import ResNet2D_autoencoder
class MOEAframeDataset(SupervisedAframeDataset):
    def __init__(
        self, 
        q: float, 
        spectrogram_shape: list[int, int], 
        frange: list[float, float],
        time_domain_length: float,
        spectrogram_model: ResNet2D_autoencoder,
        timedomain_model: ResNet1D_autoencoder,
        spectrogram_ckpt: str,
        timedomain_ckpt: str,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spectrogram_shape = spectrogram_shape
        self.frange = frange
        self.q = q
        self.time_domain_length = time_domain_length
        
        torch.serialization.add_safe_globals([ml4gw.distributions.PowerLaw, torch.distributions.transforms.AffineTransform,
                                              torch.distributions.transforms.PowerTransform, torch.distributions.uniform.Uniform,
                                              ml4gw.distributions.Cosine])
        
        ckpt = torch.load(spectrogram_ckpt, map_location=torch.device('cpu'), weights_only = False)
        state_dict = {k.replace('model.', '', 1): v for k, v in ckpt['state_dict'].items()}
        spectrogram_model.load_state_dict(state_dict)
        spectrogram_model.eval()
        self.spectrogram_model = spectrogram_model
        
        ckpt = torch.load(timedomain_ckpt, map_location=torch.device('cpu'), weights_only = False)
        state_dict = {k.replace('model.', '', 1): v for k, v in ckpt['state_dict'].items()}
        timedomain_model.load_state_dict(state_dict)
        timedomain_model.eval()
        self.timedomain_model = timedomain_model
        
    
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.qtransform = SingleQTransform(
            duration=self.hparams.kernel_length,
            sample_rate=self.hparams.sample_rate,
            q=self.q,
            spectrogram_shape=self.spectrogram_shape,
            frange = self.frange,
        )
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        sample_size = int(self.sample_length * self.hparams.sample_rate)
        stride = int(self.hparams.valid_stride * self.hparams.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)

        # split data into kernel and psd data and estimate psd
        X_bg, psd = self.psd_estimator(background)

        # sometimes at the end of a segment, there won't be
        # enough background kernels and so we'll have to inject
        # our signals on overlapping data and ditch some at the end
        step = int(len(X_bg) / len(signals))
        if not step:
            signals = signals[: len(X_bg)]
        else:
            X_bg = X_bg[::step][: len(signals)]
            psd = psd[::step][: len(signals)]

        # create `num_view` instances of the injection on top of
        # the background, each showing a different, overlapping
        # portion of the signal
        kernel_size = X_bg.size(-1)
        signal_idx = signals.shape[-1] - int(
            self.waveform_sampler.right_pad * self.hparams.sample_rate
        )
        max_start = int(signal_idx - self.left_pad_size)
        max_stop = max_start + kernel_size
        pad = max_stop - signals.size(-1)
        if pad > 0:
            signals = torch.nn.functional.pad(signals, [0, pad])

        # Prevent division by zero if we want only
        # a single validation view
        if self.hparams.num_valid_views == 1:
            step = 0
        else:
            step = kernel_size - self.left_pad_size - self.right_pad_size
            step /= self.hparams.num_valid_views - 1

        X_inj = []
        for i in range(self.hparams.num_valid_views):
            start = max_start - int(i * step)
            stop = start + kernel_size
            injected = X_bg + signals[:, :, int(start) : int(stop)]
            X_inj.append(injected)
        X_inj = torch.stack(X_inj)
        #end of super().build_val_batches
        
        X_bg = self.whitener(X_bg, psd)
        X_bg_1 = self.timedomain_model.encoder(X_bg[..., int(-self.time_domain_length*self.hparams.sample_rate):])
        X_bg_1 = self.timedomain_model.compress(X_bg_1)
        X_bg_1 = X_bg_1.flatten(start_dim = -2)
        X_bg = self.qtransform(X_bg)
        X_bg_2 = self.spectrogram_model(X_bg)
        X_bg_2 = self.spectrogram_model.encoder(X_bg)
        X_bg_2 = self.spectrogram_model.compress(X_bg_2)
        X_bg_2 = X_bg_2.flatten(start_dim=-3)
        X_bg = torch.cat([X_bg_1, X_bg_2], dim=-1)
        
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psd) 
            inj_1 = self.timedomain_model.encoder(inj[..., int(-self.time_domain_length*self.hparams.sample_rate):])
            inj_1 = self.timedomain_model.compress(inj_1)
            inj_1 = inj_1.flatten(start_dim = -2)
            inj = self.qtransform(inj)
            inj_2 = self.spectrogram_model.encoder(inj)
            inj_2 = self.spectrogram_model.compress(inj_2)
            inj_2 = inj_2.flatten(start_dim=-3)
            inj = torch.cat([inj_1, inj_2], dim=-1)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        if self.waveforms_from_disk and waveforms is None:
            raise ValueError(
                "Waveforms should be passed to the `inject` method "
                "if waveforms are being loaded from disk, got None"
            )
        
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob
        
        dec, psi, phi = self.sample_extrinsic(X[mask])
        if self.waveforms_from_disk:
            N = mask.sum().item()
            idx = torch.randperm(waveforms.shape[0])[:N]
            waveforms = waveforms[idx].to(X.device).float()
            hc, hp = waveforms[:, 0], waveforms[:, 1]
        else:
            hc, hp = self.waveform_sampler.sample(X[mask])
        
        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )
        if not self.waveforms_from_disk:
            responses = self.slice_waveforms(responses)
        kernels = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True
        )
        
        swap_indices = mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)
        
        X[mask] += kernels
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        
        X = self.whitener(X, psds)
        X_1 = self.timedomain_model.encoder(X[..., int(-self.time_domain_length*self.hparams.sample_rate):])
        X_1 = self.timedomain_model.compress(X_1)
        X_1 = X_1.flatten(start_dim=-2)
        X = self.qtransform(X)
        mins = torch.amin(X, dim = [2, 3], keepdim=True)
        maxes = torch.amax(X, dim = [2, 3], keepdim=True)
        X = (X-mins)/(maxes-mins).clamp_min(1e-8)
        X_2 = self.spectrogram_model.encoder(X)
        X_2 = self.spectrogram_model.compress(X_2)
        X_2 = X_2.flatten(start_dim=-3)
        X = torch.cat([X_1, X_2], dim=-1)
        return X, y
