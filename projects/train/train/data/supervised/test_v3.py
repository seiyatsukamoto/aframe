import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional

from train.data.supervised.supervised import SupervisedAframeDataset
#from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

from ml4gw.utils.slicing import sample_kernels
from train import augmentations as aug
from train.data.base import BaseAframeDataset, ZippedDataset
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.utils.slicing import unfold_windows
from train.data.waveforms import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
    WaveformSampler,
)
from ml4gw.transforms import Whiten
from utils.preprocessing import PsdEstimator
import lightning.pytorch as pl
from train.data.heterodyne_whiten import ht_Whiten


class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file = chirp_mass_file
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds

        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.ht_whitener = ht_Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.ht_highpass,
            self.ht_lowpass,
        )
    
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.heterodyne_transform(X_bg)
        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
        X_bg = X_bg.reshape(_B_bg, _C_bg * _M_bg, _T_bg)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj = self.heterodyne_transform(inj)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        _V_fg, _B_fg, _C_fg, _M_fg, _T_fg = X_fg.shape
        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg * _M_fg, _T_fg)

        if self.keep_last_n_seconds is not None:
            return X_bg[..., -self.keep_last_n_samples :], X_fg[
                ..., -self.keep_last_n_samples :
            ]
        else:
            return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.ht_whitener(X, psds)
        return X, y


class TopKHeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        k: int,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file = chirp_mass_file
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        self.k = k
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.ht_whitener = ht_Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.ht_highpass,
            self.ht_lowpass,
        )
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        sample_size = int((self.hparams.kernel_length + self.hparams.fduration) * self.hparams.sample_rate)
        stride = int(self.hparams.valid_stride * self.hparams.sample_rate)
        
        # split data into kernel and psd data and estimate psd
        background = background.unsqueeze(0)
        X, psd = self.psd_estimator(background)
        X = X.squeeze(0)
        
        X = unfold_windows(X, sample_size, stride=stride)
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
        kernel_size = X.size(-1)
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
            injected = X + signals[:, :, int(start) : int(stop)]
            X_inj.append(injected)
        X_inj = torch.stack(X_inj)
        #end of supervised
        
        X_td, X_ht = self.ht_whitener(X, psds)
        _B, _M, _T = X_ht.shape
        X_ht = self.topk_bin(X_ht, _M//2, _B)
        
        # whiten each view of injections
        X_fg_td = []
        X_fg_ht = []
        for inj in X_inj:
            inj_td, inj_ht = self.ht_whitener(inj, psds)
            X_fg_td.append(inj_td)
            X_fg_ht.append(inj_ht)
        X_fg_td = torch.stack(X_fg_td)
        X_fg_ht = torch.stack(X_fg_ht)
        
        _V, _B, _M, _T = X_fg_ht.shape
        X_fg_ht = X_fg_ht.view(_V*_B, _M, _T)
        X_fg_ht = self.topk_bin(X_fg_ht, _M//2, _V*_B)
        X_fg_ht = X_fg_ht.view(_V, _B, 2 * self.k, _T)
        return torch.cat([X_td, X_ht], dim=1), torch.cat([X_fg_td, X_fg_ht], dim=2)
    
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
        # sample enough waveforms to do true injections,
        # swapping, and muting
        
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob
        
        dec, psi, phi = self.sample_extrinsic(X[mask])
        # If we're loading waveforms from disk, we can
        # slice out the ones we want.
        # If not, we're generating them on the fly.
        if self.waveforms_from_disk:
            # TODO: Can we just use `mask` to slice out the
            # waveforms we want here? Copying this from the
            # old `WaveformSampler` in case it handles edge
            # cases I'm not thinking of
            N = mask.sum().item()
            idx = torch.randperm(waveforms.shape[0])[:N] 
            #idx = torch.arange(waveforms.shape[0]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            waveforms = waveforms[idx].to(X.device).float()
            hc, hp = waveforms[:, 0], waveforms[:, 1]
        else:
            hc, hp = self.waveform_sampler.sample(X[mask])

        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )
        # If we're loading waveforms from disk, we'll have sliced
        # the waveforms already in `on_before_batch_transfer`
        if not self.waveforms_from_disk:
            responses = self.slice_waveforms(responses)
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
        X_td, X_ht = self.ht_whitener(X, psds)
        _B, _M, _T = X_ht.shape
        X_ht = self.topk_bin(X_ht, _M//2, _B) #Assuming num_ifos = 2
        return torch.cat([X_td, X_ht], dim=1), y
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length) + int(self.hparams.sample_rate*self.hparams.valid_stride*(self.hparams.batch_size - 1)),
            batch_size=1,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
        )

        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        self._logger.debug(
            f"Using {self.num_workers} workers for strain data loading"
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
        if not self.waveforms_from_disk:
            return dataloader
        waveform_loader = Hdf5WaveformLoader(
            self.waveform_sampler.training_waveform_files,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )
        self._logger.info(
            f"Training on pool of {waveform_loader.total} waveforms. "
            f"Sampling {batches_per_chunk} batches per chunk "
            f"from {self.hparams.chunks_per_epoch} chunks "
            f"of size {self.hparams.chunk_size} each epoch"
        )
        waveform_loader = torch.utils.data.DataLoader(
            waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        waveform_dataset = ChunkedWaveformDataset(
            waveform_loader,
            batch_size=1,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedDataset(dataloader, waveform_dataset)
    def val_dataloader(self) -> ZippedDataset:
        background_dataset = pl.utilities.combined_loader.CombinedLoader(
            self.timeslides, mode="sequential"
        )
        iter(background_dataset)  # gives it a __len__ property
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = _SignalDataset(self.val_waveforms, self.val_params)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=False,
        )
        dataset = ZippedDataset(
            background_dataset,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )
        return dataset