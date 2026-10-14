from typing import Optional

import torch
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

class SupervisedAframeDataset(BaseAframeDataset):
    def __init__(
        self,
        *args,
        swap_prob: Optional[float] = None,
        mute_prob: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if swap_prob is not None and 0 < swap_prob < 1:
            self.swapper = aug.ChannelSwapper(swap_prob)
            self.swap_prob = swap_prob
        elif swap_prob is not None:
            raise ValueError(
                f"swap_prob must be between 0 and 1, got {swap_prob}"
            )
        else:
            self.swapper = None
            self.swap_prob = 0

        if mute_prob is not None and 0 < mute_prob < 1:
            self.muter = aug.ChannelMuter(mute_prob)
            self.mute_prob = mute_prob
        elif mute_prob is not None:
            raise ValueError(
                f"mute_frac must be between 0 and 1, got {mute_prob}"
            )
        else:
            self.muter = None
            self.mute_prob = 0

    @property
    def sample_prob(self):
        return self.hparams.waveform_prob + self.swap_prob + self.mute_prob

    def build_transforms(self):
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            self.hparams.kernel_length+self.hparams.fduration+(self.hparams.batch_size-1)*self.hparams.valid_stride,
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
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # build our strain dataset and dataloader
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

        # If we're not loading waveforms from disk, just return
        # the background dataloader
        if not self.waveforms_from_disk:
            return dataloader

        # build iterator for waveform loading
        # that will load chunks of waveforms
        # to be sampled from
        waveform_loader = Hdf5WaveformLoader(
            self.waveform_sampler.training_waveform_files,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        # calculate how many batches we'll sample from each chunk
        # based on requested chunks per epoch and batches per epoch
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

        # multiprocess waveform chunk loader
        # so we don't have to wait for waveforms
        waveform_loader = torch.utils.data.DataLoader(
            waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )

        # build a dataset that will sample from
        # iterator of chunks of waveforms
        waveform_dataset = ChunkedWaveformDataset(
            waveform_loader,
            batch_size=1,
            batches_per_chunk=batches_per_chunk,
        )

        return ZippedDataset(dataloader, waveform_dataset)
