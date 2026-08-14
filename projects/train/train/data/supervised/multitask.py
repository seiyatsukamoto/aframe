import glob
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Optional, Union

import h5py
import lightning.pytorch as pl
import torch
from ml4gw.augmentations import SignalInverter, SignalReverser
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import unfold_windows
from ml4gw.utils.slicing import sample_kernels

from train import augmentations as aug
from train.data.utils import fs as fs_utils
from train.metrics import get_timeslides
from train.data.waveforms.mt_loader import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
)
from train.data.waveforms.mt_sampler import WaveformSampler
from train.data.base import BaseAframeDataset

from utils.preprocessing import PsdEstimator

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution
TransformedDist = torch.distributions.TransformedDistribution

SECONDS_PER_DAY = 86400


class _SignalDataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, params):
        self.waveforms, self.params = waveforms, params

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, i):
        return self.waveforms[i], {k: v[i] for k, v in self.params.items()}

class ZippedDataset(torch.utils.data.IterableDataset):
    def __init__(self, *datasets, minimum: Optional[int] = None):
        super().__init__()
        self.datasets = datasets
        self.minimum = minimum

    def __len__(self):
        lengths = []
        for dset in self.datasets:
            try:
                lengths.append(len(dset))
            except Exception as e:
                raise e from None
        return self.minimum or min(lengths)

    def __iter__(self):
        return zip(*self.datasets, strict=False)

class MultitaskSupervisedAframeDataset(BaseAframeDataset):
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
        
        self.waveforms_from_disk = True
    
    @torch.no_grad()
    def build_val_batches(
        self,
        background: Tensor,
        signals: Tensor,
        params: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """
        Unfold a timeseries of background data
        into a batch of kernels, then inject
        multiple views of the provided signals
        into these timeseries.

        Args:
            background: A tensor of background data
            signals: A tensor of signals to inject
            params: A dictionary of parameter tensors to be passed to the model

        Returns:
            raw strain background kernels, injected kernels, psds,
            and (potentially truncated) params tensor
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
            params = {k: v[: len(X)] for k, v in params.items()}
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

        #end of base
        X = self.whitener(X, psd)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psd)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        params = self.apply_param_transforms(params)
        return X, X_fg, params

    @torch.no_grad()
    def inject(
        self, X, waveforms, params
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        if waveforms is None:
            raise ValueError(
                "Waveforms should be passed to the `inject` method, got None"
            )

        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        # sample enough waveforms to do true injections,
        # swapping, and muting

        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob

        dec, psi, phi = self.sample_extrinsic(X[mask])
        N = mask.sum().item()
        idx = torch.randperm(waveforms.shape[0])[:N]
        waveforms = waveforms[idx].to(X.device).float()
        params = {k: v[idx].to(X.device).float() for k, v in params.items()}
        hc, hp = waveforms[:, 0], waveforms[:, 1]

        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )

        params["dec"] = dec
        params["psi"] = psi
        params["phi"] = phi
        params["snr"] = snrs

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

        # return NaN for params that weren't injected
        still_injected = mask[idx]
        params_out = {}
        for key, vals in params.items():
            out = torch.full((X.size(0),), float("nan"), device=X.device)
            out[idx[still_injected]] = vals[still_injected]
            params_out[key] = out
        
        #end of supervised
        X = self.whitener(X, psds)
        params_out = self.apply_param_transforms(params_out)
        return X, y, params_out
    
    @property
    def sample_prob(self):
        return self.hparams.waveform_prob + self.swap_prob + self.mute_prob
    
    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        self.train_fnames, self.valid_fnames = self.train_val_split()

        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]
            if not sample_rate == self.hparams.sample_rate:
                raise ValueError(
                    f"Specified sample rate is {self.hparams.sample_rate} "
                    f"but background data is sampled at {sample_rate}"
                )

        self._logger.info(f"Validated sample rate {sample_rate}")

        # load in our validation background up front and
        # compute which timeslides we'll do on this device
        # if we're doing distributed training so we'll know
        # which waveforms to subsample

        val_background = self.load_val_background(self.valid_fnames)
        self._logger.info(
            "Constructing validation timeslides from background segments "
            f"{' '.join(self.valid_fnames)}"
        )
        self.timeslides, self.valid_loader_length = get_timeslides(
            val_background,
            self.hparams.valid_livetime,
            self.hparams.sample_rate,
            self.sample_length,
            self.hparams.valid_stride,
            self.val_batch_size,
        )

        self.val_waveforms, self.val_params = (
            self.waveform_sampler.get_val_waveforms(world_size, rank)
        )
        if self.waveforms_from_disk:
            self.waveform_sampler.get_train_waveforms(
                world_size, rank, self.device
            )
        self._logger.info("Initial dataloading complete")

        # now define some of the augmentation transforms
        # that require sample rate information
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms()
        self.transforms_to_device()

    def on_before_batch_transfer(self, batch, _):
        """
        Slice loaded waveforms before sending to device
        if not generating waveforms during training
        """
        # TODO: maybe pass indices as argument to
        # waveform loader to reduce quantity of data
        # we need to load
        if self.trainer.training and self.waveforms_from_disk:
            X, [waveforms, params] = batch
            waveforms = self.slice_waveforms(waveforms)
            batch = X, (waveforms, params)
        return batch

    # ============================================== #
    # Utilities for doing augmentation/preprocessing #
    # after tensors have been transferred to GPU     #
    # ============================================== #
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
            if self.waveforms_from_disk:
                [X], (waveforms, params) = batch
                batch = self.inject(X=X, waveforms=waveforms, params=params)
            else:
                [X] = batch
                waveforms, params = self.waveform_sampler.sample(X)
                batch = self.inject(X=X, waveforms=waveforms, params=params)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [signals, params] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg, params = self.build_val_batches(
                background=background,
                signals=signals,
                params=params,
            )
            batch = (shift, X_bg, X_fg, params)
        return batch
    
    def apply_param_transforms( #Hardcodded here !!!!!!!!!!
        self, params: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        return {'chirp_mass': torch.log10(((params['mass_1']*params['mass_2'])**(3/5))/((params['mass_1']+params['mass_2'])**(1/5)))}
    
    def val_dataloader(self) -> ZippedDataset:
        """
        Validation dataloader will iterate through batches
        in timeslides, returning both
        """
        background_dataset = pl.utilities.combined_loader.CombinedLoader(
            self.timeslides, mode="sequential"
        )
        iter(background_dataset)  # gives it a __len__ property

        # Figure out how many batches of background
        # we're going to go through, then batch the
        # signals so that they're spaced evenly
        # throughout all those batches.
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1

        # Signal dataset to return tuples of (waveform, params)
        signal_dataset = _SignalDataset(self.val_waveforms, self.val_params)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        dataset = ZippedDataset(
            background_dataset,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )
        return dataset

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # build our strain dataset and dataloader
        dataset = Hdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
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
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedDataset(dataloader, waveform_dataset)