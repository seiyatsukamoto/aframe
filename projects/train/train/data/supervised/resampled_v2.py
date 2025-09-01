import torch
import h5py
import os
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Union, List
import lightning.pytorch as pl

from ml4gw.transforms import SpectralDensity
from ml4gw.utils.slicing import unfold_windows
from ml4gw.transforms import Whiten
from ml4gw.utils.slicing import sample_kernels

from train.data.supervised.supervised import SupervisedAframeDataset
from train.data.resampled_hdf5_dataset import ResampledHdf5TimeSeriesDataset

from ledger.injections import WaveformSet, waveform_class_factory
from train import augmentations as aug
from train.data.utils import fs as fs_utils
from train.metrics import get_timeslides
from train.waveform_sampler import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformSampler,
)
from utils import x_per_y
from utils.preprocessing import PsdEstimator

Tensor = torch.Tensor

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
        return zip(*self.datasets)

class ResampledAframeDataset_v2(SupervisedAframeDataset):
    def __init__(
        self,
        resampler: Callable[[Tensor], Tensor], 
        file_sample_rate: float,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resampler = resampler
        self.file_sample_rate = file_sample_rate
        self.save_hyperparameters(ignore=['resampler'])
    
    def load_val_waveforms(self, f, world_size, rank):
        waveform_set = self.waveform_set_cls.read(f)

        if waveform_set.coalescence_time != self.signal_time:
            raise ValueError(
                "Training waveforms and validation waveforms have different "
                f"signal times, got {self.signal_time} and "
                f"{waveform_set.coalescence_time}, respectively"
            )

        length = len(waveform_set.waveforms)

        if not rank:
            self._logger.info(f"Validating on {length} waveforms")
        stop, start = self.get_slice_bounds(length, world_size, rank)

        self._logger.info(f"Loading {start - stop} validation signals")
        start, stop = -start, -stop or None
        waveforms = torch.tensor(waveform_set.waveforms[start:stop], dtype = torch.float)
        waveforms = self.resampler(waveforms)
        return waveforms
    
    def load_val_background(self, fnames: list[str]):
        self._logger.info("Loading validation background data")
        val_background = []
        for fname in fnames:
            segment = []
            with h5py.File(fname, "r") as f:
                for ifo in self.hparams.ifos:
                    segment.append(torch.tensor(f[ifo][:], dtype = torch.float))
                val_background.append(self.resampler(torch.stack(segment)))
        return val_background
    
    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)
        self.train_fnames, self.valid_fnames = self.train_val_split()
        
        # now define some of the augmentation transforms
        # that require sample rate information
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms()
        self.transforms_to_device()
        self.resampler.to('cpu')

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

        self.waveform_sampler = WaveformSampler()

        val_waveform_file = os.path.join(self.data_dir, "val_waveforms.hdf5")
        self.val_waveforms = self.load_val_waveforms(
            val_waveform_file, world_size, rank
        )
        self._logger.info("Initial dataloading complete")
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # divide batches per epoch up among all devices
        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size

        # build our strain dataset and dataloader
        dataset = ResampledHdf5TimeSeriesDataset(
            self.train_fnames,
            channels=self.hparams.ifos,
            kernel_size=int(self.hparams.file_sample_rate * self.sample_length),
            batch_size=self.hparams.batch_size,
            batches_per_epoch=self.batches_per_epoch,
            coincident=False,
            num_files_per_batch=self.hparams.num_files_per_batch,
            resampler=self.resampler,
        )
    
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        # multiprocess data loading
        local_world_size = len(self.trainer.device_ids)
        num_workers = min(6, int(os.cpu_count() / local_world_size))
        self._logger.debug(
            f"Using {num_workers} workers for strain data loading"
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # build iterator for waveform loading
        # that will load chunks of waveforms
        # to be sampled from
        waveform_loader = Hdf5WaveformLoader(
            self.train_waveform_fnames,
            batch_size=self.hparams.chunk_size,
            batches_per_epoch=self.hparams.chunks_per_epoch or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        # calculate how many batches we'll sample from each chunk
        # based on requested chunks per epoch and batches per epoch
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

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)
        return X, y


class FFTAframeDataset(SupervisedAframeDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
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
        self.whitener = Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
        self.projector = aug.WaveformProjector(
            self.hparams.ifos,
            self.hparams.sample_rate,
            self.hparams.highpass,
            self.hparams.lowpass,
        )
    
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