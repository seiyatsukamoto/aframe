import h5py
import math
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from typing import Literal, Optional

from train.data.base import BaseAframeDataset
from train.data.heterodyne_whiten import ht_Whiten
from train.data.waveforms import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
    WaveformSampler,
)
from train.metrics import get_timeslides

from utils.s3 import open_file
from ml4gw.dataloading import Hdf5TimeSeriesDataset

class ZippedTrainDataset(torch.utils.data.IterableDataset):
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
        for X, *waveforms in zip(*self.datasets, strict=False):
            waveforms = torch.concat(waveforms, dim = 0)
            yield X, waveforms
            
class ZippedValDataset(torch.utils.data.IterableDataset):
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
        for [background, b, timeslide_idx], *waveforms in zip(*self.datasets, strict=False):
            waveforms = torch.concat([w for [w] in waveforms], dim=0)
            yield (background, b, timeslide_idx), [waveforms]

class AllSkyBase(BaseAframeDataset):
    def __init__(
        self,
        waveform_samplers: list[WaveformSampler],
        waveform_sampler: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(waveform_sampler = waveform_samplers[0], *args, **kwargs)
        self.waveform_samplers = waveform_samplers
    
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
        val_waveforms=[]
        for waveform in self.waveform_samplers:
            val_waveforms.append(waveform.get_val_waveforms(
                world_size, rank
            ))
        self.val_waveforms = val_waveforms
        if self.waveforms_from_disk:
            self.waveform_sampler.get_train_waveforms(
                world_size, rank, self.device
            )
        self._logger.info("Initial dataloading complete")
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms()
        self.transforms_to_device()
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        #Background Loader
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

        #BBH Loader
        if not self.waveforms_from_disk:
            return dataloader
        
        num_datasets = len(self.waveform_samplers)
        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )
        waveform_dataset = []
        for sampler in self.waveform_samplers:
            waveform_loader = Hdf5WaveformLoader(
                sampler.training_waveform_files,
                batch_size=self.hparams.chunk_size//num_datasets,
                batches_per_epoch=self.hparams.chunks_per_epoch//num_datasets or 1,
                channels=["cross", "plus"],
                path="waveforms",
            )
            self._logger.info(
                f"Training on pool of {waveform_loader.total} BBH waveforms. "
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
            waveform_dataset.append(ChunkedWaveformDataset(
                waveform_loader,
                batch_size=self.hparams.batch_size,
                batches_per_chunk=batches_per_chunk,
            ))
        return ZippedTrainDataset(dataloader, *waveform_dataset)

    def val_dataloader(self) -> ZippedValDataset:
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
        num_ds = len(self.val_waveforms)
        signal_loaders = []
        for waveforms in self.val_waveforms:
            num_waveforms = len(waveforms)
            signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
            signal_dataset = torch.utils.data.TensorDataset(waveforms)
            signal_loaders.append(torch.utils.data.DataLoader(
                signal_dataset,
                batch_size=signal_batch_size//num_ds,
                shuffle=False,
                pin_memory=False,
            ))
        dataset = ZippedValDataset(
            background_dataset,
            *signal_loaders,
            minimum=min(self.valid_loader_length, len(signal_loaders[0])),
        )
        return dataset