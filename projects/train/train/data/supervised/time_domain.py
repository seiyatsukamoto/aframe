import math
from typing import Literal

import torch
from ml4gw.transforms import Heterodyne
from utils.augmentation import select_top_k

from train.data.base import ZippedDataset
from train.data.supervised.supervised import SupervisedAframeDataset
from train.data.waveforms import (
    ChunkedWaveformDataset,
    Hdf5WaveformLoader,
    WaveformLoader,
    WaveformSampler,
)
from train.metrics import get_timeslides



class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
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

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        return X, y


class SPLSupervisedAframeDataset(TimeDomainSupervisedAframeDataset):
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
        self.val_waveforms, self.spl_waveforms, self.cm, self.mr = self.waveform_sampler.get_val_waveforms(world_size, rank)
        if self.waveforms_from_disk:
            self.waveform_sampler.get_train_waveforms(
                world_size, rank, self.device
            )
        self._logger.info("Initial dataloading complete")
        self._logger.info("Constructing sample rate dependent transforms")
        self.build_transforms()
        self.transforms_to_device()
    
    def val_dataloader(self) -> ZippedDataset:
        background_dataset = pl.utilities.combined_loader.CombinedLoader(
            self.timeslides, mode="sequential"
        )
        iter(background_dataset)  # gives it a __len__ property
        num_waveforms = len(self.val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = torch.utils.data.TensorDataset(self.val_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        val_dataset = ZippedDataset(
            background_dataset,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )

        spl_parameters = pl.utilities.combined_loader.CombinedLoader(
            (self.cm, self.mr), mode="sequential"
        )
        num_waveforms = len(self.spl_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = torch.utils.data.TensorDataset(self.spl_waveforms)
        signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=2*signal_batch_size,
            shuffle=False,
            pin_memory=False,
        )
        spl_dataset = ZippedDataset(
            spl_parameters,
            signal_loader,
            minimum=min(self.valid_loader_length, len(signal_loader)),
        )
        return val_dataset, spl_dataset
    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            if self.waveforms_from_disk:
                [batch], waveforms = batch
                batch = self.inject(batch, waveforms)
            else:
                [batch] = batch
                batch = self.inject(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            if len(batch[0]) == 3:
                [background, _, timeslide_idx], [signals] = batch
                shift = self.timeslides[timeslide_idx].shift_size
                X_bg, X_fg = self.build_val_batches(background, signals)
                batch = (shift, X_bg, X_fg)
        return batch

class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    """
    A derived class from BaseAframeDataset and SupervisedAframeDataset, it
    applies heterodyning to strain data and returns heterodyned timeseries
    for loading data to train Aframe models. If `keep_last_n_seconds` is
    passed, returns only the final portion of the heterodyned strain.

    Args:
        chirp_mass_low (float):
            Lower bound of chirp mass range (in solar masses).
        chirp_mass_high (float):
            Upper bound of chirp mass range (in solar masses).
        num_chirp_masses (int):
            Number of chirp mass samples to generate.
        chirp_mass_spacing (Literal["linear", "log"]):
            Spacing of chirp mass grid. Use "linear" for evenly spaced
            values or "log" for logarithmic spacing.
        keep_last_n_seconds (float):
            If provided, only the last `n` seconds of the kernel_length are
            returned. Otherwise, the full kernel_length is returned.
        top_k (int):
            If provided, the top `k` chirp mass channels are selected based on
            their absolute amplitude, corresponding to the `k` matching chirp
            masses with. If `None`, all chirp mass channels are returned.
    """

    def __init__(
        self,
        chirp_mass_low: float = 1.0,
        chirp_mass_high: float = 2.5,
        num_chirp_masses: int = 100,
        chirp_mass_spacing: Literal["linear", "log"] = "log",
        keep_last_n_seconds: float = None,
        top_k: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_grid = self._create_chirp_mass_grid(
            chirp_mass_low,
            chirp_mass_high,
            num_chirp_masses,
            chirp_mass_spacing,
        )

        self.keep_last_n_seconds = keep_last_n_seconds
        self.top_k = top_k
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        else:
            self.keep_last_n_samples = None

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.heterodyne_transform = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass=self.chirp_mass_grid,
            return_type="time",
        )

    def _create_chirp_mass_grid(
        self,
        chirp_mass_low: float,
        chirp_mass_high: float,
        num_chirp_masses: int,
        chirp_mass_spacing: Literal["linear", "log"],
    ) -> torch.Tensor:
        if chirp_mass_spacing == "linear":
            return torch.linspace(
                chirp_mass_low, chirp_mass_high, num_chirp_masses
            )
        elif chirp_mass_spacing == "log":
            return torch.logspace(
                math.log10(chirp_mass_low),
                math.log10(chirp_mass_high),
                num_chirp_masses,
            )
        else:
            raise ValueError(
                f"Invalid chirp mass spacing: {chirp_mass_spacing}"
            )

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg = self.heterodyne_transform(X_bg)
        if self.top_k is not None:
            X_bg = select_top_k(
                X_bg, self.top_k, keep_last_n_samples=self.keep_last_n_samples
            )
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            inj = self.heterodyne_transform(inj)
            if self.top_k is not None:
                inj = select_top_k(
                    inj,
                    self.top_k,
                    keep_last_n_samples=self.keep_last_n_samples,
                )
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)

        if self.keep_last_n_seconds is not None:
            return X_bg[..., -self.keep_last_n_samples :], X_fg[
                ..., -self.keep_last_n_samples :
            ]
        else:
            return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        X = self.heterodyne_transform(X)
        if self.top_k is not None:
            X = select_top_k(
                X, self.top_k, keep_last_n_samples=self.keep_last_n_samples
            )

        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], y
        else:
            return X, y
