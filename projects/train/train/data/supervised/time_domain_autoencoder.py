from train.data.supervised.supervised import SupervisedAframeDataset
from typing import Optional
import torch
from ml4gw.utils.slicing import sample_kernels
from train import augmentations as aug
from ml4gw.transforms.qtransform import SingleQTransform
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
from ml4gw.transforms.decimator import Decimator
from train.data.supervised.spectrogram_autoencoder import ZippedTrainDataset
from train.data.supervised.spectrogram_autoencoder import ZippedValDataset

class TimeDomainAutoencoderAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,  
        bns_waveform_sampler: WaveformSampler,
        snr_target_ratio: float, 
        schedule: list,
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bns_waveform_sampler = bns_waveform_sampler
        self.num_samples = int(int(schedule[-1][1])*self.hparams.sample_rate)
        self.snr_target_ratio = snr_target_ratio
        self.schedule = torch.tensor(schedule, dtype=torch.int)
    
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.decimator = Decimator(sample_rate=self.hparams.sample_rate,
                              schedule=self.schedule)
    
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
        self.bbh_val_waveforms = self.waveform_sampler.get_val_waveforms(
            world_size, rank
        )
        self.bns_val_waveforms = self.bns_waveform_sampler.get_val_waveforms(
            world_size, rank
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
        bbh_waveform_loader = Hdf5WaveformLoader(
            self.waveform_sampler.training_waveform_files,
            batch_size=self.hparams.chunk_size//2,
            batches_per_epoch=self.hparams.chunks_per_epoch//2 or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        world_size, _ = self.get_world_size_and_rank()
        batches_per_epoch = self.hparams.batches_per_epoch // world_size
        batches_per_chunk = (
            int(batches_per_epoch // self.hparams.chunks_per_epoch) + 1
        )
        self._logger.info(
            f"Training on pool of {bbh_waveform_loader.total} BBH waveforms. "
            f"Sampling {batches_per_chunk} batches per chunk "
            f"from {self.hparams.chunks_per_epoch} chunks "
            f"of size {self.hparams.chunk_size} each epoch"
        )
        bbh_waveform_loader = torch.utils.data.DataLoader(
            bbh_waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        bbh_waveform_dataset = ChunkedWaveformDataset(
            bbh_waveform_loader,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )

        #BNS Loader
        bns_waveform_loader = Hdf5WaveformLoader(
            self.bns_waveform_sampler.training_waveform_files,
            batch_size=self.hparams.chunk_size//2,
            batches_per_epoch=self.hparams.chunks_per_epoch//2 or 1,
            channels=["cross", "plus"],
            path="waveforms",
        )
        self._logger.info(
            f"Training on pool of {bns_waveform_loader.total} BNS waveforms. "
            f"Sampling {batches_per_chunk} batches per chunk "
            f"from {self.hparams.chunks_per_epoch} chunks "
            f"of size {self.hparams.chunk_size} each epoch"
        )
        bns_waveform_loader = torch.utils.data.DataLoader(
            bns_waveform_loader,
            num_workers=2,
            pin_memory=pin_memory,
            persistent_workers=True,
        )
        bns_waveform_dataset = ChunkedWaveformDataset(
            bns_waveform_loader,
            batch_size=self.hparams.batch_size,
            batches_per_chunk=batches_per_chunk,
        )
        return ZippedTrainDataset(dataloader, bbh_waveform_dataset, bns_waveform_dataset)

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
        num_waveforms = len(self.bbh_val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = torch.utils.data.TensorDataset(self.bbh_val_waveforms)
        bbh_signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size//2,
            shuffle=False,
            pin_memory=False,
        )
        num_waveforms = len(self.bns_val_waveforms)
        signal_batch_size = (num_waveforms - 1) // self.valid_loader_length + 1
        signal_dataset = torch.utils.data.TensorDataset(self.bns_val_waveforms)
        bns_signal_loader = torch.utils.data.DataLoader(
            signal_dataset,
            batch_size=signal_batch_size//2,
            shuffle=False,
            pin_memory=False,
        )
        dataset = ZippedValDataset(
            background_dataset,
            bbh_signal_loader,
            bns_signal_loader,
            minimum=min(self.valid_loader_length, len(bbh_signal_loader)),
        )
        return dataset
    
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
        X_target = []
        for i in range(self.hparams.num_valid_views):
            start = max_start - int(i * step)
            stop = start + kernel_size
            injected = X_bg + signals[:, :, int(start) : int(stop)]
            X_inj.append(injected)
            injected = X_bg + self.snr_target_ratio*signals[:, :, int(start) : int(stop)]
            X_target.append(injected)
        X_inj = torch.stack(X_inj)
        X_target = torch.stack(X_target)
        #end of super().build_val_batches
        # whiten each view of injections
        X_fg = []
        X_fg_target = []
        for inj, inj_target in zip(X_inj, X_target):
            inj = self.whitener(inj, psd)
            inj = inj[..., -self.num_samples:]
            inj = self.decimator(inj)
            X_fg.append(inj)
            inj_target = self.whitener(inj_target, psd)
            inj_target = inj_target[..., -self.num_samples:]
            inj_target = self.decimator(inj_target)
            X_fg_target.append(inj_target)

        X_fg = torch.stack(X_fg)
        X_fg_target = torch.stack(X_fg_target)
        return X_fg, X_fg_target

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
        
        X_target = X.clone()
        X[mask] += kernels
        X_target[mask] += self.snr_target_ratio*kernels
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1
        
        X = self.whitener(X, psds)
        X = X[..., -self.num_samples:]
        X = self.decimator(X)
        X_target = self.whitener(X_target, psds)
        X_target = X_target[..., -self.num_samples:]
        X_target = self.decimator(X_target)
        return X, X_target