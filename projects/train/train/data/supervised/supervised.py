from typing import Optional

import torch
from ml4gw.utils.slicing import sample_kernels

from train import augmentations as aug
from train.data.base import BaseAframeDataset

from ml4gw.utils.slicing import unfold_windows

Tensor = torch.Tensor

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
            waveforms = waveforms[idx].to(X.device).float()
            #waveforms = waveforms.to(X.device)
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

class SPLSupervisedAframeDataset(SupervisedAframeDataset):
    def on_after_batch_transfer(self, batch, _):
        if self.trainer.training:
            if self.waveforms_from_disk:
                [batch], waveforms = batch
                batch = self.inject(batch, waveforms)
            else:
                [batch] = batch
                batch = self.inject(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            [background, _, timeslide_idx], [signals, m1, m2] = batch
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg, m1, m2 = self.build_val_batches(background, signals, m1, m2)
            batch = (shift, X_bg, X_fg, m1, m2)
        return batch
    def build_val_batches(
        self,
        background: Tensor,
        signals: Tensor,
        m1: Tensor, 
        m2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_size = int(self.sample_length * self.hparams.sample_rate)
        stride = int(self.hparams.valid_stride * self.hparams.sample_rate)
        background = unfold_windows(background, sample_size, stride=stride)
        X, psd = self.psd_estimator(background)
        step = int(len(X) / len(signals))
        if not step:
            signals = signals[: len(X)]
            m1 = m1[:len(X)]
            m2 = m2[:len(X)]
        else:
            X = X[::step][: len(signals)]
            psd = psd[::step][: len(signals)]
            m1 = m1[:len(X)]
            m2 = m2[:len(X)]
        kernel_size = X.size(-1)
        signal_idx = signals.shape[-1] - int(
            self.waveform_sampler.right_pad * self.hparams.sample_rate
        )
        max_start = int(signal_idx - self.left_pad_size)
        max_stop = max_start + kernel_size
        pad = max_stop - signals.size(-1)
        if pad > 0:
            signals = torch.nn.functional.pad(signals, [0, pad])
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
        return X, X_inj, psd, m1, m2