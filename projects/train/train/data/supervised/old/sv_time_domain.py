import math
import torch
from typing import Literal

from train.data.supervised.val_sv import ValSVSupervisedAframeDataset
#from ml4gw.transforms import Heterodyne
#from ml4gw.transforms.heterodyne_from_dir import Heterodyne_from_dir as Heterodyne
#from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

import torch.nn.functional as F
import numpy as np

from ml4gw.constants import MTSUN_SI


class Heterodyne(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        kernel_length: float,
        chirp_mass_file: str,
        return_type: Literal["time", "freq", "both"],
        highpass: float = 0,
        lowpass: float = 2048,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_length = kernel_length
        self.chirp_mass_file = chirp_mass_file
        self.register_buffer("heterodyning_phase", torch.tensor(np.load(self.chirp_mass_file)))
        freqs = torch.fft.rfftfreq(int(kernel_length*sample_rate), d=1.0 / sample_rate)
        mask = (freqs > lowpass) | (freqs < highpass)
        self.register_buffer("mask", mask)
        self.return_type = return_type
        if self.return_type not in {"time", "freq", "both"}:
            raise ValueError(
                "Invalid return_type. Must be one of {'time', 'freq', 'both'}."
            )
    def forward(
        self, X: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        X_fft = torch.fft.rfft(X, dim=-1)
        X_fft /= self.sample_rate
        X_heterodyned = X_fft[:, :, None] * self.heterodyning_phase[None, :, :]
        X_heterodyned[..., 0] = 0
        X_heterodyned[..., self.mask] = 0
        X_ifft = torch.fft.irfft(X_heterodyned, dim=-1)
        X_ifft *= self.sample_rate

        if self.return_type == "time":
            return X_ifft
        elif self.return_type == "freq":
            return X_heterodyned
        else:
            return X_ifft, X_heterodyned


class TimeDomainSupervisedAframeDataset(ValSVSupervisedAframeDataset):
    def build_val_batches(self, background, signals, params):
        X, X_inj, params, psd = super().build_val_batches(background, signals, params)
        X = self.whitener(X, psd)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psd)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        return X, X_fg, params
    def inject(self, X, waveforms, params):
        X, y, params, psds = super().inject(X, waveforms, params)
        X = self.whitener(X, psds)
        return X, y, params


class HeterodyneTimeDomainSupervisedAframeDataset(ValSVSupervisedAframeDataset):
    def __init__(
        self,
        phase_dir: str,
        ht_lowpass: float,
        ht_highpass: float,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.phase_dir = phase_dir
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.heterodyne_transform = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass_file=self.phase_dir,
            return_type="time",
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
        )
    def build_val_batches(self, background, signals, params):
        X_bg, X_inj, params, psds = super().build_val_batches(background, signals, params)
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
            return X_bg[..., -self.keep_last_n_samples :], X_fg[..., -self.keep_last_n_samples :], params
        else:
            return X_bg, X_fg, params
    def inject(self, X, waveforms, params):
        X, y, params, psds = super().inject(X, waveforms, params)
        X = self.whitener(X, psds)
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)
        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], y, params
        else:
            return X, y, params


class TopKHeterodyneTimeDomainSupervisedAframeDataset(ValSVSupervisedAframeDataset):
    def __init__(
        self,
        k: int,
        chirp_mass_file: str,
        ht_highpass: float,
        ht_lowpass: float,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chirp_mass_file = chirp_mass_file
        self.k = k
        self.keep_last_n_seconds = keep_last_n_seconds
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.heterodyne_transform = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass_file=self.chirp_mass_file,
            return_type="time",
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
        )
    @torch.no_grad()
    def build_val_batches(self, background, signals, params):
        X_bg, X_inj, params, psds = super().build_val_batches(background, signals, params)
        X_bg = self.whitener(X_bg, psds)
        if self.keep_last_n_seconds is not None:
            X_bg_td = X_bg[..., -self.keep_last_n_samples :].float().clone()
        else:
            X_bg = X_bg.float().clone()
        X_bg = self.heterodyne_transform(X_bg)
        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
        X_bg = X_bg.reshape(_B_bg, _C_bg * _M_bg, _T_bg)
        X_bg = self.topk_bin(X_bg, _M_bg, _B_bg)
        # whiten each view of injections
        X_fg = []
        X_fg_td = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            if self.keep_last_n_seconds is not None:
                X_fg_td.append(inj[..., -self.keep_last_n_samples :].float().clone())
            else:
                X_fg_td.append(inj.float().float().clone())
            inj = self.heterodyne_transform(inj)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        X_fg_td = torch.stack(X_fg_td)
        _V_fg, _B_fg, _C_fg, _M_fg, _T_fg = X_fg.shape
        X_fg = X_fg.view(_V_fg*_B_fg, _C_fg * _M_fg, _T_fg)
        X_fg = self.topk_bin(X_fg, _M_fg, _V_fg*_B_fg)
        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg * self.k, _T_fg)
        if self.keep_last_n_seconds is not None:
            X_bg = X_bg[..., -self.keep_last_n_samples :].float()
            X_fg = X_fg[..., -self.keep_last_n_samples :].float()
        else:
            X_bg = X_bg.float()
            X_fg = X_fg.float()
        X_bg = torch.cat([X_bg_td[:, :1, :], X_bg[:, :self.k, :], X_bg_td[:, 1:, :], X_bg[:, self.k:, :]], dim=1)
        X_fg = torch.cat([X_fg_td[:, :, :1, :], X_fg[:, :, :self.k, :], X_fg_td[:, :, 1:, :], X_fg[:, :, self.k:, :]], dim=2)
        return X_bg, X_fg, params
    @torch.no_grad()
    def inject(self, X, waveforms, params):
        X, y, params, psds = super().inject(X, waveforms, params)
        X = self.whitener(X, psds)
        if self.keep_last_n_seconds is not None:
            X_td = X[..., -self.keep_last_n_samples :].float().clone()
        else:
            X_td = X.float().clone()
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)
        X = self.topk_bin(X, _M, _B)
        if self.keep_last_n_seconds is not None:
            X = X[..., -self.keep_last_n_samples :].float()
        else:
            X = X.float()
        return torch.cat([X_td[:, :1, :], X[:, :self.k, :], X_td[:, 1:, :], X[:, self.k:, :]], dim=1), y, params
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)