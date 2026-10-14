import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
#from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

import torch.nn.functional as F
import numpy as np
from train.data.heterodyne_whiten import ht_Whiten

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


class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file_1: str,
        chirp_mass_file_2: str,
        ht_lowpass: float,
        ht_highpass: float,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file_1 = chirp_mass_file_1
        self.chirp_mass_file_2 = chirp_mass_file_2
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
            chirp_mass_file=self.chirp_mass_file_1,
            return_type="time",
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
        )
        tmp = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass_file=self.chirp_mass_file_2,
            return_type="time",
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
        )
        self.ht_whitener = ht_Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            tmp.heterodyning_phase,
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
        X_test = self.ht_whitener(X, psds)
        X = self.whitener(X, psds)
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)

        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], X_test[..., -self.keep_last_n_samples :], y
        else:
            return X, y