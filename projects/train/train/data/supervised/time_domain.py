import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
#from ml4gw.transforms import Heterodyne
#from ml4gw.transforms import Heterodyne_from_dir as Heterodyne
from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

import numpy as np
import torch.nn.functional as F
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
    """

    def __init__(
        self,
        #chirp_mass_file: str,
        phase_dir: str,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.phase_dir = phase_dir
        
        #self.chirp_mass_grid = torch.Tensor(np.load(chirp_mass_file))
        
        #self.chirp_mass_grid = self._create_chirp_mass_grid(
        #    chirp_mass_low,
        #    chirp_mass_high,
        #    num_chirp_masses,
        #    chirp_mass_spacing,
        #)

        self.keep_last_n_seconds = keep_last_n_seconds

        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        #self.heterodyne_transform = Heterodyne(
        #    sample_rate=self.hparams.sample_rate,
        #    kernel_length=self.hparams.kernel_length,
        #    chirp_mass=self.chirp_mass_grid,
        #    return_type="time",
        #)
        self.heterodyne_transform = Heterodyne(
            sample_rate=self.hparams.sample_rate,
            kernel_length=self.hparams.kernel_length,
            chirp_mass_file=self.phase_dir,
            return_type="time",
        )

    #def _create_chirp_mass_grid(
    #    self,
    #    chirp_mass_low: float,
    #    chirp_mass_high: float,
    #    num_chirp_masses: int,
    #    chirp_mass_spacing: Literal["linear", "log"],
    #) -> torch.Tensor:
    #    if chirp_mass_spacing == "linear":
    #        return torch.linspace(
    #            chirp_mass_low, chirp_mass_high, num_chirp_masses
    #        )
    #    elif chirp_mass_spacing == "log":
    #        return torch.logspace(
    #            math.log10(chirp_mass_low),
    #            math.log10(chirp_mass_high),
    #            num_chirp_masses,
    #        )
    #    else:
    #        raise ValueError(
    #            f"Invalid chirp mass spacing: {chirp_mass_spacing}"
    #        )

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
        X = self.whitener(X, psds)
        #y = X.clone() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)

        if self.keep_last_n_seconds is not None:
            return X[..., -self.keep_last_n_samples :], y
        else:
            return X, y



class TopKHeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        k: int,
        phase_dir: str,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.phase_dir = phase_dir
        self.k = k
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
            phase_dir=self.phase_dir,
            return_type="time",
        )
        
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        X_bg_td = X_bg.detach().clone()
        X_bg = self.heterodyne_transform(X_bg)
        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
        X_bg = X_bg.reshape(_B_bg, _C_bg * _M_bg, _T_bg)
        X_bg = self.topk_bin(X_bg, _M_bg, _B_bg)
        X_bg = torch.cat([X_bg_td, X_bg], dim=1)
        # whiten each view of injections
        X_fg = []
        X_fg_td = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg_td.append(inj.detach().clone())
            inj = self.heterodyne_transform(inj)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        X_fg_td = torch.stack(X_fg_td)
        _V_fg, _B_fg, _C_fg, _M_fg, _T_fg = X_fg.shape
        X_fg = X_fg.view(_V_fg*_B_fg, _C_fg * _M_fg, _T_fg)
        X_fg = self.topk_bin(X_fg, _M_fg, _V_fg*_B_fg)
        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg * self.k, _T_fg)
        X_fg = torch.cat([X_fg_td, X_fg], dim=2)
        if self.keep_last_n_seconds is not None:
            return X_bg[..., -self.keep_last_n_samples :].float(), X_fg[
                ..., -self.keep_last_n_samples :
            ].float()
        else:
            return X_bg.float(), X_fg.float()

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        if self.keep_last_n_seconds is not None:
            X_td = X[..., -self.keep_last_n_samples :].float().detach().clone()
        else:
            X_td = X.float().detach().clone()
        
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)
        X = self.topk_bin(X, _M, _B)
        if self.keep_last_n_seconds is not None:
            X = X[..., -self.keep_last_n_samples :].float()
        else:
            X = X.float()
        
        return torch.cat([X_td, X], dim=1), y

    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        
        h = torch.max(pooled[:, :_M], dim = -1)[0]
        h = h/torch.max(h, dim = -1)[0].unsqueeze(1)
        
        l = torch.max(pooled[:, _M:], dim = -1)[0]
        l = l/torch.max(l, dim = -1)[0].unsqueeze(1)
        
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        hl = hl/torch.max(hl, dim = -1)[0].unsqueeze(1)
        
        h_l_hl = torch.stack([h, l, hl])
        
        idx = torch.argmin(torch.median(torch.stack([h, l, hl]), dim = -1)[0], dim = 0) # select which detector to use
        pred = h_l_hl.topk(self.k, dim = -1)[1][idx, torch.arange(_B)] #do the topk using the idx
        pred = torch.concat([pred, pred+_M], dim = -1) #get both h and l channels
        return X[torch.arange(_B).unsqueeze(-1), pred]