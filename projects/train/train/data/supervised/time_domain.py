import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
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
        ht_lowpass: float,
        ht_highpass: float,
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
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
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
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
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
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
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
        return X_bg, X_fg
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
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
        
        return torch.cat([X_td[:, :1, :], X[:, :self.k, :], X_td[:, 1:, :], X[:, self.k:, :]], dim=1), y
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)


class NeighborhoodTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        phase_file: str,
        offsets: list[int],
        ht_highpass: float,
        ht_lowpass: float,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.phase_file = phase_file
        self.offsets = torch.tensor(offsets)
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
            chirp_mass_file=self.phase_file,
            return_type="time",
            highpass = self.ht_highpass,
            lowpass = self.ht_lowpass,
        )
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        if self.keep_last_n_seconds is not None:
            X_bg_td = X_bg[..., -self.keep_last_n_samples :].float().clone()
        else:
            X_bg = X_bg.float().clone()
        
        X_bg = self.heterodyne_transform(X_bg)
        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
        X_bg = X_bg.reshape(_B_bg, _C_bg * _M_bg, _T_bg)
        X_bg = self.nbhd_bin(X_bg, _M_bg, _B_bg)
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
        X_fg = self.nbhd_bin(X_fg, _M_fg, _V_fg*_B_fg)
        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg * len(self.offsets), _T_fg)
        if self.keep_last_n_seconds is not None:
            X_bg = X_bg[..., -self.keep_last_n_samples :].float()
            X_fg = X_fg[..., -self.keep_last_n_samples :].float()
        else:
            X_bg = X_bg.float()
            X_fg = X_fg.float()
        
        X_bg = torch.cat([X_bg_td[:, :1, :], X_bg[:, :len(self.offsets), :], X_bg_td[:, 1:, :], X_bg[:, len(self.offsets):, :]], dim=1)
        X_fg = torch.cat([X_fg_td[:, :, :1, :], X_fg[:, :, :len(self.offsets), :], X_fg_td[:, :, 1:, :], X_fg[:, :, len(self.offsets):, :]], dim=2)
        return X_bg, X_fg
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        if self.keep_last_n_seconds is not None:
            X_td = X[..., -self.keep_last_n_samples :].float().clone()
        else:
            X_td = X.float().clone()
        
        X = self.heterodyne_transform(X)
        _B, _C, _M, _T = X.shape
        X = X.view(_B, _C * _M, _T)
        X = self.nbhd_bin(X, _M, _B)
        if self.keep_last_n_seconds is not None:
            X = X[..., -self.keep_last_n_samples :].float()
        else:
            X = X.float()
        
        return torch.cat([X_td[:, :1, :], X[:, :len(self.offsets), :], X_td[:, 1:, :], X[:, len(self.offsets):, :]], dim=1), y
    
    @torch.no_grad()
    def nbhd_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        idx = torch.argmax(hl, dim = -1)
        idx = idx.unsqueeze(1).repeat(1, 5)+self.offsets.to(idx.device)
        idx = torch.clip(idx, min=0, max=99)
        
        idx = torch.concat([idx, idx+_M], dim = -1) #get both h and l channels
        return X[torch.arange(_B).unsqueeze(-1), idx]



#class NeighborhoodTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
#    def __init__(
#        self,
#        phase_file: str,
#        offsets: list[int],
#        keep_last_n_seconds: float = None,
#        *args,
#        **kwargs,
#    ):
#        super().__init__(*args, **kwargs)
#
#        self.phase_file = phase_file
#        self.offsets = torch.tensor(offsets)
#        self.keep_last_n_seconds = keep_last_n_seconds
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * self.hparams.sample_rate
#            )
#
#    def build_transforms(self, *args, **kwargs):
#        super().build_transforms(*args, **kwargs)
#        self.heterodyne_transform = Heterodyne(
#            sample_rate=self.hparams.sample_rate,
#            kernel_length=self.hparams.kernel_length,
#            chirp_mass_file=self.phase_file,
#            return_type="time",
#        )
#    
#    @torch.no_grad()
#    def build_val_batches(self, background, signals):
#        X_bg, X_inj, psds = super().build_val_batches(background, signals)
#        X_bg = self.whitener(X_bg, psds)
#        
#        X_bg = self.heterodyne_transform(X_bg)
#        _B_bg, _C_bg, _M_bg, _T_bg = X_bg.shape
#        X_bg = X_bg.reshape(_B_bg, _C_bg * _M_bg, _T_bg)
#        X_bg = self.nbhd_bin(X_bg, _M_bg, _B_bg)
#        # whiten each view of injections
#        X_fg = []
#        for inj in X_inj:
#            inj = self.whitener(inj, psds)
#            inj = self.heterodyne_transform(inj)
#            X_fg.append(inj)
#        X_fg = torch.stack(X_fg)
#        _V_fg, _B_fg, _C_fg, _M_fg, _T_fg = X_fg.shape
#        X_fg = X_fg.view(_V_fg*_B_fg, _C_fg * _M_fg, _T_fg)
#        X_fg = self.nbhd_bin(X_fg, _M_fg, _V_fg*_B_fg)
#        X_fg = X_fg.view(_V_fg, _B_fg, _C_fg * len(self.offsets), _T_fg)
#        if self.keep_last_n_seconds is not None:
#            return X_bg[..., -self.keep_last_n_samples :].float(), X_fg[
#                ..., -self.keep_last_n_samples :
#            ].float()
#        else:
#            return X_bg, X_fg
#    
#    @torch.no_grad()
#    def inject(self, X, waveforms=None):
#        X, y, psds = super().inject(X, waveforms)
#        X = self.whitener(X, psds)
#        
#        X = self.heterodyne_transform(X)
#        _B, _C, _M, _T = X.shape
#        X = X.view(_B, _C * _M, _T)
#        X = self.nbhd_bin(X, _M, _B)
#        if self.keep_last_n_seconds is not None:
#            return X[..., -self.keep_last_n_samples :].float(), y
#        else:
#            return X.float(), y
#    
#    @torch.no_grad()
#    def nbhd_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        idx = torch.argmax(hl, dim = -1)
#        idx = idx.unsqueeze(1).repeat(1, 5)+self.offsets.to(idx.device)
#        idx = torch.clip(idx, min=0, max=99)
#        
#        idx = torch.concat([idx, idx+_M], dim = -1) #get both h and l channels
#        return X[torch.arange(_B).unsqueeze(-1), idx]