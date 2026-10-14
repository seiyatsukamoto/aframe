import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
#from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

import torch.nn.functional as F
import numpy as np
from train.data.heterodyne_whiten import ht_Whiten

from ml4gw.constants import MTSUN_SI


class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        batches_per_batch: int,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file = chirp_mass_file
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        self.batches_per_batch

        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.ht_whitener = ht_Whiten(
            self.hparams.kernel_length,
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.ht_highpass,
            self.ht_lowpass,
            self.batches_per_batch
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
        X = self.ht_whitener(X, psds)
        return X, y


class TopKHeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        batches_per_batch: int,
        k: int,
        keep_last_n_seconds: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file = chirp_mass_file
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        self.batches_per_batch = batches_per_batch
        self.k = k
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.ht_whitener = ht_Whiten(
            self.hparams.kernel_length,
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.ht_highpass,
            self.ht_lowpass,
            self.batches_per_batch
        )
    
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg_td, X_bg_ht = self.ht_whitener(X_bg, psds)
        _B, _M, _T = X_bg_ht.shape
        X_bg_ht = self.topk_bin(X_bg_ht, _M//2, _B)
        
        # whiten each view of injections
        X_fg_td = []
        X_fg_ht = []
        for inj in X_inj:
            inj_td, inj_ht = self.ht_whitener(inj, psds)
            X_fg_td.append(inj_td)
            X_fg_ht.append(inj_ht)
        X_fg_td = torch.stack(X_fg_td)
        X_fg_ht = torch.stack(X_fg_ht)
        
        _V, _B, _M, _T = X_fg_ht.shape
        X_fg_ht = X_fg_ht.view(_V*_B, _M, _T)
        X_fg_ht = self.topk_bin(X_fg_ht, _M//2, _V*_B)
        X_fg_ht = X_fg_ht.view(_V, _B, 2 * self.k, _T)
        return torch.cat([X_bg_td, X_bg_ht], dim=1), torch.cat([X_fg_td, X_fg_ht], dim=2)
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X_td, X_ht = self.ht_whitener(X, psds)
        _B, _M, _T = X_ht.shape
        X_ht, pred = self.topk_bin(X_ht, _M//2, _B) #Assuming num_ifos = 2
        return torch.cat([X_td, X_ht], dim=1), y
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)