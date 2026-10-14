import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset
#from ml4gw.transforms.heterodyne_from_file import Heterodyne_from_file as Heterodyne

import torch.nn.functional as F
import numpy as np
from utils.heterodyne_whiten import ht_Whiten

from ml4gw.constants import MTSUN_SI

import h5py
class HeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        keep_last_n_seconds: float = None,
        batches_per_batch: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.chirp_mass_file = chirp_mass_file
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        self.batches_per_batch = batches_per_batch
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.whitener = ht_Whiten(
            self.hparams.kernel_length,
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.hparams.lowpass,
            self.hparams.highpass,
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
        X = self.whitener(X, psds)
        return X, y


class TopKHeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        k: int,
        keep_last_n_seconds: float = None,
        batches_per_batch: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.chirp_mass_file = chirp_mass_file
        self.batches_per_batch = batches_per_batch
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.keep_last_n_seconds = keep_last_n_seconds
        self.k = k
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
        

    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.whitener = ht_Whiten(
            self.hparams.kernel_length,
            self.hparams.fduration,
            self.hparams.sample_rate,
            torch.tensor(np.load(self.chirp_mass_file)),
            self.keep_last_n_samples,
            self.hparams.highpass,
            self.hparams.lowpass,
            self.ht_highpass,
            self.ht_lowpass,
            self.batches_per_batch
        )

    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg_td, X_bg_ht = self.whitener(X_bg, psds)
        _B, _M, _T = X_bg_ht.shape
        X_bg_ht = self.topk_bin(X_bg_ht, _M//2, _B)
        
        # whiten each view of injections
        X_fg_td = []
        X_fg_ht = []
        for inj in X_inj:
            inj_td, inj_ht = self.whitener(inj, psds)
            X_fg_td.append(inj_td)
            X_fg_ht.append(self.topk_bin(inj_ht, _M//2, _B))
        X_fg_td = torch.stack(X_fg_td)
        X_fg_ht = torch.stack(X_fg_ht)
        return torch.cat([X_bg_td, X_bg_ht], dim=1), torch.cat([X_fg_td, X_fg_ht], dim=2)
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X_td, X_ht = self.whitener(X, psds, validating = False)
        _B, _M, _T = X_ht.shape
        X_ht = self.topk_bin(X_ht, _M//2, _B)
        return torch.cat([X_td, X_ht], dim=1), y
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)

#from ml4gw.utils.slicing import sample_kernels
#
#class test(SupervisedAframeDataset):
#    def __init__(
#        self,
#        chirp_mass_file: str,
#        ht_lowpass: float,
#        ht_highpass: float,
#        k: int,
#        keep_last_n_seconds: float = None,
#        batches_per_batch: int = 2,
#        *args,
#        **kwargs,
#    ):
#        super().__init__(*args, **kwargs)
#        self.chirp_mass_file = chirp_mass_file
#        self.batches_per_batch = batches_per_batch
#        self.ht_highpass = ht_highpass
#        self.ht_lowpass = ht_lowpass
#        self.keep_last_n_seconds = keep_last_n_seconds
#        self.k = k
#        if self.keep_last_n_seconds is not None:
#            self.keep_last_n_samples = int(
#                self.keep_last_n_seconds * self.hparams.sample_rate
#            )
#        
#
#    def build_transforms(self, *args, **kwargs):
#        super().build_transforms(*args, **kwargs)
#        self.whitener = ht_Whiten(
#            self.hparams.kernel_length,
#            self.hparams.fduration,
#            self.hparams.sample_rate,
#            torch.tensor(np.load(self.chirp_mass_file)),
#            self.keep_last_n_samples,
#            self.hparams.highpass,
#            self.hparams.lowpass,
#            self.ht_highpass,
#            self.ht_lowpass,
#            self.batches_per_batch
#        )
#        print(self.hparams.kernel_length,
#              self.hparams.fduration,
#              self.hparams.sample_rate,
#              torch.tensor(np.load(self.chirp_mass_file)),
#              self.keep_last_n_samples,
#              self.hparams.highpass,
#              self.hparams.lowpass,
#              self.ht_highpass,
#              self.ht_lowpass,
#              self.batches_per_batch)
#    
#    @torch.no_grad()
#    def build_val_batches(self, background, signals):
#        X_bg, X_inj, psds = super().build_val_batches(background, signals)
#        X_bg_td, X_bg_ht = self.whitener(X_bg, psds, validating = True)
#        _B, _M, _T = X_bg_ht.shape
#        X_bg_ht, pred = self.topk_bin(X_bg_ht, _M//2, _B)
#        
#        # whiten each view of injections
#        X_fg_td = []
#        X_fg_ht = []
#        for inj in X_inj:
#            inj_td, inj_ht = self.whitener(inj, psds, validating = True)
#            X_fg_td.append(inj_td)
#            tmp, pred = self.topk_bin(inj_ht, _M//2, _B)
#            X_fg_ht.append(tmp)
#        X_fg_td = torch.stack(X_fg_td)
#        X_fg_ht = torch.stack(X_fg_ht)
#        return torch.cat([X_bg_td, X_bg_ht], dim=1), torch.cat([X_fg_td, X_fg_ht], dim=2), pred
#    def on_after_batch_transfer(self, batch, _):
#        if self.trainer.training:
#            if self.waveforms_from_disk:
#                [batch], waveforms = batch
#                batch = self.inject(batch, waveforms)
#            else:
#                [batch] = batch
#                batch = self.inject(batch)
#        elif self.trainer.validating or self.trainer.sanity_checking:
#            [background, _, timeslide_idx], [signals] = batch
#            shift = self.timeslides[timeslide_idx].shift_size
#            X_bg, X_fg, pred = self.build_val_batches(background, signals)
#            batch = (shift, X_bg, X_fg, pred)
#        return batch
#    
#    @torch.no_grad()
#    def inject(self, X, waveforms=None):
#        X, psds = self.psd_estimator(X)
#        X = self.inverter(X)
#        X = self.reverser(X)
#        rvs = torch.rand(size=X.shape[:1], device=X.device)
#        mask = rvs < self.sample_prob
#        dec, psi, phi = self.sample_extrinsic(X[mask])
#        N = mask.sum().item()
#        idx = torch.arange(waveforms.shape[0]) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#        waveforms = waveforms[idx].to(X.device).float()
#        hc, hp = waveforms[:, 0], waveforms[:, 1]
#        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
#        responses = self.projector(
#            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
#        )
#        kernels = sample_kernels(
#            responses, kernel_size=X.size(-1), coincident=True
#        )
#        swap_indices = mute_indices = []
#        idx = torch.where(mask)[0]
#        if self.swapper is not None:
#            kernels, swap_indices = self.swapper(kernels)
#        if self.muter is not None:
#            kernels, mute_indices = self.muter(kernels)
#        X[mask] += kernels
#        mask[idx[swap_indices]] = 0
#        mask[idx[mute_indices]] = 0
#        y = torch.zeros((X.size(0), 1), device=X.device)
#        y[mask] += 1
#        X_td, X_ht = self.whitener(X, psds, validating = False)
#        _B, _M, _T = X_ht.shape
#        X_ht, pred = self.topk_bin(X_ht, _M//2, _B) #Assuming num_ifos = 2
#        return torch.cat([X_td, X_ht], dim=1), y, pred
#    
#    def topk_bin(self, X, _M, _B):
#        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
#        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
#        pred = hl.topk(self.k, dim = -1)[1]
#        pred = torch.concat([pred, pred+_M], dim = -1)
#        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
#        return torch.gather(X, dim=1, index=pred), pred