import math
import torch
from typing import Literal

from train.data.supervised.all_sky_supervised import AllSkySupervised

import torch.nn.functional as F
import numpy as np

from ml4gw.constants import MTSUN_SI

class TimeDomainSupervisedAframeDataset(AllSkySupervised):
    def __init__(
        self,
        *args,
        keep_last_n_seconds: float = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(keep_last_n_seconds*self.hparams.sample_rate)
        else:
            self.keep_last_n_samples = None
        
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        if self.keep_last_n_samples is not None:
            X_bg = X_bg[..., -self.keep_last_n_samples:]
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            if self.keep_last_n_samples is not None:
                X_fg.append(inj[..., -self.keep_last_n_samples:])
            else:
                X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        if self.keep_last_n_samples is not None:
            X = X[..., -self.keep_last_n_samples:]
        return X, y


from train.data.heterodyne_whiten import ht_Whiten
from utils.s3 import open_file
class LowHighMassFusion(AllSkySupervised):
    def __init__(
        self,
        *args,
        chirp_mass_file: str,
        ht_lowpass: float,
        ht_highpass: float,
        k: int,
        models: list[str],
        keep_last_n_seconds: float = None,
        batches_per_batch: int = 2,
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
        self.models = models
    
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
        graphs = []
        for model in self.models:
            with open_file(model, "rb") as f:
                graph = torch.jit.load(f, map_location="cpu")
            graph.eval()
            graphs.append(graph)
        
        self.graphs = torch.nn.ModuleList(graphs)
    
    @torch.no_grad()
    def compute_low(self, X):
        X = self.graphs[0].time_domain_resnet.conv1(X)
        X = self.graphs[0].time_domain_resnet.bn1(X)
        X = self.graphs[0].time_domain_resnet.relu(X)
        X = self.graphs[0].time_domain_resnet.maxpool(X)
        for layer in self.graphs[0].time_domain_resnet.residual_layers.children():
            X = layer(X)
        X = self.graphs[0].time_domain_resnet.avgpool(X)
        X = X.squeeze(-1)
        return X
    @torch.no_grad()
    def compute_high(self, X):
        X = self.graphs[1].conv1(X)
        X = self.graphs[1].bn1(X)
        X = self.graphs[1].relu(X)
        X = self.graphs[1].maxpool(X)
        for layer in self.graphs[1].residual_layers.children():
            X = layer(X)
        X = self.graphs[1].avgpool(X)
        X = X.squeeze(-1)
        return X
        
    @torch.no_grad()
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg_td, X_bg_ht = self.whitener(X_bg, psds, validating = True)
        _B, _M, _T = X_bg_ht.shape
        X_bg_ht = self.topk_bin(X_bg_ht, _M//2, _B)
        
        # whiten each view of injections
        X_fg_td = []
        X_fg_ht = []
        for inj in X_inj:
            inj_td, inj_ht = self.whitener(inj, psds, validating = True)
            X_fg_td.append(inj_td)
            X_fg_ht.append(self.topk_bin(inj_ht, _M//2, _B))
        X_fg_td = torch.stack(X_fg_td)
        X_fg_ht = torch.stack(X_fg_ht)
        
        bg_low = self.compute_low(torch.cat([X_bg_td, X_bg_ht], dim=1))
        bg_high = self.compute_high(X_bg_td)

        fg_low = []
        for view in torch.cat([X_fg_td, X_fg_ht], dim=2):
            fg_low.append(self.compute_low(view))
        
        fg_high = []
        for view in X_fg_td:
            fg_high.append(self.compute_high(view))
        
        fg_low = torch.stack(fg_low)
        fg_high = torch.stack(fg_high)
        return torch.cat([bg_low, bg_high], dim = 1), torch.cat([fg_low, fg_high], dim = 2)
    
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X_td, X_ht = self.whitener(X, psds, validating = False)
        _B, _M, _T = X_ht.shape
        X_ht = self.topk_bin(X_ht, _M//2, _B) #Assuming num_ifos = 2
        
        low = self.compute_low(torch.cat([X_td, X_ht], dim=1))
        high = self.compute_high(X_td)
        return torch.cat([low, high], dim = 1), y, torch.cat([X_td, X_ht], dim=1), X, psds#!!!!!!!!!!!!!!!!!!
    
    def topk_bin(self, X, _M, _B):
        pooled = F.avg_pool1d(X.abs(), kernel_size = 31, stride = 5, padding = 0)
        hl = torch.max(pooled[:, :_M]*pooled[:, _M:], dim = -1)[0]
        pred = hl.topk(self.k, dim = -1)[1]
        pred = torch.concat([pred, pred+_M], dim = -1)
        pred = pred.unsqueeze(-1).expand(-1, -1, X.size(-1))
        return torch.gather(X, dim=1, index=pred)
    