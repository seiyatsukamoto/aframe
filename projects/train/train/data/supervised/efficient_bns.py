import math
import torch
from typing import Literal

from train.data.supervised.supervised import SupervisedAframeDataset, SPLSupervisedAframeDataset
#from train.data.supervised.deterministic.supervised_light import SupervisedAframeDataset

import torch.nn.functional as F
import numpy as np

from utils.efficient_heterodyne import ht_Whiten

class TopKHeterodyneTimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        heterodyne_phase_file: str,
        keep_last_n_seconds: float,
        batches_per_batch: int = 4,
        k: int = 5,
        ht_highpass: float | None = None,
        ht_lowpass: float | None = None,
        return_preds: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heterodyning_phase = torch.tensor(np.load(heterodyne_phase_file))
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.return_preds = return_preds
        self.batches_per_batch = batches_per_batch
        self.k = k
        self.keep_last_n_seconds = keep_last_n_seconds
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.whitener = ht_Whiten(
            kernel_length = self.hparams.kernel_length,
            fduration = self.hparams.fduration,
            sample_rate = self.hparams.sample_rate,
            heterodyning_phase = self.heterodyning_phase,
            keep_last_n_samples = self.keep_last_n_samples,
            lowpass = self.hparams.lowpass,
            highpass = self.hparams.highpass,
            ht_highpass = self.ht_highpass,
            ht_lowpass = self.ht_lowpass,
            k = self.k,
            return_preds = self.return_preds,
        )
    @torch.no_grad()
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
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        n = self.batches_per_batch
        step = X.shape[0] // n
        inputs, labels = None, None
        for i in range(n):
            X_w, y, psds = super().inject(X[step * i : step * (i + 1)], waveforms)
            X_w = self.whitener(X_w, psds)
            del psds
            if inputs is None:
                inputs = torch.empty((X.shape[0], *X_w.shape[1:]))
                labels = torch.empty((X.shape[0], *y.shape[1:]))
            inputs[step * i : step * (i + 1)] = X_w
            labels[step * i : step * (i + 1)] = y
            del X_w, y
        del X
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

class SPLTopkDataset(SPLSupervisedAframeDataset):
    def __init__(
        self,
        heterodyne_phase_file: str,
        keep_last_n_seconds: float,
        batches_per_batch: int = 4,
        k: int = 5,
        ht_highpass: float | None = None,
        ht_lowpass: float | None = None,
        return_preds: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heterodyning_phase = torch.tensor(np.load(heterodyne_phase_file))
        self.ht_highpass = ht_highpass
        self.ht_lowpass = ht_lowpass
        self.return_preds = return_preds
        self.batches_per_batch = batches_per_batch
        self.k = k
        self.keep_last_n_seconds = keep_last_n_seconds
        if self.keep_last_n_seconds is not None:
            self.keep_last_n_samples = int(
                self.keep_last_n_seconds * self.hparams.sample_rate
            )
    def build_transforms(self, *args, **kwargs):
        super().build_transforms(*args, **kwargs)
        self.whitener = ht_Whiten(
            kernel_length = self.hparams.kernel_length,
            fduration = self.hparams.fduration,
            sample_rate = self.hparams.sample_rate,
            heterodyning_phase = self.heterodyning_phase,
            keep_last_n_samples = self.keep_last_n_samples,
            lowpass = self.hparams.lowpass,
            highpass = self.hparams.highpass,
            ht_highpass = self.ht_highpass,
            ht_lowpass = self.ht_lowpass,
            k = self.k,
            return_preds = self.return_preds,
        )
    @torch.no_grad()
    def build_val_batches(self, background, signals, m1, m2):
        X_bg, X_inj, psds, m1, m2 = super().build_val_batches(background, signals, m1, m2)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)
        X_fg = torch.stack(X_fg)
        return X_bg, X_fg, m1, m2
    @torch.no_grad()
    def inject(self, X, waveforms=None):
        n = self.batches_per_batch
        step = X.shape[0] // n
        inputs, labels = None, None
        for i in range(n):
            X_w, y, psds = super().inject(X[step * i : step * (i + 1)], waveforms)
            X_w = self.whitener(X_w, psds)
            del psds
            if inputs is None:
                inputs = torch.empty((X.shape[0], *X_w.shape[1:]))
                labels = torch.empty((X.shape[0], *y.shape[1:]))
            inputs[step * i : step * (i + 1)] = X_w
            labels[step * i : step * (i + 1)] = y
            del X_w, y
        del X
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels