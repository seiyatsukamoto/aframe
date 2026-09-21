import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from ledger.events import EventSet, RecoveredInjectionSet
from ledger.injections import InjectionParameterSet
from utils.cosmology import DEFAULT_COSMOLOGY, get_astrophysical_volume
from priors.priors import end_o3_ratesandpops
from priors.priors import log_normal_masses
import itertools
from collections import defaultdict
from typing import Optional

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAUROC

from utils import x_per_y

from ledger.events import EventSet
class Postprocessor:
    def __init__(
        self,
        inference_sampling_rate: float,
        integration_window_length: float,
        cluster_window_length: float,
    ) -> None:
        """
        Condensed Preprocessor from infer used for calculating background
        """
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_window_size = int((inference_sampling_rate * integration_window_length) + 1)
        self.cluster_window_size = int(inference_sampling_rate * cluster_window_length)
    
    def integrate(self, y: np.ndarray) -> np.ndarray:
        window_size = self.integration_window_size
        window = np.ones((window_size,)) / window_size
        integrated = np.convolve(y, window, mode="full")
        return integrated[: -window_size + 1]
    
    def cluster(self, y) -> EventSet:
        window_size = int(self.cluster_window_size // 2)
        i = np.argmax(y[:window_size])
        events = []
        while i < len(y):
            val = y[i]
            window = y[i + 1 : i + 1 + window_size]
            if (val < window).any():
                i += np.argmax(window) + 1
            else:
                events.append(val)
                i += window_size + 1
        events = np.array(events)
        return events
    
    def __call__(self, y: Optional[np.ndarray] = None) -> EventSet:
        y = self.integrate(y)
        y = self.cluster(y)
        return y

class TimeSlideSV(Metric):
    def __init__(
        self, 
        rejected_params: Path, 
        inference_sampling_rate: float,
        integration_window_length: float,
        cluster_window_length: float, 
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.add_state("shifts", default=[])
        self.add_state("background", default=[])
        self.add_state("foreground", default=[])
        self.add_state("mass_1", default=[])
        self.add_state("mass_2", default=[])
        
        rejected_params = InjectionParameterSet.read(rejected_params)
        for mass in ['mass_1', 'mass_2']:
            val = getattr(rejected_params, mass)
            setattr(rejected_params, mass, val / (1 + rejected_params.redshift))
        self.rejected_params = rejected_params
        self.source, _ = end_o3_ratesandpops(DEFAULT_COSMOLOGY)
        self.source_rejected_probs = self.get_prob(self.source, self.rejected_params.mass_1, self.rejected_params.mass_2)
        zprior = self.source["redshift"]
        zmin, zmax = zprior.minimum, zprior.maximum
        decprior = self.source["dec"]
        decrange = (decprior.minimum, decprior.maximum)
        self.v0 = get_astrophysical_volume(zmin, zmax, DEFAULT_COSMOLOGY, decrange)/10**9

        self.postprocessor = Postprocessor(inference_sampling_rate = inference_sampling_rate, 
                                           integration_window_length = integration_window_length, 
                                           cluster_window_length = cluster_window_length)
        self.mass_combos = [[35, 35], [35, 20], [20, 20], [20,10]]
        self.sigma = .1

    def update(
        self, shift: int, background: torch.Tensor, foreground: torch.Tensor, mass_1: torch.Tensor, mass_2: torch.Tensor #source frame
    ) -> None:
        self.shifts.append(torch.Tensor([shift]).to(background.device))
        self.background.append(background)
        self.foreground.append(foreground)
        self.mass_1.append(mass_1)
        self.mass_2.append(mass_2)

    def get_prob(self, prior, mass_1, mass_2):
        sample = {"mass_1": mass_1, "mass_2": mass_2}
        return prior.prob(sample, axis=0)

    def compute_sv(self, threshold, detection_statistics, weights):
        mask = detection_statistics >= threshold
        mus = (weights * mask).sum(-1, keepdims=True)
        var_summands = weights * (mask - mus)
        stds = (var_summands**2).sum(-1) ** 0.5
        return mus[:, 0], stds
    
    def compute(self):
        foreground, background, mass_1, mass_2 = [], defaultdict(list), [], []
        for i, bg, fg, m1, m2 in zip(
            self.shifts, self.background, self.foreground, self.mass_1, self.mass_2, strict=True
        ):
            foreground.append(fg)
            background[i.item()].append(bg)
            mass_1.append(m1)
            mass_2.append(m2)
        
        foreground = torch.cat(foreground)
        mass_1 = torch.cat(mass_1)
        mass_2 = torch.cat(mass_2)
        for key in background.keys():
            background[key] = torch.cat(background[key]).squeeze(1)
        
        for key in background.keys():
            background[key] = self.postprocessor(background[key])
        
        background = np.concatenate([background[key] for key in background.keys()])
        foreground = foreground.numpy()
        mass_1 = mass_1.numpy()
        mass_2 = mass_2.numpy()

        #convert masses to redshifted masses
        source_probs = self.get_prob(self.source, mass_1, mass_2)
        thresholds = np.sort(background)[::-1]
        weights = np.zeros((len(self.mass_combos), len(source_probs)))
        for i, combo in enumerate(self.mass_combos):
            prior, _ = log_normal_masses(
                *combo, sigma=self.sigma, cosmology=DEFAULT_COSMOLOGY
            )
            prob = self.get_prob(prior, mass_1, mass_2)
            rejected_prob = self.get_prob(prior, self.rejected_params.mass_1, self.rejected_params.mass_2)
            weight = prob / source_probs
            rejected_weights = rejected_prob / self.source_rejected_probs
            norm = weight.sum() + rejected_weights.sum()
            weight /= norm
            weights[i] = weight
        aframe_sv = np.empty((len(weights), len(thresholds)))
        aframe_err = np.empty((len(weights), len(thresholds)))
        for i, t in enumerate(thresholds):
            mu, _ = self.compute_sv(t, foreground, weights)
            aframe_sv[:, i] = mu
        return aframe_sv * self.v0