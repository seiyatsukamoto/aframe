from typing import Callable, Optional
import torch
import numpy as np

from ..sampler import WaveformSampler
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
from ml4gw.gw import compute_observed_strain, get_ifo_geometry, compute_network_snr
from ml4gw.waveforms.conversion import bilby_spins_to_lalsim
from train.conversion import add_mass_params
from astropy.cosmology import Cosmology
from utils.cosmology import DEFAULT_COSMOLOGY

class RejectedSamplingGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        approximant: Callable,
        duration: float,
        f_min: float,
        f_ref: float,
        training_prior: Callable,
        acceptance_rate: float,
        highpass: float,
        lowpass: float,
        snr_threshold: float = 4,
        cosmology: Cosmology = DEFAULT_COSMOLOGY,
        **kwargs,
    ):
        """
        Rejection sampling for reatime waveform generation
        """
        super().__init__(*args, **kwargs)
        prior, detector_frame_prior = training_prior()
        self.prior = prior
        self.detector_frame_prior = detector_frame_prior
        self.approximant = approximant
        self.f_ref = f_ref
        self.waveform_generator = TimeDomainCBCWaveformGenerator(
            approximant,
            self.sample_rate,
            duration,
            f_min,
            f_ref,
            self.right_pad,
        )
        self.acceptance_rate = acceptance_rate
        tensors, vertices = get_ifo_geometry(*self.ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)
        self.snr_threshold = snr_threshold
        self.cosmology = DEFAULT_COSMOLOGY
        self.highpass = highpass
        self.lowpass = lowpass
    
    def convert_to_detector_frame(self, samples: dict[str, np.ndarray]):
        for key in ["mass_1", "mass_2", "chirp_mass", "total_mass"]:
            if key in samples:
                samples[key] = samples[key] * (1 + samples["redshift"])
        return samples
    
    def precessing_to_lalsimulation_parameters(self, parameters: dict[str, torch.Tensor]):
        mass_1 = parameters["mass_1"]
        incl, s1x, s1y, s1z, s2x, s2y, s2z = bilby_spins_to_lalsim(
            parameters["theta_jn"],
            parameters["phi_jl"],
            parameters["tilt_1"],
            parameters["tilt_2"],
            parameters["phi_12"],
            parameters["a_1"],
            parameters["a_2"],
            parameters["mass_1"],
            parameters["mass_2"],
            self.f_ref,
            torch.zeros(len(mass_1), device=mass_1.device),
        )

        parameters["s1x"] = s1x
        parameters["s1y"] = s1y
        parameters["s1z"] = s1z
        parameters["s2x"] = s2x
        parameters["s2y"] = s2y
        parameters["s2z"] = s2z
        parameters["inclination"] = incl
        return parameters
    
    def sample(self, N: int, psd: torch.Tensor):
        device = psd.device
        total_accepted = 0
        num_samples = int(N/self.acceptance_rate)
        _W = self.waveform_generator.duration * self.sample_rate
        _C, _ = psd.shape
        signals = torch.empty(N, _C, _W)
        while total_accepted < N:
            params = self.prior.sample(num_samples)
            if not self.detector_frame_prior:
                params = self.convert_to_detector_frame(params)
            for key in params:
                params[key] = torch.tensor(params[key], device = device)
            params = self.precessing_to_lalsimulation_parameters(params)
            params = add_mass_params(params)
            params['phic'] = params['phase']
            params['distance'] = self.cosmology.luminosity_distance(params['redshift']).value
            hc, hp = self.waveform_generator(**params)
            polarizations = {
                "cross": torch.Tensor(hc),
                "plus": torch.Tensor(hp),
            }
            projected = compute_observed_strain(
                torch.Tensor(params["dec"]),
                torch.Tensor(params["psi"]),
                torch.Tensor(params["ra"]),
                self.tensors,
                self.vertices,
                self.sample_rate,
                **polarizations,
            )
            snrs = compute_network_snr(
                projected,
                psd,
                self.sample_rate,
                self.highpass,
                self.lowpass)
            mask = snrs >= self.snr_threshold
            num_accepted = mask.sum()
            if num_accepted > N - total_accepted:
                total_left = N - total_accepted
                signals[total_accepted:total_left] = projected[mask][:total_left]
            else:
                signals[total_accepted:total_accepted+num_accepted] = projected[mask]
            total_accepted += num_accepted
            num_samples = int((N - total_accepted)/self.acceptance_rate)
        return signals
