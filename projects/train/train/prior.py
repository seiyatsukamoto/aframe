from typing import Callable, Optional

import torch

from bilby.core.prior import Triangular
from torch.distributions import Distribution
from scipy.stats import gaussian_kde
import numpy as np
import h5py

def logit(data, bounds, inv_transform=False):
    if not inv_transform:
        transform = np.log((data-bounds[0])/(bounds[1]-data))
    elif inv_transform:
        transform = (bounds[1]*np.exp(data)+bounds[0])/(1+np.exp(data))
    return transform

class KDEDistribution(Distribution):
    def __init__(
        self,
        file,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with h5py.File(file, 'r') as f:
            m1 = f['parameters']['mass_1'][:]
            m2 = f['parameters']['mass_2'][:]
            z = f['parameters']['redshift'][:]
        m1 = m1/(1+z)
        m2 = m2/(1+z)
        cm = ((m1*m2)**(3/5))/((m1+m2)**(1/5))
        mr = m2/m1
        self.cm_limits = [min(np.log(cm))-0.0001, max(np.log(cm))+0.0001]
        self.mr_limits = [min(np.log(mr))-0.0001, max(np.log(mr))+0.0001]
        log_cm_t = logit(np.log(cm), [min(np.log(cm))-0.0001, max(np.log(cm))+0.0001])
        log_mr_t = logit(np.log(mr), [min(np.log(mr))-0.0001, max(np.log(mr))+0.0001])
        self.kernel = gaussian_kde([log_cm_t, log_mr_t], 
                                   bw_method = 0.1)
    def sample(self, X):
        samples = self.kernel.resample(X[0])
        cm = logit(samples[0], 
                   self.cm_limits, 
                   inv_transform=True)
        mr = logit(samples[1], 
                   self.mr_limits, 
                   inv_transform=True)
        cm = torch.exp(torch.Tensor(cm))
        mr = torch.exp(torch.Tensor(mr))
        mr = torch.clamp(mr, max = 0.999)
        return cm, mr

class TorchTriangular(Triangular):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
    
    def sample(self, *args, **kwargs):
        data = super().sample(*args, **kwargs)
        data = torch.Tensor(data)
        return data

class AframePrior:
    def __init__(
        self,
        priors: dict[str, torch.distributions.Distribution | TorchTriangular],
        conversion_function: Optional[Callable] = None,
        constraint_function: Optional[Callable] = None,
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            priors:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
            constraint_function:
                A callable that takes a dictionary of sampled parameters,
                discards any that don't satisfy the constraint, and returns
                the remaining parameters
        """
        super().__init__()
        self.priors = priors
        self.conversion_function = conversion_function or (lambda x: x)
        self.constraint_function = constraint_function or (lambda x: x)

    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Generates random samples from the prior

        Args:
            N: Number of samples to generate
            device: Device to place the samples
        """
        # sample parameters from prior
        parameters = {
            k: v.sample((N,)).to(device) for k, v in self.priors.items()
        }
        if 'log_mass_ratio' in parameters.keys():
            parameters['mass_ratio'] = 10**parameters['log_mass_ratio']
            _ = parameters.pop('log_mass_ratio')
        # perform any necessary conversions
        # to from sampled parameters to
        # waveform generation parameters
        parameters = self.conversion_function(parameters)

        # Discard any samples that don't meet the constraint
        parameters = self.constraint_function(parameters)

        keys = list(parameters.keys())
        while len(parameters[keys[0]]) < N:
            new_params = {
                k: v.sample((N,)).to(device) for k, v in self.priors.items()
            }
            new_params = self.conversion_function(parameters)
            new_params = self.constraint_function(parameters)
            parameters = {
                key: torch.cat([parameters[key], new_params[key]])
                for key in keys
            }

        return {k: v[:N] for k, v in parameters.items()}

    def log_prob(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the log probability of samples under the prior.
        TODO: This calculation is incorrect if a constraint function
        is specified. We don't use this function anywhere currently,
        so not too big a deal, but we'll need to fix this if we do.

        Args:
            samples:
                Dictionary where key is parameter and
                value is tensor of samples
        """

        first = samples[list(samples.keys())[0]]
        log_probs = torch.ones(len(first), device=first.device)
        for param, tensor in samples.items():
            log_probs += self.priors[param].log_prob(tensor).to(first.device)
        return log_probs

class KDEPrior(AframePrior):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        A class for sampling parameters from a prior distribution

        Args:
            priors:
                A dictionary of parameter samplers that take an integer N
                and return a tensor of shape (N, ...) representing
                samples from the prior distribution
            conversion_function:
                A callable that takes a dictionary of sampled parameters
                and returns a dictionary of waveform generation parameters
            constraint_function:
                A callable that takes a dictionary of sampled parameters,
                discards any that don't satisfy the constraint, and returns
                the remaining parameters
        """
        super().__init__(*args, **kwargs)
    
    def __call__(
        self,
        N: int,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Generates random samples from the prior

        Args:
            N: Number of samples to generate
            device: Device to place the samples
        """
        # sample parameters from prior
        parameters = {}
        for k, v in self.priors.items():
            if k == 'cm_mr':
                samples = v.sample((N,))
                parameters['chirp_mass'] = samples[0].to(device)
                parameters['mass_ratio'] = samples[1].to(device)
            else:
               parameters[k] = v.sample((N,)).to(device)
        
        parameters = self.conversion_function(parameters)
        return {k: v[:N] for k, v in parameters.items()}

class ParameterTransformer(torch.nn.Module):
    """
    Helper class for applying preprocessing
    transformations to inference parameters
    """

    def __init__(self, **transforms: Callable):
        super().__init__()
        self.transforms = transforms

    def forward(
        self,
        parameters: dict[str, torch.Tensor],
    ):
        # transform parameters
        transformed = {k: v(parameters[k]) for k, v in self.transforms.items()}
        # update parameter dict
        parameters.update(transformed)
        return parameters
