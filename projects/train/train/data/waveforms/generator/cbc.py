from typing import Callable

import torch
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator

from .generator import WaveformGenerator


class CBCGenerator(WaveformGenerator):
    def __init__(
        self,
        *args,
        approximant: Callable,
        duration: float,
        f_min: float,
        f_ref: float,
        **kwargs,
    ):
        """
        A lightweight wrapper around
        `ml4gw.waveforms.generator.TimeDomainCBCWaveformGenerator`
        to make it compatible with
        `aframe.train.train.data.waveforms.generator.WaveformGenerator`.
        Args:
            *args:
                Positional arguments passed to
                `aframe.train.train.data.waveforms.generator.WaveformGenerator`
            approximant:
                A callable that takes parameter tensors
                and returns the cross and plus polarizations.
                For example, `ml4gw.waveforms.IMRPhenomD()`
            duration:
                Duration of the waveform in seconds
            f_min:
                Lowest frequency at which waveform signal content
                is generated
            f_ref:
                Reference frequency
            **kwargs:
                Keyword arguments passed to
                `aframe.train.train.data.waveforms.generator.WaveformGenerator`
        """
        super().__init__(*args, **kwargs)

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

    def forward(self, **parameters) -> torch.Tensor:
        hc, hp = self.waveform_generator(**parameters)
        waveforms = torch.stack([hc, hp], dim=1)
        hc, hp = waveforms.transpose(1, 0)
        return hc.float(), hp.float()

class SPLGenerator(CBCGenerator): #Takes in log mass ratio instead of mass ratio
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def get_val_waveforms(self, world_size, rank):
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        waveform_set = self.waveform_set_cls.read(self.val_waveform_file)
        return (torch.Tensor(waveform_set.waveforms[start:stop]), 
                torch.Tensor(waveform_set.mass_1[start:stop]), 
                torch.Tensor(waveform_set.mass_2[start:stop]))