from collections.abc import Callable

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


class SPLGenerator(CBCGenerator):
    def __init__(
        self,
        spl_waveform_file,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.spl_waveform_file = spl_waveform_file
        waveform_set = self.waveform_set_cls.read(spl_waveform_file)
        self.num_spl_waveforms = len(waveform_set)
    
    def get_val_waveforms(self, world_size, rank):
        """
        Returns validation waveforms for this device
        """
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        waveform_set = self.waveform_set_cls.read(self.val_waveform_file)
        val_waveforms = torch.Tensor(waveform_set.waveforms[start:stop])
        
        start, stop = self.get_slice_bounds(
            self.num_spl_waveforms, world_size, rank
        )
        waveform_set = self.waveform_set_cls.read(self.spl_waveform_file)
        spl_waveforms = torch.Tensor(waveform_set.waveforms[start:stop])
        mass_1 = torch.Tensor(waveform_set.parameters.mass_1[start:stop])
        mass_2 = torch.Tensor(waveform_set.parameters.mass_2[start:stop])
        mr = mass_2/mass_1
        cm = ((mass_1*mass_2)**(3/5))/((mass_1+mass_2)*(1/5))
        return val_waveforms, spl_waveforms, cm, mr