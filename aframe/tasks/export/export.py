import os

import law
import luigi
from luigi.util import inherits

from aframe.base import AframeSingularityTask
from aframe.config import paths
from aframe.parameters import PathParameter
from aframe.tasks.export.target import ModelRepositoryTarget

import numpy as np

class ExportParams(law.Task):
    fduration = luigi.FloatParameter()
    kernel_length = luigi.FloatParameter()
    inference_sampling_rate = luigi.FloatParameter()
    sample_rate = luigi.FloatParameter()
    batch_file = luigi.Parameter(default="")
    streams_per_gpu = luigi.IntParameter()
    aframe_instances = luigi.IntParameter()
    preproc_instances = luigi.IntParameter()
    clean = luigi.BoolParameter()
    batch_size = luigi.IntParameter()
    psd_length = luigi.FloatParameter()
    highpass = luigi.FloatParameter()
    lowpass = luigi.OptionalFloatParameter(default="")
    q = luigi.OptionalFloatParameter(default="")
    fftlength = luigi.OptionalFloatParameter(default="")
    ifos = luigi.ListParameter()
    repository_directory = PathParameter(
        default=paths().results_dir / "model_repo"
    )
    train_task = luigi.TaskParameter()
    platform = luigi.Parameter(
        default="TENSORRT",
        description="Platform to use for exporting model for inference",
    )
    resample_rates = luigi.ListParameter()
    kernel_lengths = luigi.ListParameter()
    high_passes = luigi.ListParameter()
    low_passes = luigi.ListParameter()
    inference_sampling_rates = luigi.ListParameter()
    starting_offsets = luigi.ListParameter()
    classes = luigi.ListParameter()
    layers = luigi.ListParameter()

    model_type = luigi.Parameter(default="export")

@inherits(ExportParams)
class ExportLocal(AframeSingularityTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repository_directory.mkdir(exist_ok=True, parents=True)

    def output(self):
        return ModelRepositoryTarget(self.repository_directory, self.platform)

    def requires(self):
        return self.train_task.req(self)

    @property
    def default_image(self):
        return "export.sif"

    @property
    def num_ifos(self):
        return len(self.ifos)

    def run(self):
        from hermes.quiver import Platform

        # convert string to Platform enum
        platform = Platform[self.platform]

        # Assuming a convention for batch file/model file
        # names and locations
        weights = self.input().path
        weights_dir = os.path.dirname(weights)
        batch_file = weights_dir + "/batch.h5"
        if self.model_type == 'mm_export':
            from export.mm_main import mm_export
            from export.mm_modules import separate_model
            separate_model(weights,
                           batch_file,
                           self.num_ifos,
                           self.kernel_length,
                           self.sample_rate,
                           self.batch_size,
                           self.classes,
                           self.layers,
                           self.inference_sampling_rates,)
            mm_export(
                weights,
                self.repository_directory,
                batch_file,
                self.num_ifos,
                self.kernel_length,
                self.inference_sampling_rate,
                self.sample_rate,
                self.batch_size,
                self.fduration,
                self.psd_length,
                self.resample_rates,
                self.kernel_lengths,
                self.high_passes,
                self.low_passes,
                self.inference_sampling_rates,
                self.starting_offsets,
                self.classes,
                self.fftlength,
                self.q,
                self.highpass,
                self.lowpass,
                self.streams_per_gpu,
                self.aframe_instances,
                self.preproc_instances,
                platform,
                clean=self.clean,)
        else:
            from export.main import export
            export(
                weights,
                self.repository_directory,
                batch_file,
                self.num_ifos,
                self.kernel_length,
                self.inference_sampling_rate,
                self.sample_rate,
                self.batch_size,
                self.fduration,
                self.psd_length,
                self.resample_rates,
                self.kernel_lengths,
                self.high_passes,
                self.low_passes,
                self.inference_sampling_rates,
                self.starting_offsets,
                self.classes,
                self.fftlength,
                self.q,
                self.highpass,
                self.lowpass,
                self.streams_per_gpu,
                self.aframe_instances,
                self.preproc_instances,
                platform,
                clean=self.clean,
            )
