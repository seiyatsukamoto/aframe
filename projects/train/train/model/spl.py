import os
import torch

from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase
from train.model.supervised import SupervisedAframe
from train.metrics import TimeSlideAUROC
from train.prior import KDEDistribution
from train.data.waveforms.generator.cbc import CBCGenerator


Tensor = torch.Tensor

class SPLAframeModel(SupervisedAframe):
    def __init__(self, dist_file, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)
        self.dist_file = dist_file
    def on_train_epoch_start(self, *args, **kwargs):
        super().on_train_epoch_start(*args, **kwargs)
        if os.path.exists(self.dist_file):
            self.trainer.dataset.waveform_sampler.training_prior.priors['cm_mr']=KDEDistribution(file = self.dist_file, validate_args = False)