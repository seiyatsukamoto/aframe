import torch
from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase
from torchmetrics import MeanAbsoluteError
from train.model.supervised import SupervisedAframe
Tensor = torch.Tensor

class SupervisedAutoencoderAframe(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)

    def forward(self, X):
        return self.model(X)

    def score(self, X):
        return self(X)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, X_target = batch
        y = self(X)
        return torch.nn.functional.mse_loss(y, X_target)

    def validation_step(self, batch, _) -> None:
        shift, X_fg, X_fg_target = batch
        num_views, batch, *shape = X_fg.shape
        X_fg = X_fg.view(num_views * batch, *shape)
        X_fg_target = X_fg_target.view(num_views * batch, *shape)
        y_fg = self.score(X_fg)
        self.metric.update(X_fg_target.contiguous(), y_fg.contiguous())
        self.log(
            "fg_loss",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )