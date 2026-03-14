import torch
from architectures.supervised import SupervisedArchitecture

from train.model.base import AframeBase
from torchmetrics import MeanAbsoluteError
from train.model.supervised import SupervisedAframe
Tensor = torch.Tensor

class SupervisedSpectrogramAutoencoder(SupervisedAframe):
    def __init__(self, arch: SupervisedArchitecture, fg_metric: MeanAbsoluteError, *args, **kwargs) -> None:
        super().__init__(arch, *args, **kwargs)
        self.fg_metric = fg_metric

    def forward(self, X):
        return self.model(X)

    def score(self, X):
        return self(X)

    def train_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        X, X_target = batch
        y = self(X)
        return torch.nn.functional.l1_loss(y, X_target)

    def validation_step(self, batch, _) -> None:
        X_bg, X_fg, X_fg_target = batch
        y_bg = self.score(X_bg)
        
        num_views, batch, *shape = X_fg.shape
        X_fg = X_fg.view(num_views * batch, *shape)
        X_fg_target = X_fg_target.view(num_views * batch, *shape)
        y_fg = self.score(X_fg)
        
        self.metric.update(X_bg.contiguous(), y_bg.contiguous())
        self.fg_metric.update(X_fg_target.contiguous(), y_fg.contiguous())

        self.log(
            "MAE_bg",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "MAE_fg",
            self.fg_metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )