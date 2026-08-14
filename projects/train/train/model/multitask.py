from typing import List

import torch
import torch.nn.functional as F
from architectures import Architecture

from train.metrics import TimeSlideAUROC
from train.model.supervised import SupervisedAframe

Tensor = torch.Tensor


class SupervisedMultiTaskAframe(SupervisedAframe):
    """
    Multi-task model that jointly optimizes binary classification
    and injection parameter regression.

    The architecture's forward pass must return a tuple
    ``(logits, param_estimates)`` where ``logits`` has shape ``(N, 1)``
    and ``param_estimates`` has shape ``(N, len(param_names))``.

    Validation uses the standard AUROC metric on the classification head,
    so the existing validation infrastructure is fully reused.

    Args:
        arch:
            Architecture whose forward pass returns (logits, param_estimates).
        param_names:
            Ordered list of parameter names to regress on. Must match the
            output ordering of the architecture's regression head.
        regression_weight:
            Scalar multiplier applied to the regression loss before summing
            with the classification loss.
    """

    def __init__(
        self,
        arch: Architecture,
        metric: TimeSlideAUROC,
        param_names: List[str],
        *args,
        regression_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(arch, metric, *args, **kwargs)
        self.param_names = param_names
        self.regression_weight = regression_weight

    def score(self, X: Tensor) -> Tensor:
        logits, params = self(X)
        return logits, params

    def train_step(self, batch):
        X, y, params = batch
        logits, param_estimates = self(X)

        clf_loss = F.binary_cross_entropy_with_logits(logits, y)

        targets = torch.stack([params[k] for k in self.param_names], dim=1)
        mask = ~torch.isnan(targets).any(dim=1)
        if mask.any():
            reg_loss = F.mse_loss(param_estimates[mask], targets[mask])
        else:
            reg_loss = param_estimates.sum() * 0.0

        return {"classification_loss": clf_loss, "regression_loss": reg_loss}

    def compute_loss_fn(self, classification_loss, regression_loss):
        return classification_loss + self.regression_weight * regression_loss

    def validation_step(self, batch, _) -> None:
        shift, X_bg, X_inj, params = batch

        y_bg, _ = self.score(X_bg)

        # compute predictions over multiple views of
        # each injection and use their average as our
        # prediction
        num_views, batch, *shape = X_inj.shape
        X_inj = X_inj.view(num_views * batch, *shape)
        y_fg, reg_preds = self.score(X_inj)
        y_fg = y_fg.view(num_views, batch)
        y_fg = y_fg.mean(0)

        # include the shift associated with this data
        # in our outputs to reconstruct background
        # timeseries at aggregation time
        self.metric.update(shift, y_bg, y_fg)
        self.metric.update(y_fg)
        # lightning will take care of updating then
        # computing the metric at the end of the
        # validation epoch
        self.log(
            "valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )