"""Kinetic Langevin Diffusion Model — Lightning wrapper."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any

import torch
from mattergen.diffusion.lightning_module import DiffusionLightningModule, OptimizerPartial, SchedulerPartial

from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule

if TYPE_CHECKING:
    from mattergen.diffusion.data.batched_data import BatchedData

    from kldm_plus.metrics.csp import CSPMetrics

# Map KineticLoss field keys → WandB-friendly metric names.
_FIELD_RENAME: dict[str, str] = {
    "pos": "loss_pos",
    "cell": "loss_cell",
    "atomic_numbers": "loss_atomic_numbers",
}


class KLDMLightningModule(DiffusionLightningModule[KineticDiffusionModule]):
    """LightningModule for the Kinetic Langevin Diffusion Model.

    Overrides metric logging to use ``{split}/{name}`` format matching kldm_frnct:
    - ``train/loss_weighted``, ``val/loss_weighted``   — total weighted loss
    - ``train/loss_pos``,      ``val/loss_pos``        — kinetic-Langevin position component
    - ``train/loss_cell``,     ``val/loss_cell``       — lattice VP-SDE component
    - ``train/loss_atomic_numbers``, ``val/loss_atomic_numbers`` — atom-type masking (de-novo only)
    - ``val/valid``, ``val/match_rate``, ``val/rmse``  — structure quality (requires ``val_metrics``)

    EMA is applied to ``diffusion_module.model`` weights starting at ``ema_start``
    epoch, matching kldm_frnct behaviour (decay=0.999, start=500 by default).
    """

    def __init__(
        self,
        diffusion_module: KineticDiffusionModule,
        optimizer_partial: OptimizerPartial | None = None,
        scheduler_partials: Sequence[dict[str, Any | SchedulerPartial]] | None = None,
        val_metrics: CSPMetrics | None = None,
        ema_decay: float = 0.999,
        ema_start: int = 500,
    ) -> None:
        super().__init__(
            diffusion_module=diffusion_module,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
        )
        self.val_metrics = val_metrics
        self.ema_start = ema_start

        self.ema_model: torch.optim.swa_utils.AveragedModel = torch.optim.swa_utils.AveragedModel(
            diffusion_module.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
        )

    # ------------------------------------------------------------------
    # Override mattergen's _calc_loss to use {split}/{name} logging format
    # ------------------------------------------------------------------

    def _calc_loss(
        self,
        batch: BatchedData,
        train: bool,  # noqa: FBT001
    ) -> torch.Tensor | None:
        loss, metrics = self.diffusion_module.calc_loss(batch)
        split = "train" if train else "val"
        batch_size = batch.get_batch_size()

        self.log(
            f"{split}/loss_weighted",
            loss,
            on_step=train,
            on_epoch=True,
            prog_bar=train,
            batch_size=batch_size,
            sync_dist=True,
        )
        for k, v in metrics.items():
            name = _FIELD_RENAME.get(k, k)
            self.log(
                f"{split}/{name}",
                v,
                on_step=train,
                on_epoch=True,
                prog_bar=train,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss

    # ------------------------------------------------------------------
    # EMA update — mirrors kldm_frnct LitKLDM.on_train_batch_end
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.trainer.current_epoch >= self.ema_start:
            self.ema_model.update_parameters(self.diffusion_module.model)

    # ------------------------------------------------------------------
    # Validation-epoch structure-quality metrics (requires sampling)
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is None:
            return
        summary = self.val_metrics.summarize()
        for k, v in summary.items():
            self.log(
                f"val/{k}",
                v,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
