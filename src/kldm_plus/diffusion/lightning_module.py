"""Kinetic Langevin Diffusion Model — Lightning wrapper."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import Any

from mattergen.diffusion.lightning_module import DiffusionLightningModule, OptimizerPartial, SchedulerPartial

from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule
from kldm_plus.metrics.csp import CSPMetrics  # noqa: TC001


class KLDMLightningModule(DiffusionLightningModule[KineticDiffusionModule]):
    """LightningModule for the Kinetic Langevin Diffusion Model.

    Wraps a :class:`KineticDiffusionModule`.  When ``val_metrics`` is provided
    (a :class:`~kldm_plus.metrics.csp.CSPMetrics` instance), it is called at the
    end of every validation epoch and the results are logged.
    """

    def __init__(
        self,
        diffusion_module: KineticDiffusionModule,
        optimizer_partial: OptimizerPartial | None = None,
        scheduler_partials: Sequence[dict[str, Any | SchedulerPartial]] | None = None,
        val_metrics: CSPMetrics | None = None,
    ) -> None:
        super().__init__(
            diffusion_module=diffusion_module,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
        )
        self.val_metrics = val_metrics

    def on_validation_epoch_start(self) -> None:
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def validation_step(self, val_batch: Any, batch_idx: int) -> Any:
        loss = super().validation_step(val_batch, batch_idx)
        if self.val_metrics is not None:
            # Accumulate ground-truth structures from the clean batch so that
            # structure validity is tracked every epoch without requiring
            # expensive full denoising runs.
            from kldm_plus.metrics.csp import chemgraph_to_structures

            structs = chemgraph_to_structures(
                val_batch,
                angles_loc=self.val_metrics.angles_loc,
                angles_scale=self.val_metrics.angles_scale,
            )
            self.val_metrics.update(structs, structs)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is None:
            return
        summary = self.val_metrics.summarize()
        for k, v in summary.items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=True, sync_dist=True)
