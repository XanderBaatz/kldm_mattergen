"""Kinetic Langevin Diffusion Model — Lightning wrapper."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any

import torch
from mattergen.diffusion.lightning_module import DiffusionLightningModule, OptimizerPartial, SchedulerPartial

from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule
from kldm_plus.diffusion.sampling import make_sampler

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
        sampling_eval_batches: int = 0,
        sampling_N: int = 1000,  # noqa: N803
    ) -> None:
        super().__init__(
            diffusion_module=diffusion_module,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
        )
        self.val_metrics = val_metrics
        self.ema_start = ema_start
        self._sampling_eval_batches = sampling_eval_batches
        self._sampling_N = sampling_N
        self._val_batches_for_sampling: list[BatchedData] = []

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
        self._val_batches_for_sampling = []

    def validation_step(self, batch: BatchedData, batch_idx: int) -> torch.Tensor | None:
        result = super().validation_step(batch, batch_idx)
        if self.val_metrics is not None and self._sampling_eval_batches > 0 and len(self._val_batches_for_sampling) < self._sampling_eval_batches:
            self._val_batches_for_sampling.append(batch)
        return result

    def on_validation_epoch_end(self) -> None:
        if self.val_metrics is None:
            return
        if self._val_batches_for_sampling and self._sampling_N > 0:
            try:
                sampler = make_sampler(
                    diffusion_module=self.diffusion_module,
                    device=self.device,
                    N=self._sampling_N,
                )
                for batch in self._val_batches_for_sampling:
                    # Ensure vel field exists (simplified parameterisation: v₀ = 0).
                    # Also set vel_batch so pc_sampler.get_batch_idx('vel') resolves.
                    try:
                        if batch["vel"] is None:
                            raise KeyError  # noqa: TRY301
                    except (
                        KeyError,
                        AttributeError,
                    ):
                        batch = batch.replace(
                            vel=torch.zeros_like(batch["pos"]),
                            vel_batch=batch.batch,
                        )
                    pred, _ = sampler.sample(conditioning_data=batch)
                    self.val_metrics.update_from_chemgraphs(pred, batch)
                    # Diagnostic: log a quick summary so the .err file shows what's happening
                    self._log_sample_diagnostics(pred)
            except Exception:
                import logging as _logging

                _logging.getLogger(__name__).exception(
                    "Sampling failed during validation - metrics will be skipped this epoch."
                )
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

    def _log_sample_diagnostics(self, pred: BatchedData) -> None:
        """Log cell volumes and validity counts from one sampled batch to stderr."""
        import logging
        import math

        import numpy as np
        from kldm_plus.metrics.csp import _decode_cell_6d
        from pymatgen.core import Lattice

        log = logging.getLogger(__name__)
        try:
            cell = pred["cell"].detach().cpu()
            batch_idx = pred.get_batch_idx("pos")
            b = cell.shape[0]

            angles_loc = self.val_metrics.angles_loc if self.val_metrics else 0.0
            angles_scale = self.val_metrics.angles_scale if self.val_metrics else 0.35
            lengths_loc_scale = getattr(self.val_metrics, "lengths_loc_scale", None)

            volumes = []
            for i in range(b):
                n_atoms = int((batch_idx == i).sum().item())
                lengths_loc = lengths_scale = None
                if lengths_loc_scale and n_atoms in lengths_loc_scale:
                    loc_t, scale_t = lengths_loc_scale[n_atoms]
                    lengths_loc = np.asarray(loc_t)
                    lengths_scale = np.asarray(scale_t)
                lengths, angles = _decode_cell_6d(
                    cell[i], angles_loc, angles_scale, lengths_loc, lengths_scale
                )
                try:
                    vol = Lattice.from_parameters(
                        float(lengths[0]), float(lengths[1]), float(lengths[2]),
                        float(angles[0]), float(angles[1]), float(angles[2]),
                    ).volume
                except Exception:
                    vol = float("nan")
                volumes.append(vol)
            vols = np.array(volumes)
            log.info(
                "epoch %d | sampled cell volumes (Å³): min=%.3f mean=%.3f max=%.3f | n_valid_vol=%d/%d",
                self.current_epoch,
                np.nanmin(vols),
                np.nanmean(vols),
                np.nanmax(vols),
                int(np.sum(vols > 0.1)),
                b,
            )
        except Exception:
            log.debug("_log_sample_diagnostics failed", exc_info=True)
