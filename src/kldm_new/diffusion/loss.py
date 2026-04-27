"""Loss functions for KLDM training.

Provides :class:`KLDMLoss` which computes:

* **TDM loss** on velocity — MSE between the model output and the
  wrapped-normal score target (simplified parameterization).
* **Cell loss** — standard denoising-score-matching (score x std) loss
  on the 3x3 lattice matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch  # noqa: TC002
from torch import Tensor  # noqa: TC002
from torch_scatter import scatter_mean

from kldm_new.diffusion.corruption import KLDMMultiCorruption  # noqa: TC001

if TYPE_CHECKING:
    from mattergen.diffusion.data.batched_data import BatchedData


class KLDMLoss:
    """Combined loss for KLDM: TDM (pos/vel) + DSM (cell).

    Parameters
    ----------
    weight_vel : float
        Weight for the velocity (TDM) loss term.
    weight_cell : float
        Weight for the cell (DSM) loss term.

    """

    def __init__(
        self,
        weight_vel: float = 1.0,
        weight_cell: float = 1.0,
    ) -> None:
        """Initialize the KLDM loss with specified weights for velocity and cell losses."""
        self.weight_vel = weight_vel
        self.weight_cell = weight_cell

    def __call__(  # noqa: PLR0913
        self,
        *,
        multi_corruption: KLDMMultiCorruption,
        batch: BatchedData,
        noisy_batch: BatchedData,
        score_model_output: BatchedData,
        t: Tensor,
        node_is_unmasked: torch.LongTensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the total loss.

        Args:
            multi_corruption: The KLDM multi-corruption instance.
            batch: **Clean** batch (before corruption).
            noisy_batch: **Noisy** batch (after corruption at time *t*).
            score_model_output: Model output batch with ``vel`` and ``cell`` fields.
            t: Diffusion time ``(B, 1)``.
            node_is_unmasked: Optional per-atom mask.

        Returns:
            ``(total_loss, loss_dict)``

        """
        pos_batch_idx = noisy_batch.get_batch_idx("pos")

        if pos_batch_idx is None:
            msg = "pos_batch_idx cannot be None"
            raise ValueError(msg)

        batch_size = noisy_batch.get_batch_size()

        # ----- Velocity (TDM) loss -----
        vel_target = multi_corruption.pos_sde.training_target(
            pos_0=batch["pos"],
            pos_t=noisy_batch["pos"],
            v_0=batch["vel"],
            v_t=noisy_batch["vel"],
            t=t,
            batch_idx=pos_batch_idx,
        )
        vel_pred = score_model_output["vel"]
        vel_loss_per_atom = ((vel_pred - vel_target) ** 2).sum(dim=-1)  # (N,)

        if node_is_unmasked is not None:
            vel_loss_per_atom = vel_loss_per_atom * node_is_unmasked

        # Mean per crystal, then mean over batch
        vel_loss = scatter_mean(vel_loss_per_atom, pos_batch_idx, dim=0, dim_size=batch_size).mean()

        # ----- Cell (DSM) loss: score_times_std target -----
        cell_0 = batch["cell"]  # (B, 3, 3)
        cell_t = noisy_batch["cell"]  # (B, 3, 3)
        cell_pred = score_model_output["cell"]  # (B, 3, 3): model predicts -noise

        # Target = -(noise) = -(cell_t - mean) / std
        mean, std = multi_corruption.cell_sde.marginal_prob(cell_0, t, batch_idx=None, batch=noisy_batch)
        # noise = (cell_t - mean) / std → target = -noise  (score_times_std convention)
        noise = (cell_t - mean) / std.clamp(min=1e-8)
        cell_target = -noise

        cell_loss = ((cell_pred - cell_target) ** 2).mean()

        # ----- Combine -----
        total_loss = self.weight_vel * vel_loss + self.weight_cell * cell_loss

        loss_dict = {
            "vel": vel_loss.item(),
            "cell": cell_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, loss_dict
