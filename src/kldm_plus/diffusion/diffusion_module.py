from __future__ import annotations

import torch
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.diffusion_module import BatchTransform, DiffusionModule, T
from mattergen.diffusion.losses import Loss  # noqa: TC002
from mattergen.diffusion.model_utils import convert_model_out_to_score
from mattergen.diffusion.score_models.base import ScoreModel  # noqa: TC002
from mattergen.diffusion.timestep_samplers import TimestepSampler  # noqa: TC002

from kldm_plus.diffusion.corruption.kinetic_multi_corruption import KineticMultiCorruption


class KineticDiffusionModule(DiffusionModule):
    """Diffusion module for the kinetic Langevin KLDM.

    All training logic is inherited from ``DiffusionModule``; the only
    customization needed is fixing ``_get_device`` so it uses a field that is
    always present in the clean batch (``pos``), rather than the first entry
    of ``sdes`` which is ``"vel"`` — a field that only exists after
    ``KineticMultiCorruption.sample_marginal`` has been called.

    ``score_fn`` (used during sampling) currently converts only the ``sdes``
    fields (``vel``, ``cell``) back to scores.  The ``pos`` field is not
    registered as an SDE, so a dedicated predictor must handle its score
    conversion directly (see ``kinetic_sdeevinExponentialPredictor``, to be
    implemented).
    """

    def __init__(
        self,
        model: ScoreModel,
        corruption: KineticMultiCorruption,
        loss_fn: Loss,
        pre_corruption_fn: BatchTransform | None = None,
        timestep_sampler: TimestepSampler | None = None,
    ) -> None:
        """Initialize the diffusion module."""
        super().__init__(
            model=model,
            corruption=corruption,
            loss_fn=loss_fn,
            pre_corruption_fn=pre_corruption_fn,
            timestep_sampler=timestep_sampler,
        )

    def _get_device(self, batch: T) -> torch.device:
        # ``pos`` is always present in both clean and noisy batches; the first
        # SDE key (``"vel"``) is absent from the clean batch and would raise
        # KeyError if used here.
        return batch["pos"].device

    def score_fn(self, x: T, t: torch.Tensor) -> T:
        """Compute scores for all fields.

        The base ``DiffusionModule.score_fn`` only iterates over ``corruption.sdes``
        (``"vel"``, ``"cell"``), but the KLDM model outputs the kinetic_sde score target
        in the ``"pos"`` slot rather than in ``"vel"``.  This override:

        1. Converts ``"cell"`` via the standard ``convert_model_out_to_score``.
        2. Reconstructs the velocity score from kinetic Langevin physics::

               score_v = -v_t / σ_v² + model_out["pos"] × prefactor_t × √σ_norm_t

           where ``prefactor_t = tanh(γτ/2)`` and ``σ_norm_t = E[‖score_WN‖²]``.

        3. Passes ``model_out["pos"]`` through unchanged so ``kinetic_sdeevinPosPredictor``
           can access the raw kinetic_sde target during sampling.
        """
        if not isinstance(self.corruption, KineticMultiCorruption):
            msg = f"Expected {KineticMultiCorruption.__name__}"
            raise TypeError(msg)
        kinetic_sde = self.corruption.kinetic_sde

        model_out: T = self.model(x, t)

        # ---- cell ----------------------------------------------------------------
        cell_batch_idx = self.corruption._get_batch_indices(x).get("cell")  # noqa: SLF001
        cell_score = convert_model_out_to_score(
            model_out=model_out["cell"],
            sde=self.corruption.sdes["cell"],
            model_target=self.model_targets["cell"],
            t=t,
            batch_idx=cell_batch_idx,
            batch=x,
        )

        # ---- velocity: reconstruct full score from kinetic_sde physics ---------------
        vel_batch_idx = self.corruption._get_batch_indices(x)["vel"]  # noqa: SLF001
        vel_t = x["vel"]
        tau = kinetic_sde._t_internal(t)  # [B] internal time  # noqa: SLF001
        _, sigma_v_t = kinetic_sde.marginal_prob(x=vel_t, t=t, batch_idx=vel_batch_idx)
        prefactor_t = maybe_expand(torch.tanh(kinetic_sde.gamma * tau / 2.0), batch=vel_batch_idx, like=vel_t)
        sigma_norm_t = maybe_expand(torch.sqrt(kinetic_sde._sigma_norm_t(t)), batch=vel_batch_idx, like=vel_t)  # noqa: SLF001
        score_vel = -(vel_t / sigma_v_t**2) + model_out["pos"] * prefactor_t * sigma_norm_t

        return model_out.replace(vel=score_vel, cell=cell_score)
