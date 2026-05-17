from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.diffusion_module import BatchTransform, DiffusionModule, T
from mattergen.diffusion.timestep_samplers import UniformTimestepSampler

from kldm_plus.diffusion.corruption.kinetic_multi_corruption import KineticMultiCorruption
from kldm_plus.diffusion.model_utils import convert_model_out_to_score
from kldm_plus.diffusion.training.model_target import ModelTarget

if TYPE_CHECKING:
    from mattergen.diffusion.losses import Loss
    from mattergen.diffusion.score_models.base import ScoreModel
    from mattergen.diffusion.timestep_samplers import TimestepSampler


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
        """Initialize without MatterGen enum coercion for model targets."""
        torch.nn.Module.__init__(self)
        self.model = model
        self.corruption = corruption
        self.loss_fn = loss_fn
        self.pre_corruption_fn = pre_corruption_fn or (lambda x: x)
        self.model_targets = {k: ModelTarget.from_any(v) for k, v in loss_fn.model_targets.items()}
        self.timestep_sampler = timestep_sampler or UniformTimestepSampler(
            min_t=1e-5,
            max_t=corruption.T,
        )
        self._register_corruption_modules()

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

          1. Converts ``"cell"`` using kldm's model-utils conversion helper.
        2. Reconstructs the velocity score from kinetic Langevin physics::

               score_v = -v_t / σ_v² + model_out["pos"] × prefactor_t × √σ_norm_t

           where ``prefactor_t = tanh(γτ/2)`` and ``σ_norm_t = E[‖score_WN‖²]``.

        3. Passes ``model_out["pos"]`` through unchanged so ``kinetic_sdeevinPosPredictor``
           can access the raw kinetic_sde target during sampling.
        """  # noqa: RUF002
        if not isinstance(self.corruption, KineticMultiCorruption):
            msg = f"Expected {KineticMultiCorruption.__name__}"
            raise TypeError(msg)
        kinetic_sde = self.corruption.kinetic_sde

        model_out: T = self.model(x, t)

        # ---- cell ----------------------------------------------------------------
        cell_batch_idx = cast("torch.LongTensor", self.corruption._get_batch_indices(x).get("cell"))  # noqa: SLF001
        cell_target = ModelTarget.from_any(
            getattr(self.loss_fn, "cell_model_target", self.model_targets["cell"])
        )
        cell_score = convert_model_out_to_score(
            model_target=cell_target,
            sde=self.corruption.sdes["cell"],
            model_out=model_out["cell"],
            noisy_x=x["cell"],
            batch_idx=cell_batch_idx,
            t=t,
            batch=x,
        )

        # ---- velocity: reconstruct full score from kinetic_sde physics ---------------
        vel_batch_idx = cast("torch.LongTensor", self.corruption._get_batch_indices(x)["vel"])  # noqa: SLF001
        vel_t = x["vel"]
        tau = kinetic_sde.tau(t)  # [B] internal time
        _, sigma_v_t = kinetic_sde.marginal_prob(x=vel_t, t=t, batch_idx=vel_batch_idx)
        prefactor_t = maybe_expand(torch.tanh(kinetic_sde.gamma * tau / 2.0), batch=vel_batch_idx, like=vel_t)
        sigma_norm_t = maybe_expand(torch.sqrt(kinetic_sde._sigma_norm_t(t)), batch=vel_batch_idx, like=vel_t)  # noqa: SLF001
        score_vel = -(vel_t / sigma_v_t**2) + model_out["pos"] * prefactor_t * sigma_norm_t

        return model_out.replace(vel=score_vel, cell=cell_score)
