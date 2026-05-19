"""Predictors for kinetic Langevin reverse-time sampling.

Two predictors implement the one reverse step of the PC sampler:

``KinLangevinEMPredictor`` (registered for ``"vel"``)
    Exponential integrator (EI) for the OU velocity reverse step::

        v_{t-Δt} = exp(Δτ) v_t + 2(exp(Δτ)-1) score_v + √(exp(2Δτ)-1) noise

    where ``Δτ = γ·tf·|Δt|`` is the *internal* time step.  The score
    ``score_v`` is the reconstructed full velocity score produced by
    ``KineticDiffusionModule.score_fn``::

        score_v = -v_t/σ_v² + model_out["pos"] × prefactor_t × √σ_norm_t

``KinLangevinPosPredictor`` (registered for ``"pos"``)
    Deterministic position update driven by the current velocity::

        pos_{t-Δt} = wrap(pos_t - Δτ·v_t)

    Position in the kinetic Langevin SDE has **no noise term**; the
    predictor is therefore fully deterministic (sample == mean) and ignores
    its ``score`` argument.
"""  # noqa: RUF002

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mattergen.diffusion.sampling.predictors import Predictor
from torch import Tensor

from kldm_plus.diffusion.corruption.kinetic_multi_corruption import KinLangevinPosCoupled
from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE
from kldm_plus.nn.utils import scatter_center

if TYPE_CHECKING:
    from mattergen.diffusion.corruption.corruption import Corruption
    from mattergen.diffusion.data.batched_data import BatchedData
    from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


class KinLangevinEMPredictor(Predictor):
    """Exponential-integrator predictor for the **velocity** field.

    Implements the reverse-time step of the OU velocity process using the
    matrix-exponential (exponential integrator) scheme.  This is numerically
    more accurate than naive Euler-Maruyama for the OU process because it
    exactly integrates the linear drift::

        v_{t-Δt} = exp(γΔτ) v_t + 2(exp(γΔτ)-1) s_v(v_t, x_t, t)
                   + √(exp(2γΔτ)-1) ξ,  ξ ~ N(0, I)  [zero-CoG per graph]

    where ``Δτ = tf·|Δt|`` is the internal time step and ``s_v`` is the
    full velocity score (see ``KineticDiffusionModule.score_fn``).

    The predictor is registered for ``"vel"`` via a matching
    ``is_compatible`` check against ``KineticLangevinSDE``.
    """  # noqa: RUF002

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        """Check if predictor is compatible with corruption process."""
        return isinstance(corruption, KineticLangevinSDE)

    def update_given_score(  # noqa: PLR0913
        self,
        *,
        x: Tensor,
        t: Tensor,  # noqa: ARG002
        dt: Tensor,
        batch_idx: torch.LongTensor,
        score: Tensor,
        batch: BatchedData | None,  # noqa: ARG002
    ) -> SampleAndMean:
        """One EI reverse step for velocity.

        Args:
            x: noisy velocity ``v_t``, shape ``[num_atoms, 3]``.
            t: per-graph diffusion time ``[B]`` (unused here; kept for interface).
            dt: scalar step size (negative, going backwards in time).
            batch_idx: node→graph index ``[num_atoms]``.
            score: reconstructed velocity score ``s_v``, shape ``[num_atoms, 3]``.
            batch: full noisy batch (unused here; kept for interface).

        Returns:
            ``(v_sample, v_mean)``, both shape ``[num_atoms, 3]``.

        """
        sde = self.corruption
        if not isinstance(sde, KineticLangevinSDE):
            msg = f"Expected {KineticLangevinSDE.__name__}, got {type(sde).__name__}"
            raise TypeError(msg)

        # Map external |Δt| → internal Δτ (scalar or near-scalar)
        dt_tau = sde.tau(dt.abs())
        exp_dt = math.exp(dt_tau.item())
        expm1_dt = math.expm1(dt_tau.item())
        std = math.sqrt(max(math.expm1(2.0 * dt_tau.item()), 0.0))

        # Zero-CoG noise (velocity lives in the zero-CoG subspace)
        noise = scatter_center(torch.randn_like(x), index=batch_idx)

        vel_mean = exp_dt * x + 2.0 * expm1_dt * score
        vel_sample = vel_mean + std * noise

        return vel_sample, vel_mean


class KinLangevinPosPredictor(Predictor):
    """Deterministic position predictor for kinetic Langevin sampling.

    Position in the kinetic Langevin SDE is driven **only** by velocity; there
    is no Brownian noise term.  The reverse-time step is therefore::

        pos_{t-Δt} = wrap(pos_t - Δτ·v_t,  period=scale_pos)

    where ``v_t`` is the current noisy velocity (taken from ``batch["vel"]``
    *before* the velocity predictor has updated it — this ordering matches
    TDM's ``reverse_step_em``).

    Because there is no stochasticity, ``sample == mean``.

    The predictor is registered for ``"pos"`` via ``KinLangevinPosCoupled``;
    the ``score`` argument (raw model output for pos) is intentionally ignored
    since the pos update is score-free.
    """

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        """Check if predictor is compatible with corruption process."""
        return isinstance(corruption, KinLangevinPosCoupled)

    def update_given_score(  # noqa: PLR0913
        self,
        *,
        x: Tensor,
        t: Tensor,  # noqa: ARG002
        dt: Tensor,
        batch_idx: torch.LongTensor,  # noqa: ARG002
        score: Tensor,  # noqa: ARG002
        batch: BatchedData | None,
    ) -> SampleAndMean:
        """Deterministic reverse step for position.

        Args:
            x: noisy position ``pos_t``, shape ``[num_atoms, 3]``.
            t: per-graph diffusion time (unused).
            dt: scalar step size (negative).
            batch_idx: node→graph index (unused — no CoG centering needed for pos).
            score: raw model output for pos (intentionally ignored).
            batch: full noisy batch; ``batch["vel"]`` provides the current velocity.

        Returns:
            ``(pos_new, pos_new)`` — deterministic, same tensor for sample and mean.

        """
        sde = self.corruption
        if not isinstance(sde, KinLangevinPosCoupled):
            msg = f"Expected {KinLangevinPosCoupled.__name__}, got {type(sde).__name__}"
            raise TypeError(msg)
        kinlang = sde._kinlang  # noqa: SLF001

        # Internal time step (scalar)
        dt_tau = kinlang.tau(dt.abs()).item()

        # pos reverse: x_{t-dt} = wrap(x_t - Δτ·v_t)
        if batch is None:
            msg = "batch must be provided for KinLangevinPosPredictor"
            raise ValueError(msg)
        vel_t = batch["vel"]  # current velocity (BEFORE vel predictor updates it)
        pos_new = kinlang.wrap_pos(x - dt_tau * vel_t, period=kinlang.scale_pos)

        return pos_new, pos_new
