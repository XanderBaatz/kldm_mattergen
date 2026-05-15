"""Langevin corrector for kinetic Langevin reverse-time sampling.

``KinLangevinLangevinCorrector`` (registered for ``"vel"``)
    Applies one Langevin MCMC correction step to the velocity field::

        step_size = (snr × ‖noise‖ / ‖score_v‖)² × 2
        v_new = v_t + step_size × score_v + √(2 × step_size) × ξ

    This is identical to the mattergen ``LangevinCorrector`` but
    explicitly adapted for ``KineticLangevinSDE`` (which is not a
    ``BaseVPSDE``) and enforces zero-CoG noise so the corrector
    preserves the zero-centre-of-velocity constraint.

Reference
---------
TDM.reverse_step_corrector in ``src/kldm_frnct/model/tdm.py``
"""

from __future__ import annotations

import torch
from mattergen.diffusion.corruption.corruption import Corruption  # noqa: TC002
from mattergen.diffusion.corruption.sde_lib import ScoreFunction
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean, Sampler
from torch import Tensor
from torch_scatter import scatter_add

from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE
from kldm_plus.diffusion.corruption.utils import _scatter_center


class KinLangevinLangevinCorrector(Sampler):
    """Langevin MCMC corrector for the **velocity** field.

    Applies a single annealed Langevin step using the reconstructed velocity
    score produced by ``KineticDiffusionModule.score_fn``::

        step_size = min((snr × ‖ξ‖_G / ‖s_v‖_G)² × 2,  max_step_size)
        v_{corrected} = v + step_size × s_v + √(2 × step_size) × ξ

    where ``‖·‖_G`` denotes the graph-averaged norm and ``ξ`` is a
    zero-CoG Gaussian noise vector.

    The step size is determined automatically by the ``snr`` (signal-to-noise
    ratio) parameter analogously to the standard mattergen
    ``LangevinCorrector``.

    The corrector is registered for ``"vel"`` via a matching
    ``is_compatible`` check against ``KineticLangevinSDE``.

    Args:
        corruption: the ``KineticLangevinSDE`` associated with ``"vel"``.
        score_fn: score function (unused when called via ``step_given_score``
            from ``PredictorCorrector._denoise``; kept for interface).
        n_steps: number of Langevin steps per noise level (stored for
            ``update_fn``; ``_denoise`` manages its own outer loop).
        snr: signal-to-noise ratio controlling the Langevin step size.
        max_step_size: upper bound on the step size coefficient.

    """

    def __init__(
        self,
        corruption: Corruption,
        score_fn: ScoreFunction | None = None,
        n_steps: int = 1,
        snr: float = 0.2,
        max_step_size: float = 1.0,
    ) -> None:
        super().__init__(corruption=corruption, score_fn=score_fn)
        self.n_steps = n_steps
        self.snr = snr
        self.max_step_size = torch.tensor(max_step_size)

    @classmethod
    def is_compatible(cls, corruption: Corruption) -> bool:
        return isinstance(corruption, KineticLangevinSDE)

    def step_given_score(
        self,
        *,
        x: Tensor,
        score: Tensor,
        batch_idx: Tensor,
        t: Tensor,  # noqa: ARG002
        dt: Tensor,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> SampleAndMean:
        """One annealed Langevin correction step for velocity.

        Args:
            x: current velocity ``v_t``, shape ``[num_atoms, 3]``.
            score: reconstructed velocity score ``s_v``, shape ``[num_atoms, 3]``.
            batch_idx: node→graph index ``[num_atoms]``.
            t: per-graph diffusion time (unused).
            dt: step size (unused; step size is derived from snr/norms).

        Returns:
            ``(v_corrected, v_mean)``, both shape ``[num_atoms, 3]``.

        """
        # Zero-CoG noise (preserves zero-centre-of-velocity)
        noise = _scatter_center(torch.randn_like(x), batch_idx)

        # Per-atom squared norms → sum per graph
        B = int(batch_idx.max().item()) + 1
        grad_norm_sq = score.square().sum(dim=-1)  # [num_atoms]
        noise_norm_sq = noise.square().sum(dim=-1)  # [num_atoms]

        grad_norm = torch.sqrt(scatter_add(grad_norm_sq, index=batch_idx, dim_size=B, dim=0)).mean()  # scalar

        noise_norm = torch.sqrt(scatter_add(noise_norm_sq, index=batch_idx, dim_size=B, dim=0)).mean()  # scalar

        # Adaptive step size (same formula as mattergen LangevinCorrector with alpha=1)
        step_size = (self.snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2
        step_size = torch.minimum(step_size, self.max_step_size.to(step_size.device))

        mean = x + step_size * score
        v_corrected = mean + torch.sqrt(2.0 * step_size) * noise

        return v_corrected, mean

    def update_fn(
        self,
        *,
        x: Tensor,
        t: Tensor,
        dt: Tensor,
        batch_idx: Tensor,
        batch: BatchedData | None = None,
    ) -> SampleAndMean:
        """Run ``n_steps`` Langevin corrections using the internal score function."""
        assert self.score_fn is not None, "score_fn must be set to use update_fn"
        for _ in range(self.n_steps):
            score = self.score_fn(x=x, t=t, batch_idx=batch_idx)
            x, mean = self.step_given_score(x=x, score=score, batch_idx=batch_idx, t=t, dt=dt, batch=batch)
        return x, mean
