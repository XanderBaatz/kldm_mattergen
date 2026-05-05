"""Custom correctors for KLDM reverse-time sampling.

Provides:

* :class:`TDMLangevinCorrector` — adaptive Langevin corrector on *velocity*
  for the kinetic-Langevin process.

The corrector mathematics is self-contained; no delegation to
:class:`~kldm_new.diffusion.tdm.KineticLangevinSDE` is required.
"""

from __future__ import annotations

import torch
from mattergen.diffusion.corruption.corruption import B  # noqa: TC002
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean  # noqa: TC002
from torch import Tensor

from kldm_new.diffusion.tdm import KineticLangevinSDE


class TDMLangevinCorrector:
    r"""Adaptive Langevin corrector on **velocity** for TDM.

    At each corrector step the velocity is updated via:

    .. math::
        v \\leftarrow v + \\delta\\,s_\\theta + \\sqrt{2\\delta}\\,z,
        \\qquad z \\sim \\mathcal{N}(0,I)

    where the adaptive step size is:

    .. math::
        \\delta = \\frac{\\tau}{\\mathbb{E}[\\|s_\\theta\\|^2]}

    Position is **not** updated during correction (only velocity is corrected).

    Parameters
    ----------
    corruption : KineticLangevinSDE
        The kinetic Langevin SDE (used only for ``isinstance`` checks).
    score_fn : callable or None
        Score function — unused here, kept for interface symmetry.
    n_steps : int
        Number of Langevin corrector steps per predictor step.
    tau : float
        SNR parameter for adaptive step size.

    """

    def __init__(
        self,
        corruption: KineticLangevinSDE,
        score_fn=None,
        n_steps: int = 1,
        tau: float = 0.5,
    ) -> None:
        """Initialise the Langevin corrector."""
        self.corruption = corruption
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.tau = tau

    def step_given_score(
        self,
        *,
        x: Tensor,
        batch_idx: B = None,  # noqa: ARG002
        score: Tensor,
        t: Tensor,  # noqa: ARG002
        dt: float,  # noqa: ARG002
    ) -> SampleAndMean:
        """Single adaptive Langevin step on velocity.

        Args:
            x: Current velocity ``(N, 3)``.
            score: Predicted score ``(N, 3)``.
            t: Current diffusion time (unused — step is score-only).
            dt: Time step size (unused — step is score-only).

        Returns:
            ``(v_new, v_mean)`` where ``v_mean`` is the deterministic part.

        """
        score_norm_sq = (score**2).mean().clamp(min=1e-8)
        delta = self.tau / score_norm_sq

        v_mean = x + delta * score
        noise = torch.randn_like(x)
        v_new = v_mean + (2.0 * delta).sqrt() * noise
        return v_new, v_mean

    @classmethod
    def is_compatible(cls, corruption: object) -> bool:  # noqa: D102
        return isinstance(corruption, KineticLangevinSDE)
