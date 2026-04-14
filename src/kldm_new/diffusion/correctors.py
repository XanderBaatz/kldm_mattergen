"""Custom correctors for KLDM reverse-time sampling.

Provides:

* :class:`TDMLangevinCorrector` — adaptive Langevin corrector on *velocity*
  for the kinetic-Langevin process.
"""

from __future__ import annotations

import torch
from mattergen.diffusion.corruption.corruption import B  # noqa: TC002
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean  # noqa: TC002
from torch import Tensor

from kldm_new.diffusion.tdm import KineticLangevinSDE


class TDMLangevinCorrector:
    """Adaptive Langevin corrector on velocity for TDM.

    At each corrector step, the velocity is updated via:

    .. math::
        v \\leftarrow v + \\delta \\cdot \\text{score} + \\sqrt{2\\delta} \\cdot z

    where ``delta = tau / mean(||score||^2)`` is an adaptive step size.

    Parameters
    ----------
    corruption : KineticLangevinSDE
        The kinetic Langevin SDE.
    score_fn : callable or None
        Score function (unused — scores passed directly via ``step_given_score``).
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
    ):
        self.corruption = corruption
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.tau = tau

    def step_given_score(
        self,
        *,
        x: Tensor,
        batch_idx: B = None,
        score: Tensor,
        t: Tensor,
        dt: float,
    ) -> SampleAndMean:
        """Single Langevin corrector step on velocity.

        Args:
            x: Current velocity ``(N, 3)``.
            batch_idx: Atom → graph mapping.
            score: Predicted score ``(N, 3)``.
            t: Current diffusion time (unused, kept for interface compat).
            dt: Time step (unused, kept for interface compat).

        Returns:
            Updated velocity.

        """
        v_new, _ = self.corruption.reverse_step_pc_corrector(
            v=x,
            pos=torch.zeros_like(x),  # position update done externally
            score=score,
            tau=self.tau,
            batch_idx=batch_idx,
        )
        return (v_new, v_new)

    @classmethod
    def is_compatible(cls, corruption) -> bool:
        return isinstance(corruption, KineticLangevinSDE)
