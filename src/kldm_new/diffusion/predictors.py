"""Custom predictors for KLDM reverse-time sampling.

Provides:

* :class:`TDMPredictor` — exponential-integrator / DDIM-like predictor for
  the coupled (velocity, position) kinetic-Langevin process.
* Re-exports MatterGen's :class:`AncestralSamplingPredictor` for the cell.
"""

from __future__ import annotations

import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData
from mattergen.diffusion.sampling.predictors import Predictor
from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean
from torch import Tensor

from kldm_new.diffusion.tdm import KineticLangevinSDE


class TDMPredictor(Predictor):
    """Predictor for the TDM kinetic-Langevin process.

    Implements a DDIM-like deterministic predictor that updates both
    velocity and position simultaneously using the exponential integrator.

    Parameters
    ----------
    corruption : KineticLangevinSDE
        The kinetic Langevin SDE instance.
    score_fn : callable or None
        Score function ``(x, t, batch_idx) -> score``.
    use_ddim : bool
        If True, use the DDIM-like predictor. Otherwise use the EM
        exponential integrator.

    """

    def __init__(
        self,
        corruption: KineticLangevinSDE,
        score_fn=None,
        use_ddim: bool = True,
    ):
        super().__init__(corruption=corruption, score_fn=score_fn)
        assert isinstance(corruption, KineticLangevinSDE)
        self._tdm = corruption
        self.use_ddim = use_ddim

    def update_given_score(
        self,
        *,
        x: Tensor,
        t: Tensor,
        dt: float,
        batch_idx: B = None,
        score: Tensor,
        batch: BatchedData | None = None,
    ) -> SampleAndMean:
        """Update velocity given the predicted score.

        Note: this only updates **velocity**.  Position must be updated
        by the caller (the KLDM sampler) using the returned velocity.
        The ``SampleAndMean`` holds the new velocity in both fields
        (deterministic predictor → sample == mean for DDIM).
        """
        if self.use_ddim:
            v_new, _ = self._tdm.reverse_step_pc_predictor(
                v=x,
                pos=torch.zeros_like(x),  # pos update done externally
                score=score,
                t=t,
                dt=dt,
                batch_idx=batch_idx,
            )
        else:
            v_new, _ = self._tdm.reverse_step_em(
                v=x,
                pos=torch.zeros_like(x),
                score=score,
                dt=dt,
                batch_idx=batch_idx,
            )
        return (v_new, v_new)

    @classmethod
    def is_compatible(cls, corruption) -> bool:
        return isinstance(corruption, KineticLangevinSDE)
