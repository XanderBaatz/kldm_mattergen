"""Timestep samplers for KLDM training.

Re-exports MatterGen's :class:`TimestepSampler` protocol and
:class:`UniformTimestepSampler`, and adds :class:`LogUniformTimestepSampler`
which is not in the upstream library.

All samplers return a ``(batch_size,)`` float tensor matching MatterGen's
convention; :meth:`~kldm_new.model.lit_module.LitKLDM._sample_t` reshapes
to ``(batch_size, 1)`` for broadcasting against per-atom SDE fields.
"""

from __future__ import annotations

import math

import torch
from mattergen.diffusion.timestep_samplers import TimestepSampler, UniformTimestepSampler

__all__ = ["LogUniformTimestepSampler", "TimestepSampler", "UniformTimestepSampler"]


class LogUniformTimestepSampler:
    """Log-uniform sampler over ``[min_t, max_t]``.

    Equivalent to sampling uniformly in log-space, which over-samples small
    ``t`` relative to uniform sampling.  This can improve training at early
    (low-noise) diffusion times where the score has the steepest gradients.

    Parameters
    ----------
    min_t : float
        Minimum diffusion time.
    max_t : float
        Maximum diffusion time.

    """

    def __init__(self, *, min_t: float = 1e-3, max_t: float = 1.0) -> None:
        """Initialize the LogUniformTimestepSampler."""
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample batch_size times log-uniformly from [min_t, max_t]."""
        log_min = math.log(self.min_t)
        log_max = math.log(self.max_t)
        log_t = torch.rand(batch_size, device=device) * (log_max - log_min) + log_min
        return log_t.exp()
