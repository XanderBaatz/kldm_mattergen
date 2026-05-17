from __future__ import annotations

from enum import Enum


class ModelTarget(str, Enum):
    """kldm_plus model targets for training/sampling.

    Includes MatterGen-compatible targets plus KLDM-specific continuous targets.
    """

    score_times_std = "score_times_std"
    logits = "logits"
    x0 = "x0"
    eps = "eps"

    @classmethod
    def from_any(cls, value: ModelTarget | Enum | str) -> ModelTarget:
        """Normalize target values from config/enum inputs."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            return cls(value.value)
        return cls(str(value))


# Backward-compatible alias for existing imports.
KLDMModelTarget = ModelTarget
