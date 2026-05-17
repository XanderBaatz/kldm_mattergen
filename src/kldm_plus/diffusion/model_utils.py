from __future__ import annotations

from typing import Any

import torch
from mattergen.diffusion.corruption.sde_lib import SDE

from kldm_plus.diffusion.training.model_target import ModelTarget


def convert_model_out_to_score(
    *,
    model_target: ModelTarget,
    sde: SDE,
    model_out: torch.Tensor,
    noisy_x: torch.Tensor,
    batch_idx: torch.LongTensor,
    t: torch.Tensor,
    batch: Any,
) -> torch.Tensor:
    """Convert model outputs to score with KLDM targets.

    Supported model targets:
    - ``score_times_std``: model predicts score * std
    - ``eps``: model predicts raw noise
    - ``x0``: model predicts clean sample
    - ``logits``: passthrough for categorical fields
    """
    resolved_target = ModelTarget.from_any(model_target)

    if resolved_target == ModelTarget.logits:
        return model_out

    mean_coeff, std = sde.mean_coeff_and_std(
        x=model_out,
        t=t,
        batch_idx=batch_idx,
        batch=batch,
    )

    if resolved_target == ModelTarget.score_times_std:
        return model_out / std
    if resolved_target == ModelTarget.eps:
        return -model_out / std
    if resolved_target == ModelTarget.x0:
        return -(noisy_x - mean_coeff * model_out) / (std**2)

    msg = f"Unsupported model_target: {resolved_target}"
    raise ValueError(msg)
