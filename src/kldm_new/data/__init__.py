"""Data transforms for KLDM — add velocity and build edges on ChemGraph batches."""

from __future__ import annotations

import torch
from mattergen.diffusion.data.batched_data import BatchedData
from torch import Tensor
from torch_scatter import scatter_mean


def add_velocity(batch: BatchedData) -> BatchedData:
    """Add a zero-initialised velocity field to the batch.

    The velocity is sampled from a zero-CoG Gaussian at training time
    (handled by the corruption), but we need the field to exist on the
    clean batch so that ``batch["vel"]`` doesn't fail.

    Args:
        batch: A :class:`ChemGraph` batch with ``pos``.

    Returns:
        Batch with an additional ``vel`` field of shape ``(N, 3)`` = 0.

    """
    pos = batch["pos"]
    vel = torch.zeros_like(pos)
    return batch.replace(vel=vel)
