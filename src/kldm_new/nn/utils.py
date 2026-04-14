"""Scatter / wrapping helpers for GNN operations."""

from __future__ import annotations

import torch
from torch import Tensor
from torch_scatter import scatter_mean


def scatter_center(pos: Tensor, index: Tensor) -> Tensor:
    """Subtract per-graph centre of mass so that positions are zero-CoG."""
    return pos - scatter_mean(pos, index=index, dim=0)[index]


def wrap(x: Tensor, period: float = 1.0) -> Tensor:
    """Wrap values into [0, period).

    Uses ``atan2(sin, cos)`` for differentiability at boundaries.
    """
    scaled = x * (2.0 * torch.pi / period)
    return torch.atan2(torch.sin(scaled), torch.cos(scaled)) / (2.0 * torch.pi / period)
