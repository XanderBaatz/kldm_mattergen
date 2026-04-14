"""Lattice corruption — re-exports MatterGen's LatticeVPSDE for cell 3×3."""

from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    make_noise_symmetric_preserve_variance,
)

__all__ = ["LatticeVPSDE", "make_noise_symmetric_preserve_variance"]
