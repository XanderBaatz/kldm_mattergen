"""KLDM multi-corruption: TDM for positions + VPSDE for lattice cell.

This module provides :class:`KLDMMultiCorruption` which orchestrates
the coupled kinetic-Langevin diffusion on fractional coordinates
(via :class:`KineticLangevinSDE`) together with standard VP diffusion
on the 3×3 lattice cell matrix (via :class:`LatticeVPSDE`).

Because TDM couples position and velocity, the standard ``sample_marginal``
dispatch in MatterGen's :class:`MultiCorruption` is not sufficient — we
override it to handle velocity sampling and position wrapping.
"""

from __future__ import annotations

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE
from mattergen.diffusion.data.batched_data import BatchedData
from torch import Tensor

from kldm_new.diffusion.tdm import KineticLangevinSDE


class KLDMMultiCorruption(MultiCorruption):
    """Multi-corruption for KLDM: TDM (pos+vel) + LatticeVPSDE (cell).

    The constructor accepts a ``KineticLangevinSDE`` for position diffusion
    and any ``SDE`` for lattice diffusion.  Velocity is treated as an auxiliary
    field stored on the batch under the key ``"vel"``.

    Parameters
    ----------
    pos_sde : KineticLangevinSDE
        Kinetic Langevin SDE for fractional coordinates.
    cell_sde : SDE
        VP-SDE for the cell matrix (typically :class:`LatticeVPSDE`).

    """

    def __init__(
        self,
        pos_sde: KineticLangevinSDE,
        cell_sde: SDE,
    ):
        # Register cell SDE in the standard MatterGen dict
        super().__init__(sdes={"cell": cell_sde})
        self._pos_sde = pos_sde

    @property
    def pos_sde(self) -> KineticLangevinSDE:
        return self._pos_sde

    @property
    def cell_sde(self) -> SDE:
        return self.sdes["cell"]

    @property
    def corrupted_fields(self) -> list[str]:
        return ["pos", "vel", "cell"]

    @property
    def T(self) -> float:  # type: ignore[override]
        # Both SDEs must share the same T
        return self._pos_sde.T

    def sample_marginal(self, batch: BatchedData, t: Tensor) -> BatchedData:  # type: ignore[override]
        """Corrupt *all* fields in-place: pos (wrapped), vel (Gaussian), cell (VP).

        This overrides the parent because TDM position sampling requires
        knowing the *clean* velocity and position simultaneously.

        Args:
            batch: Clean batch (must have ``pos``, ``vel``, ``cell`` fields).
            t: Diffusion time ``(B, 1)``.

        Returns:
            New :class:`BatchedData` with corrupted fields.

        """
        # -- Atom-level indexing --
        pos_batch_idx = batch.get_batch_idx("pos")  # (N,)

        # -- Clean tensors --
        pos_0: Tensor = batch["pos"]
        vel_0: Tensor = batch["vel"]
        cell_0: Tensor = batch["cell"]

        # -- Velocity: standard Gaussian marginal --
        vel_t = self._pos_sde.sample_marginal(vel_0, t, batch_idx=pos_batch_idx, batch=batch)

        # -- Position: wrapped displacement marginal --
        pos_t = self._pos_sde.sample_pos_marginal(pos_0, vel_0, t, batch_idx=pos_batch_idx)

        # -- Cell: delegate to LatticeVPSDE --
        cell_t = self.cell_sde.sample_marginal(cell_0, t, batch_idx=None, batch=batch)

        return batch.replace(pos=pos_t, vel=vel_t, cell=cell_t)
