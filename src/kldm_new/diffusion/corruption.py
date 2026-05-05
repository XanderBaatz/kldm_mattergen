"""KLDM multi-corruption: TDM for positions/velocities + VPSDE for lattice cell.

This module provides :class:`KLDMMultiCorruption` which orchestrates
the coupled kinetic-Langevin **forward process** on fractional coordinates
(via :class:`~kldm_new.diffusion.tdm.KineticLangevinSDE`) together with
standard VP diffusion on the 3x3 lattice matrix (via  # noqa: RUF001
:class:`~kldm_new.diffusion.lattice_sde.LatticeSubVPSDE`).

Because TDM couples position and velocity, the standard ``sample_marginal``
dispatch in MatterGen's :class:`~mattergen.diffusion.corruption.multi_corruption.MultiCorruption`
is not sufficient — we override it to handle the joint (vel, pos) forward
corruption correctly using the conditional displacement marginal.
"""

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE  # noqa: RUF100, TC001, TC002
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: RUF100, TC001, TC002
from torch import Tensor  # noqa: TC002

from kldm_new.diffusion.tdm import KineticLangevinSDE  # noqa: TC001


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
    ) -> None:
        """Initialize the multi-corruption with the given SDEs."""
        # Register cell SDE in the standard MatterGen dict
        super().__init__(sdes={"cell": cell_sde})
        self._pos_sde = pos_sde

    @property
    def pos_sde(self) -> KineticLangevinSDE:
        """Accessor for the position SDE."""
        return self._pos_sde

    @property
    def cell_sde(self) -> SDE:
        """Accessor for the cell SDE."""
        return self.sdes["cell"]

    @property
    def corrupted_fields(self) -> list[str]:
        """Fields corrupted by this multi-corruption."""
        return ["pos", "vel", "cell"]

    @property
    def T(self) -> float:  # type: ignore[override]  # noqa: N802
        """Diffusion time horizon. Both SDEs must share the same T."""
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

        # -- Position: conditional displacement marginal r | v_0, v_t (Corollary) --
        # Passing vel_t triggers the conditional form in KineticLangevinSDE.sample_pos.
        pos_t = self._pos_sde.sample_pos(pos_0, vel_0, vel_t, t, batch_idx=pos_batch_idx)

        # -- Cell: delegate to LatticeVPSDE --
        cell_t = self.cell_sde.sample_marginal(cell_0, t, batch_idx=None, batch=batch)

        return batch.replace(pos=pos_t, vel=vel_t, cell=cell_t)
