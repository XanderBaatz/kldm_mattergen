from __future__ import annotations

from typing import TYPE_CHECKING

from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE  # noqa: TC001
from mattergen.diffusion.corruption.multi_corruption import Diffusable, MultiCorruption

if TYPE_CHECKING:
    from torch import Tensor

    from mattergen.common.diffusion.corruption import LatticeVPSDE


class KLDMCorruption(MultiCorruption):
    """MultiCorruption that handles coupled (pos, vel) kinetic Langevin dynamics before delegating independent fields (cell) to the parent.

    ``KineticLangevinSDE`` is intentionally **not** registered in the parent's
    ``sdes`` dict.  Registering it there would corrupt ``vel`` independently of
    ``pos``, breaking the kinetic Langevin coupling:

        1. ``v_t`` is sampled first.
        2. ``pos_t`` is sampled conditioned on that *same* ``v_t``.

    ``MultiCorruption.sample_marginal`` is a plain dict comprehension - it has no
    mechanism to share an intermediate sample between two field corruptions.
    Owning the coupled step here and only forwarding independent fields to the
    parent is the cleanest solution.

    Note: ``kinlang.T == 1.0 == cell_sde.T``.  The internal kinetic Langevin
    time horizon (``tf=2.0``) is encapsulated inside ``KineticLangevinSDE`` via
    ``tau``; the external scheduler always operates on ``[0, 1]``.
    """

    def __init__(
        self,
        torus_sde: KineticLangevinSDE,
        cell_sde: LatticeVPSDE,
    ) -> None:
        """Initialize KLDM coupled multi corruption.

        Args:
            torus_sde (KineticLangevinSDE): kinetic Langevin SDE
            cell_sde (LatticeVPSDE): VP SDE or LatticeVPSDE

        """
        # Only independent fields go to parent.
        # kinlang.T == 1.0 == cell_sde.T, so the parent's T-equality assertion holds.
        super().__init__(sdes={"cell": cell_sde})
        self.torus_sde = torus_sde

    @property
    def corrupted_fields(self) -> list[str]:
        """All corrupted fields, including the coupled (pos, vel) pair."""
        return ["pos", "vel", *super().corrupted_fields]

    def sample_marginal(self, batch: Diffusable, t: Tensor) -> Diffusable:
        """Corrupt all fields.

        Steps
        -----
        1. Sample ``v_t`` from the velocity OU marginal.
        2. Sample ``pos_t`` conditioned on the *same* ``v_t`` (coupling preserved).
        3. Delegate remaining independent fields (``cell``) to parent.
        """
        batch_idx = batch.get_batch_idx("pos")
        v0 = batch["vel"]

        # Coupled step - v_t and pos_t share the same noise draw
        v_t = self.torus_sde.sample_marginal(x=v0, t=t, batch_idx=batch_idx, batch=batch)
        x_t = self.torus_sde.sample_pos(x0=batch["pos"], v0=v0, vt=v_t, t=t, batch_idx=batch_idx)

        # Independent fields (cell, ...) via parent
        pre_noisy = batch.replace(vel=v_t, pos=x_t)
        return super().sample_marginal(pre_noisy, t)
