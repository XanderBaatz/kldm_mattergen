from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003

import torch
from mattergen.diffusion.corruption.corruption import B, Corruption
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE  # noqa: TC002
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch import Tensor

from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE  # noqa: TC001


class KinLangevinPosCoupled(Corruption):
    """Minimal ``Corruption`` for the ``pos`` field under kinetic Langevin dynamics.

    Position under kinetic Langevin dynamics has **no independent noise source**;
    it is deterministically driven by velocity.  This class satisfies the mattergen
    ``Corruption`` protocol so the PC sampler can include ``"pos"`` in its update
    loop via ``KinLangevinPosPredictor``.

    * ``prior_sampling`` — uniform random fractional coordinates on ``[0, 1)^3``.
    * ``sample_marginal`` — identity (the real coupled sampling is done by
      ``KineticMultiCorruption.sample_marginal``).
    * ``T`` — 1.0 (matches all other fields).
    """

    def __init__(self, kinlang: KineticLangevinSDE) -> None:
        self._kinlang = kinlang

    @property
    def T(self) -> float:  # noqa: N802
        return 1.0

    def marginal_prob(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "pos marginal is a wrapped-normal, not Gaussian — use KineticMultiCorruption.sample_marginal for the coupled sample."
        )

    def prior_sampling(
        self,
        shape: tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx: B = None,
    ) -> Tensor:
        """Uniform random fractional coordinates on ``[0, 1)^3``."""
        return torch.rand(shape)

    def prior_logp(
        self,
        z: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        raise NotImplementedError("pos prior log-p not implemented for kinetic Langevin")

    def sample_marginal(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        """Identity — the real coupled sampling is done by ``KineticMultiCorruption``."""
        return x


class KineticMultiCorruption(MultiCorruption):
    """``MultiCorruption`` that handles the coupled (pos, vel) kinetic Langevin forward process.

    ``vel`` is registered as a standard SDE field (OU process) and sampled by the
    parent ``MultiCorruption``.  ``pos`` is **not** registered in the parent; it is
    sampled here after ``vel_t`` is realized, preserving the coupling:

        vel_t | vel_0 ~ N(exp(-γt) vel_0, σ_v(t)² I)       [parent, independent]
        pos_t | pos_0, vel_0, vel_t ~ WrappedNormal(...)    [here, coupled]

    Other fields (e.g. ``cell``, ``atomic_numbers``) are passed through to the
    parent and sampled independently.  The lattice is always stored as ``cell``
    regardless of its shape (3x3 or 6D).
    """  # noqa: RUF002

    def __init__(
        self,
        kinetic_sde: KineticLangevinSDE,
        sdes: Mapping[str, SDE] | None = None,
        discrete_corruptions: Mapping | None = None,
    ) -> None:
        """Initialize kinetic MultiCorruption with coupled (pos + vel).

        Args:
            kinetic_sde: Kinetic Langevin SDE — drives the OU velocity process and
                the coupled position sampling.
            sdes: SDEs for independent fields.  Typically
                ``{"cell": LatticeVPSDE(...)}`` (3x3) or
                ``{"cell": VPSDE(...)}`` (6D).
            discrete_corruptions: Discrete corruption processes (e.g. masking for
                ``atomic_numbers``).

        """
        self.kinetic_sde = kinetic_sde
        # Coupled pos "corruption" — identity on forward, uniform prior for sampling.
        # Must be set BEFORE super().__init__ because the corruptions property override
        # is invoked during parent initialisation (T-value assertion).
        self._pos_coupled = KinLangevinPosCoupled(kinetic_sde)
        super().__init__(
            sdes={"vel": kinetic_sde, **(sdes or {})},
            discrete_corruptions=discrete_corruptions or {},
        )

    # ------------------------------------------------------------------
    # Expose "pos" in the corruptions dict so the PC sampler can route
    # KinLangevinPosPredictor updates to the position field.
    # ------------------------------------------------------------------

    @property
    def corruptions(self) -> Mapping[str, Corruption]:  # type: ignore[override]
        combined = {**super().corruptions, "pos": self._pos_coupled}
        return {k: combined[k] for k in sorted(combined)}

    # ------------------------------------------------------------------
    # Override: resolve batch indices for known field types without
    # requiring a custom ChemGraph subclass.
    # ------------------------------------------------------------------

    def _get_batch_indices(
        self,
        batch: BatchedData,
    ) -> dict[str, torch.Tensor]:
        indices = {}
        for k in self.corrupted_fields:
            try:
                indices[k] = batch.get_batch_idx(k)
            except (
                NotImplementedError,
                KeyError,
            ):
                # Fallback: node-level field not registered in ChemGraph (e.g. 'vel')
                # use the same node→graph mapping as 'pos'
                indices[k] = batch.get_batch_idx("pos")
        return indices

    # ------------------------------------------------------------------
    # Override: coupled (pos, vel) sampling
    # ------------------------------------------------------------------

    def sample_marginal(
        self,
        batch: BatchedData,
        t: torch.Tensor,
    ) -> BatchedData:
        """Sample all noisy fields, handling the kinetic Langevin coupling.

        Steps:
            1. Initialize ``vel`` to zero if absent (simplified parameterization: v₀ = 0).
            2. Let parent sample ``vel_t`` (OU) and all other registered fields.
            3. Sample ``pos_t`` from the wrapped-normal conditioned on the realized ``vel_t``.
        """
        # Simplified parameterization: v_0 = 0
        try:
            if batch["vel"] is None:
                raise KeyError  # noqa: TRY301
        except (
            KeyError,
            AttributeError,
        ):
            batch = batch.replace(vel=torch.zeros_like(batch["pos"]))

        # Parent samples vel_t (independent OU) + cell, atomic_numbers, ...
        noisy_batch = super().sample_marginal(batch, t)

        # Coupled: sample pos_t using the realized vel_t
        pos_t = self.kinetic_sde.sample_pos(
            x0=batch["pos"],
            v0=batch["vel"],
            vt=noisy_batch["vel"],
            t=t,
            batch_idx=batch.get_batch_idx("pos"),
        )

        return noisy_batch.replace(pos=pos_t)
