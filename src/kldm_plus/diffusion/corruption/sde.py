import math

import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE, VESDE, VPSDE
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: F811, RUF100, TC001, TC002
from torch import Tensor

from kldm_plus.diffusion.corruption.utils import scatter_center, sigma_norm

__all__ = [
    "VESDE",
    "VPSDE",
]


class KineticLangevinPhysics:
    """Mixin: pure kinetic Langevin physics.

    The forward SDE for velocity and (unwrapped) displacement reads:

        dr = v dt
        dv = -γ v dt + sqrt(2γ) dW_v

    with wrapped positions x_t = wrap(x_0 + r_t).

    At time t, marginal distributions are:

    * ``v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)``
    * ``r_t | v_0, v_t ~ N(mu_r(v_0, v_t, t), sigma_r(t)² I)``   (Corollary 1 and 2)

    Concrete subclasses must call this mixin's ``__init__`` and must expose:
    ``scale_pos``, ``tf``, ``gamma``, ``k_wn``, ``_n_sigmas``, ``_sigma_norms``.
    """  # noqa: RUF002

    def __init__(  # noqa: D417, PLR0913
        self,
        scale_pos: float = 1.0,
        tf: float = 2.0,
        gamma: float = 1.0,
        k_wn: int = 13,
        n_sigmas: int = 2_000,
        loss_pos_scale: float | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize physical parameters and pre-compute the sigma-norm table.

        Args:
            scale_pos: Torus period for the *corruption* (sample_pos / wrap_pos).
                Positions must live in ``[0, scale_pos)``.
            tf: Final internal time (total diffusion duration).
            gamma: Friction coefficient.
            k_wn: Number of wrapping images for the wrapped-normal approximation.
            n_sigmas: Resolution of the pre-computed sigma-norm lookup table.
            loss_pos_scale: Torus period used **only for the loss** (sigma-norm table
                and ``d_log_p_WN``).  Defaults to ``2π`` — matching kldm_frnct, which
                stores positions in ``[0, 2π)``.  This keeps ``sigma/T ≤ 0.155`` at
                ``t=1`` (versus ``0.977`` if ``T=scale_pos=1``), preventing sigma-norm
                underflow and float32 cancellation in ``d_log_p_WN``.

        """
        super().__init__(**kwargs)

        if gamma <= 0.0:
            msg = "gamma must be positive"
            raise ValueError(msg)

        self.scale_pos = scale_pos
        self.loss_pos_scale = loss_pos_scale if loss_pos_scale is not None else 2.0 * math.pi
        self.tf = tf
        self.gamma = gamma
        self.k_wn = k_wn
        self._n_sigmas = n_sigmas

        # Pre-compute the sigma-norm lookup table using T=scale_pos (native torus period).
        # This is consistent with the loss, which evaluates d_log_p_WN with T=scale_pos.
        # Set n_sigmas=0 to skip pre-computation.
        if n_sigmas > 0:
            with torch.no_grad():
                tau_linspace = torch.linspace(0.0, tf, n_sigmas)
                sigma_r_vals = self._sigma_r_tau(tau_linspace)
                self._sigma_norms: Tensor | None = sigma_norm(sigma_r_vals, T=self.scale_pos, N=k_wn)
        else:
            self._sigma_norms = None

    # ------------------------------------------------------------------
    # Internal time rescaling
    # ------------------------------------------------------------------

    def tau(self, t: Tensor) -> Tensor:
        """Map external scheduler time t ∈ [0, 1] → internal time τ ∈ [0, tf]."""
        return t * self.tf

    # ------------------------------------------------------------------
    # Position-process statistics
    # ------------------------------------------------------------------

    def _sigma_r_tau(self, tau: Tensor) -> Tensor:
        """Displacement std at *internal* time τ."""
        gamma = self.gamma
        return torch.sqrt(
            torch.clamp(
                (2.0 / gamma**2) * (gamma * tau - 2.0 * torch.tanh(gamma * tau / 2.0)),
                min=1e-12,
            )
        )

    def _sigma_norm_t(self, t: Tensor) -> Tensor:
        """Look up E[||score_WN||²] for external time t ∈ [0, 1] from the pre-computed table."""
        if self._sigma_norms is None:
            msg = "sigma_norms table is disabled (n_sigmas=0); cannot call _sigma_norm_t"
            raise RuntimeError(msg)
        tau = self.tau(t)
        n = len(self._sigma_norms)

        idx = torch.round(tau / self.tf * n).long() - 1
        idx = idx.clamp(0, n - 1)

        return self._sigma_norms.to(t.device)[idx]

    # ------------------------------------------------------------------
    # Torus geometry
    # ------------------------------------------------------------------

    @staticmethod
    def wrap_pos(x: Tensor, period: float = 1.0) -> Tensor:
        """Wrap group element (position) into [0, period) — torus expm."""
        return torch.remainder(x, period)

    @staticmethod
    def wrap_disp(x: Tensor, period: float = 1.0) -> Tensor:
        """Wrap Lie algebra element (displacement) into [-period/2, period/2) — torus logm."""
        return torch.remainder(x + period / 2, period) - period / 2

    # ------------------------------------------------------------------
    # Displacement (position) marginal
    # ------------------------------------------------------------------

    def displacement_marginal(
        self,
        v0: Tensor,
        t: Tensor,
        vt: Tensor,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """Mean and std of the displacement r_τ conditioned on (v_0, v_τ).

            r_τ | v_0, v_τ ~ N(mu_r_τ, sigma_r_τ² I)

        Uses the numerically stable tanh form to avoid cancellation at small t.
        """
        gamma = self.gamma
        t = self.tau(t)
        t = maybe_expand(x=t, batch=batch_idx, like=v0)

        mu_r_t = (torch.tanh(gamma * t / 2.0) / gamma) * (vt + v0)
        sigma_r_t = torch.sqrt(
            torch.clamp(
                (2.0 / gamma**2.0) * (gamma * t - 2.0 * torch.tanh((gamma * t) / 2.0)),
                min=1e-12,
            )
        )
        return mu_r_t, sigma_r_t

    def sample_pos(
        self,
        x0: Tensor,
        v0: Tensor,
        vt: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample noisy torus position x_t given (x_0, v_0, v_t).

        Steps:
            1. Draw zero-CoG Gaussian noise z.
            2. r_t = wrap_disp(mu_r_t + sigma_r_t * z)  — Lie algebra element.
            3. x_t = wrap_pos(x_0 + r_t)                — group element.
        """
        mu_r_t, sigma_r_t = self.displacement_marginal(v0=v0, t=t, vt=vt, batch_idx=batch_idx)
        z = torch.randn_like(x0)

        if batch_idx is not None:
            z = scatter_center(z, index=batch_idx)
        r_t = self.wrap_disp(mu_r_t + sigma_r_t * z, self.scale_pos)

        return self.wrap_pos(x0 + r_t, period=self.scale_pos)


class KineticLangevinSDE(KineticLangevinPhysics, SDE):
    """Mattergen ``SDE`` for the **velocity** field of kinetic Langevin dynamics.

    The velocity follows an OU process (independent of position):

        v_τ | v_0 ~ N(exp(-γτ) v_0, (1 - exp(-2γτ)) I)

    This class implements the mattergen ``SDE`` / ``Corruption`` interface for that
    single field.  Coupling with the position field is handled externally by
    ``KineticMultiCorruption``.

    External time t ∈ [0, 1] maps to internal time τ = t * tf via ``tau``.
    SDE coefficients in external time pick up a factor of tf via the chain rule.
    """  # noqa: RUF002

    @property
    def T(self) -> float:  # noqa: N802
        """End time of the diffusion process in external scheduler time."""
        return 1.0

    def sde(
        self,
        x: Tensor,
        t: Tensor,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """OU drift and diffusion in **external** time t ∈ [0, 1].

        Physical SDE: dv = -γ v dτ + sqrt(2γ) dW_τ.
        After reparametrization τ = tf * t:  drift = -γ tf v, diffusion = sqrt(2γ tf).
        """  # noqa: RUF002
        drift = -self.gamma * self.tf * x
        diffusion = torch.full_like(input=x, fill_value=(2.0 * self.gamma * self.tf) ** 0.5)

        return drift, diffusion

    def marginal_prob(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std for velocity: v_τ | v_0 ~ N(exp(-γτ) v_0, (1-exp(-2γτ)) I)."""  # noqa: RUF002
        v0 = x
        t = self.tau(t)
        t = maybe_expand(x=t, batch=batch_idx, like=v0)

        mu_v_t = torch.exp(-self.gamma * t) * v0
        sigma_v_t = torch.sqrt((-torch.expm1(-2.0 * self.gamma * t)).clamp(min=1e-12))

        return mu_v_t, sigma_v_t

    def sample_marginal(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        """Sample v_t ~ marginal(v_0=x, t) with zero-CoG noise per crystal."""
        mean, std = self.marginal_prob(x, t, batch_idx, batch)
        device = batch_idx.device if batch_idx is not None else None
        z = torch.randn_like(x, device=device)

        if batch_idx is not None:
            z = scatter_center(z, index=batch_idx)

        return mean + std * z

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,  # noqa: ARG002
        batch_idx: B = None,
    ) -> Tensor:
        """Sample from the velocity prior p(v_T) = N(0, I), zero-CoG per crystal."""
        device = batch_idx.device if batch_idx is not None else None
        z = torch.randn(*shape, device=device)

        if batch_idx is not None:
            z = scatter_center(z, index=batch_idx)

        return z

    def prior_logp(
        self,
        z: Tensor,
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> Tensor:
        """Log-probability under the velocity prior. Used for calculating score."""
        d = z.shape[-1]
        logp = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))

        return logp - 0.5 * (z**2).sum(dim=-1)
