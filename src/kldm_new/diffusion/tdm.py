"""Kinetic Langevin (TDM) SDE on the torus — MatterGen-native implementation.

This module implements the *Torus Diffusion Model* (TDM) used for
fractional-coordinate diffusion in KLDM.  The forward process is a
**kinetic Langevin** diffusion that couples an auxiliary **velocity** field
to the atom **positions** (fractional coordinates on [0, 1)^3).

The class :class:`KineticLangevinSDE` extends MatterGen's :class:`SDE`
interface so that it can be plugged into :class:`MultiCorruption` and
the standard :class:`DiffusionModule` training loop.  However, because
position and velocity are *coupled*, the predictor / corrector steps
require custom implementations (see ``predictors.py`` / ``correctors.py``).
"""

import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE
from torch import Tensor

from kldm_new.diffusion import d_log_p_wrapped_normal, sigma_norm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wrap(x: Tensor, period: float = 1.0) -> Tensor:
    """Wrap *x* into [0, period)."""
    return torch.remainder(x, period)


def _expm1(x: Tensor) -> Tensor:
    """exp(x) - 1, numerically stable."""
    return torch.expm1(x)


# ---------------------------------------------------------------------------
# Kinetic Langevin SDE
# ---------------------------------------------------------------------------


class KineticLangevinSDE(SDE):
    """Kinetic Langevin diffusion on the 3-D torus.

    The forward SDE for velocity and (unwrapped) displacement reads::

        dv = -v dt + sqrt(2) dW_v
        dr = v dt

    with wrapped positions ``pos_t = wrap(pos_0 + r_t)``.

    At time *t* the **marginal** distributions are:

    * ``v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)``
    * ``r_t | v_0 ~ N(mu_r(v_0, t), sigma_r(t)² I)``   (see code for mu_r, sigma_r)
    * ``pos_t = wrap(pos_0 + wrap(r_t))``  where ``wrap`` is mod *scale_pos*.

    The model predicts the **score of the wrapped-normal** for *r* (equiv. to
    the joint score w.r.t. ``(v, pos)``).  A simplified-parameterization
    target is returned by :meth:`training_target`.

    Parameters
    ----------
    scale_pos : float
        Period of the torus (default ``1.0`` for fractional coordinates).
    tf : float
        Terminal diffusion time *T*.
    k_wn_score : int
        Number of periodic images for the wrapped-normal score.
    simplified_parameterization : bool
        If ``True`` (default), the training target is re-weighted by
        ``prefactor * sqrt(sigma_norm)`` which stabilizes training.

    """

    def __init__(  # noqa: PLR0913
        self,
        scale_pos: float = 1.0,
        tf: float = 2.0,
        k_wn_score: int = 13,
        simplified_parameterization: bool = True,  # noqa: FBT001, FBT002
        gamma: float = 1.0,
        conditional_velocity: bool = True,  # noqa: FBT001, FBT002
        sigma_norm_table_size: int = 2000,
    ) -> None:
        """Initialize the KineticLangevinSDE."""
        super().__init__()
        self.scale_pos = scale_pos
        self._tf = tf
        self.k_wn_score = k_wn_score
        self.simplified_parameterization = simplified_parameterization
        self.gamma = gamma
        self.conditional_velocity = conditional_velocity

        if gamma <= 0.0:  # else we have imaginary diffusion coefficients and the SDE is not well-defined
            msg = "gamma must be positive"
            raise ValueError(msg)

        if simplified_parameterization:
            # Precompute sigma_norm on a log-spaced grid using float64 for
            # accuracy. Registered as buffers so they move to GPU automatically.
            # At training time we interpolate instead of re-estimating via MC,
            # which eliminates float32 rounding noise and per-step variance.
            sigma_grid = torch.logspace(-6, 0, sigma_norm_table_size, dtype=torch.float64)
            sn_grid = sigma_norm(
                sigma_grid,
                T=scale_pos,
                N=k_wn_score,
                sn=50_000,  # high-accuracy precomputation, done once
            ).float()
            self.register_buffer("_sn_log_sigma", sigma_grid.log().float())
            self.register_buffer("_sn_values", sn_grid)

    # ---- MatterGen SDE interface ------------------------------------------

    @property
    def T(self) -> float:  # noqa: N802
        """The end time of the diffusion process."""
        return self._tf

    def sde(
        self,
        x: Tensor,
        t: Tensor,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Instantaneous drift and diffusion for *velocity*.

        Returns ``(f, g)`` such that ``dv = f dt + g dW``.
        Position is propagated deterministically via ``dpos = v dt``
        and must be handled by the predictor.
        """
        # SDE acts on velocity (x=v)
        drift = -self.gamma * x
        diffusion = torch.full_like(x, (2.0 * self.gamma) ** 0.5)
        return drift, diffusion

    # ---- Marginal distributions -------------------------------------------

    def marginal_prob(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        r"""Marginal mean and std for velocity at time t.

        v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)
        """  # noqa: RUF002
        t_exp = maybe_expand(t, batch_idx, x)
        mean = torch.exp(-self.gamma * t_exp) * x
        std = torch.sqrt((1.0 - torch.exp(-2.0 * self.gamma * t_exp)).clamp(min=1e-12))
        return mean, std

    def _displacement_marginal(
        self,
        v0: Tensor,
        t: Tensor,
        vt: Tensor | None = None,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Marginal mean and std of the **displacement** *r* at time *t*.

        Conditional (conditional_velocity=True, vt required) — Corollary:
            mu_r  = (1 - exp(-γt)) / (γ(1 + exp(-γt))) · (vt + v0)
            σ_r²  = (2/γ²)(γt + 4γ/(exp(γt)+1) - 2γ)

        Marginal (conditional_velocity=False) — Lemma:
            mu_r  = (1 - exp(-γt)) / γ · v0
            σ_r²  = (2/γ²)(γt - 2(1-exp(-γt)) + ½(1-exp(-2γt)))

        Args:
            v0: Initial velocity, shape ``(N, 3)``.
            t: Diffusion time, shape ``(B, 1)`` or broadcastable.
            vt: Velocity at time t
            batch_idx: Maps atoms → graphs.

        Returns:
            ``(mu_r, sigma_r)`` each with shape ``(N, 3)``.

        """  # noqa: RUF002
        gamma = self.gamma
        t_exp = maybe_expand(t, batch_idx, v0)
        exp_gt = torch.exp(-gamma * t_exp)

        if self.conditional_velocity:
            if vt is None:
                msg = "vt must be provided when conditional_velocity is True"
                raise ValueError(msg)

            # Conditional
            mu_r = ((1.0 - exp_gt) / (gamma * (1.0 + exp_gt))) * (v0 + vt)

            # Numerically stable form using tanh identity:
            #   t + 4/(e^t+1) - 2  ≡  t - 2·tanh(t/2)
            # The naive form suffers catastrophic cancellation in float32 at
            # small t (e.g. t=0.01 gives sigma_r_sq=0 instead of ~1e-7),
            # producing score targets 10,000× too large on GPU.
            y = gamma * t_exp
            sigma_r_sq = (2.0 / gamma**2) * (y - 2.0 * torch.tanh(y / 2.0))
        else:
            # Marginal
            mu_r = ((1.0 - exp_gt) / gamma) * v0

            # Numerically stable form using expm1 identity:
            #   t - 2(1-e^{-t}) + ½(1-e^{-2t})  ≡  t + 2·expm1(-t) - ½·expm1(-2t)
            # expm1 is accurate at small arguments where direct subtraction cancels.
            y = gamma * t_exp
            sigma_r_sq = (2.0 / gamma**2) * (y + 2.0 * torch.expm1(-y) - 0.5 * torch.expm1(-2.0 * y))

        sigma_r = torch.sqrt(torch.clamp(sigma_r_sq, min=1e-12))

        return mu_r, sigma_r

    def sample_marginal(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        """Sample noisy **velocity** at time *t* (standard Gaussian diffusion).

        Position sampling must be done explicitly via :meth:`sample_pos_marginal`
        because it depends on both ``v_0`` and ``pos_0``.
        """
        mean, std = self.marginal_prob(x, t, batch_idx, batch)
        return mean + std * torch.randn_like(x)

    def sample_pos(
        self,
        pos_0: Tensor,
        v_0: Tensor,
        v_t: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample noisy position at time t.

        Delegates to :meth:`_displacement_marginal` which selects the correct
        distribution based on ``conditional_velocity``:

        - ``conditional_velocity=True`` (Corollary): ``Y_t | v_t, v_0`` - requires ``v_t``.
        - ``conditional_velocity=False`` (Lemma): ``Y_t | v_0`` - ``v_t`` is ignored.

        Raises
        ------
        ValueError
            If ``conditional_velocity=True`` but ``v_t`` is ``None``.

        """
        mu_r, sigma_r = self._displacement_marginal(v0=v_0, t=t, vt=v_t, batch_idx=batch_idx)
        eps = torch.randn_like(pos_0)
        r = mu_r + sigma_r * eps
        return _wrap(pos_0 + _wrap(r, self.scale_pos), self.scale_pos)

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
    ) -> Tensor:
        """Sample velocity from the stationary distribution ``N(0, I)``."""
        return torch.randn(*shape)

    def prior_logp(
        self,
        z: Tensor,
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> Tensor:
        """Log-probability under the velocity prior ``N(0, I)``."""
        d = z.shape[-1]
        logp = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        return logp - 0.5 * (z**2).sum(dim=-1)

    # ---- Training target --------------------------------------------------

    def _prefactor_t(self, t_exp: Tensor) -> Tensor:
        r"""Prefactor :math:`\frac{1-e^{-\gamma t}}{\gamma(1+e^{-\gamma t})}`.

        This is :math:`\partial\mu_r / \partial\mathbf{v}_t` in the conditional
        displacement mean (Corollary of Lemma — general :math:`\gamma`):

        .. math::
            \mu_{r|v}(t) = \frac{1-e^{-\gamma t}}{\gamma(1+e^{-\gamma t})}
                           (\mathbf{v}_t + \mathbf{v}_0)

        At :math:`\gamma=1` this reduces to the original TDM expression
        :math:`(1-e^{-t}) / (1+e^{-t})`.
        """
        exp_gt = torch.exp(-self.gamma * t_exp)
        return (1.0 - exp_gt) / (self.gamma * (1.0 + exp_gt).clamp(min=1e-8))

    def _lookup_sigma_norm(self, sigma: Tensor) -> Tensor:
        """Interpolate sigma_norm from the precomputed lookup table.

        Uses linear interpolation in log-sigma space, which is accurate
        because sigma_norm is smooth and monotone.  Values outside the
        table range are clamped to the nearest endpoint.

        Args:
            sigma: 1-D tensor of sigma values (all positive).

        Returns:
            Tensor of the same shape with sigma_norm estimates.

        """
        log_sigma = sigma.clamp(min=1e-7).log()
        # searchsorted returns insertion index in sorted array
        idx = torch.searchsorted(self._sn_log_sigma, log_sigma).clamp(1, len(self._sn_log_sigma) - 1)
        lo, hi = idx - 1, idx
        log_s_lo = self._sn_log_sigma[lo]
        log_s_hi = self._sn_log_sigma[hi]
        # Linear interpolation weight in log-sigma space
        w = ((log_sigma - log_s_lo) / (log_s_hi - log_s_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
        return self._sn_values[lo] * (1.0 - w) + self._sn_values[hi] * w

    def training_target(  # noqa: PLR0913
        self,
        pos_0: Tensor,
        pos_t: Tensor,
        v_0: Tensor,
        t: Tensor,
        v_t: Tensor | None = None,
        batch_idx: B = None,
    ) -> Tensor:
        r"""Compute the training target for the score network.

        The target is the **score of the wrapped-normal displacement**:

        .. math::
            \\text{target} = \\nabla_r \\log p_{WN}(r \\mid \\mu_r, \\sigma_r)

        Under simplified parameterization (default), this is further re-scaled
        by ``prefactor / sqrt(sigma_norm)`` so that the model directly predicts
        a quantity with unit-ish variance.

        Args:
            pos_0: Clean positions ``(N, 3)``.
            pos_t: Noisy positions ``(N, 3)``.
            v_0: Clean velocities ``(N, 3)``.
            t: Diffusion time ``(B, 1)``.
            v_t: Noisy velocities at time t (required if conditional_velocity=True).
            batch_idx: Atom → graph mapping ``(N,)``.

        Returns:
            Target tensor ``(N, 3)``.

        """
        t_exp = maybe_expand(t, batch_idx, pos_0)

        mu_r, sigma_r = self._displacement_marginal(
            v0=v_0,
            t=t,
            vt=v_t if self.conditional_velocity else None,
            batch_idx=batch_idx,
        )

        # Displacement mapped to [-scale/2, scale/2]
        diff = pos_t - pos_0
        r = torch.remainder(diff + self.scale_pos / 2, self.scale_pos) - self.scale_pos / 2

        score_wn = d_log_p_wrapped_normal(r, mu_r, sigma_r, N=self.k_wn_score, T=self.scale_pos)

        # Apply the prefactor (matches original TDM target_pos_t scaling)
        prefactor = self._prefactor_t(t_exp)
        target = prefactor * score_wn

        if self.simplified_parameterization:
            # sigma_r has shape (N_atoms, 3) but sigma only depends on t
            # (one per structure), so there are at most batch_size unique values.
            # Computing sigma_norm over all N_atoms*3 entries creates a tensor
            # of shape (2N+1, sn, N_atoms*3) that easily OOMs on GPU.
            # Instead, de-duplicate → lookup from precomputed table → broadcast back.
            sigma_flat = sigma_r.reshape(-1)
            sigma_unique, inv_idx = torch.unique(sigma_flat, return_inverse=True)
            sn_unique = self._lookup_sigma_norm(sigma_unique)
            sn = sn_unique[inv_idx].reshape(sigma_r.shape)
            target = target / torch.sqrt(sn.clamp(min=1e-8))

        return target

    # ---- Reverse-time helpers (used by predictors / correctors) -----------

    def reverse_step_em(
        self,
        v: Tensor,
        pos: Tensor,
        score: Tensor,
        dt: float,
        batch_idx: B = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Single Euler-Maruyama reverse step (exponential integrator).

        Args:
            v: Current velocity ``(N, 3)``.
            pos: Current position ``(N, 3)``.
            score: Model-predicted score for velocity ``(N, 3)``.
            dt: Time step size (positive, we go backward).
            batch_idx: Atom → graph mapping.

        Returns:
            ``(v_new, pos_new)``

        """
        gamma = self.gamma
        exp_dt = torch.exp(torch.tensor(gamma * dt, device=v.device))
        expm1_dt = _expm1(torch.tensor(gamma * dt, device=v.device))
        expm1_2dt = _expm1(torch.tensor(2.0 * gamma * dt, device=v.device))

        noise = torch.randn_like(v)
        v_new = exp_dt * v + 2.0 * expm1_dt * score + torch.sqrt(expm1_2dt.abs()) * noise
        pos_new = _wrap(pos - dt * v_new, self.scale_pos)
        return v_new, pos_new

    def reverse_step_pc_predictor(  # noqa: PLR0913
        self,
        v: Tensor,
        pos: Tensor,
        score: Tensor,
        t: Tensor,
        dt: float,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """DDIM-like predictor step for TDM.

        Uses the ratio ``alpha_t / alpha_s`` (where ``s = t - dt``) to
        deterministically update velocity, then propagates position.
        """
        t_exp = maybe_expand(t, batch_idx, v)
        s = t_exp - dt  # target time

        alpha_t = torch.exp(-self.gamma * t_exp)
        alpha_s = torch.exp(-self.gamma * s.clamp(min=0.0))

        sigma_t = torch.sqrt((1.0 - alpha_t**2).clamp(min=1e-12))
        sigma_s = torch.sqrt((1.0 - alpha_s**2).clamp(min=1e-12))

        # "Predicted x0" = (v - sigma_t * score) / alpha_t
        v0_hat = (v - sigma_t * score) / alpha_t.clamp(min=1e-8)
        v_new = alpha_s * v0_hat + sigma_s * score

        pos_new = _wrap(pos - dt * v_new, self.scale_pos)
        return v_new, pos_new

    def reverse_step_pc_corrector(
        self,
        v: Tensor,
        pos: Tensor,
        score: Tensor,
        tau: float = 0.5,
        batch_idx: B = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Langevin corrector on **velocity** with adaptive step size.

        Step size ``delta = tau / mean(||score||²)``.
        """
        score_norm_sq = (score**2).mean()
        delta = tau / score_norm_sq.clamp(min=1e-8)

        noise = torch.randn_like(v)
        v_new = v + delta * score + (2.0 * delta).sqrt() * noise
        # Position unchanged during corrector (only velocity is corrected)
        return v_new, pos
