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

from __future__ import annotations

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

    * ``v_t | v_0 ~ N(exp(-t) v_0, (1-exp(-2t)) I)``
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
        ``prefactor * sqrt(sigma_norm)`` which stabilises training.

    """

    def __init__(
        self,
        scale_pos: float = 1.0,
        tf: float = 2.0,
        k_wn_score: int = 13,
        simplified_parameterization: bool = True,
    ):
        super().__init__()
        self.scale_pos = scale_pos
        self._tf = tf
        self.k_wn_score = k_wn_score
        self.simplified_parameterization = simplified_parameterization

    # ---- MatterGen SDE interface ------------------------------------------

    @property
    def T(self) -> float:
        return self._tf

    def sde(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Instantaneous drift and diffusion for *velocity*.

        Returns ``(f, g)`` such that ``dv = f dt + g dW``.
        Position is propagated deterministically via ``dpos = v dt``
        and must be handled by the predictor.
        """
        v = x  # SDE acts on velocity
        drift = -v
        diffusion = torch.full_like(v, 2.0**0.5)
        return drift, diffusion

    # ---- Marginal distributions -------------------------------------------

    def marginal_prob(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std for **velocity** at time *t*.

        .. math::
            v_t \\sim \\mathcal{N}(e^{-t} v_0,\\; (1-e^{-2t}) I)
        """
        t_exp = maybe_expand(t, batch_idx, x)
        mean_coeff = torch.exp(-t_exp)
        mean = mean_coeff * x
        std = torch.sqrt(1.0 - torch.exp(-2.0 * t_exp))
        return mean, std

    def _displacement_marginal(
        self,
        v0: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std of the **displacement** *r* at time *t*.

        .. math::
            \\mu_r = (1 - e^{-t}) v_0
            \\sigma_r^2 = 2t - 3 + 4 e^{-t} - e^{-2t}

        Args:
            v0: Initial velocity, shape ``(N, 3)``.
            t: Diffusion time, shape ``(B, 1)`` or broadcastable.
            batch_idx: Maps atoms → graphs.

        Returns:
            ``(mu_r, sigma_r)`` each with shape ``(N, 3)``.

        """
        t_exp = maybe_expand(t, batch_idx, v0)
        mu_r = (1.0 - torch.exp(-t_exp)) * v0
        sigma_r_sq = 2.0 * t_exp - 3.0 + 4.0 * torch.exp(-t_exp) - torch.exp(-2.0 * t_exp)
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

    def sample_pos_marginal(
        self,
        pos_0: Tensor,
        v_0: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample noisy **position** (fractional coords) at time *t*.

        ``pos_t = wrap(pos_0 + wrap(mu_r + sigma_r * eps))``
        """
        mu_r, sigma_r = self._displacement_marginal(v_0, t, batch_idx)
        eps = torch.randn_like(pos_0)
        r = mu_r + sigma_r * eps
        pos_t = _wrap(pos_0 + _wrap(r, self.scale_pos), self.scale_pos)
        return pos_t

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample velocity from the stationary distribution ``N(0, I)``."""
        return torch.randn(*shape)

    def prior_logp(
        self,
        z: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        """Log-probability under the velocity prior ``N(0, I)``."""
        d = z.shape[-1]
        logp = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        logp = logp - 0.5 * (z**2).sum(dim=-1)
        return logp

    # ---- Training target --------------------------------------------------

    def training_target(
        self,
        pos_0: Tensor,
        pos_t: Tensor,
        v_0: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Compute the training target for the score network.

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
            batch_idx: Atom → graph mapping ``(N,)``.

        Returns:
            Target tensor ``(N, 3)``.

        """
        mu_r, sigma_r = self._displacement_marginal(v_0, t, batch_idx)

        # Displacement (unwrapped difference, mapped to [-T/2, T/2])
        diff = pos_t - pos_0
        r = torch.remainder(diff + self.scale_pos / 2, self.scale_pos) - self.scale_pos / 2

        score_wn = d_log_p_wrapped_normal(r, mu_r, sigma_r, N=self.k_wn_score, T=self.scale_pos)

        if self.simplified_parameterization:
            # prefactor = sigma_r / sqrt(1 - exp(-2t))
            t_exp = maybe_expand(t, batch_idx, pos_0)
            vel_std = torch.sqrt(1.0 - torch.exp(-2.0 * t_exp))
            prefactor = sigma_r / vel_std.clamp(min=1e-8)

            sn = sigma_norm(sigma_r, T=self.scale_pos, N=self.k_wn_score)
            target = score_wn * prefactor / torch.sqrt(sn.clamp(min=1e-8))
        else:
            target = score_wn

        return target

    # ---- Reverse-time helpers (used by predictors / correctors) -----------

    def reverse_step_em(
        self,
        v: Tensor,
        pos: Tensor,
        score: Tensor,
        dt: float,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """Single Euler–Maruyama reverse step (exponential integrator).

        Args:
            v: Current velocity ``(N, 3)``.
            pos: Current position ``(N, 3)``.
            score: Model-predicted score for velocity ``(N, 3)``.
            dt: Time step size (positive, we go backward).
            batch_idx: Atom → graph mapping.

        Returns:
            ``(v_new, pos_new)``

        """
        exp_dt = torch.exp(torch.tensor(dt, device=v.device))
        expm1_dt = _expm1(torch.tensor(dt, device=v.device))
        expm1_2dt = _expm1(torch.tensor(2.0 * dt, device=v.device))

        noise = torch.randn_like(v)
        v_new = exp_dt * v + 2.0 * expm1_dt * score + torch.sqrt(expm1_2dt.abs()) * noise
        pos_new = _wrap(pos - dt * v_new, self.scale_pos)
        return v_new, pos_new

    def reverse_step_pc_predictor(
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

        alpha_t = torch.exp(-t_exp)
        alpha_s = torch.exp(-s.clamp(min=0.0))

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
        batch_idx: B = None,
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
