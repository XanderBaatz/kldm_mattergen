"""Custom predictors for KLDM reverse-time sampling.

Provides:

* :class:`TDMEMPredictor` -- exponential-integrator (Euler-Maruyama) predictor
  for the coupled kinetic-Langevin velocity process.
* :class:`TDMDDIMPredictor` — DDIM-like deterministic predictor.

Both classes update **velocity only**; the :class:`~kldm_new.diffusion.sampling.KLDMSampler`
then propagates position via ``pos_new = wrap(pos + dt * vel_new)``.

The reverse-step mathematics is self-contained in each class; no delegation
to :class:`~kldm_new.diffusion.tdm.KineticLangevinSDE` is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.sampling.predictors import Predictor
from torch import Tensor

from kldm_new.diffusion.tdm import KineticLangevinSDE

if TYPE_CHECKING:
    from mattergen.diffusion.sampling.predictors_correctors import SampleAndMean


def _expm1(x: Tensor) -> Tensor:
    return torch.expm1(x)


class TDMEMPredictor(Predictor):
    r"""Exponential-integrator (Euler-Maruyama) predictor for TDM velocity.

    Implements the stochastic reverse step:

    .. math::
        v_{s} = e^{\\gamma\\,\\Delta t}\\,v_t
                + 2\\,(e^{\\gamma\\,\\Delta t}-1)\\,s_\\theta(v_t,t)
                + \\sqrt{e^{2\\gamma\\,\\Delta t}-1}\\,z

    where :math:`z \\sim \\mathcal{N}(0,I)` and :math:`\\Delta t = t - s > 0`.

    Parameters
    ----------
    corruption : KineticLangevinSDE
        The kinetic Langevin SDE.
    score_fn : callable or None
        Score function ``(x, t, batch_idx) → score``.

    """

    def __init__(
        self,
        corruption: KineticLangevinSDE,
        score_fn=None,
    ) -> None:
        """Initialise the EM predictor."""
        super().__init__(corruption=corruption, score_fn=score_fn)
        if not isinstance(corruption, KineticLangevinSDE):
            msg = f"TDMEMPredictor requires KineticLangevinSDE, got {type(corruption)}"
            raise TypeError(msg)

    def update_given_score(
        self,
        *,
        x: Tensor,
        t: Tensor,
        dt: float,
        batch_idx: B = None,  # noqa: ARG002
        score: Tensor,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> SampleAndMean:
        """Euler–Maruyama reverse step on velocity.

        Args:
            x: Current velocity ``(N, 3)``.
            t: Current time (unused — step uses dt only).
            dt: Reverse time step (positive scalar, going backward).
            score: Predicted score ``(N, 3)``.

        Returns:
            ``(v_new, v_mean)`` where ``v_mean`` is the deterministic part
            (no noise).

        """
        gamma = self.corruption.gamma
        gdt = torch.tensor(gamma * dt, device=x.device, dtype=x.dtype)
        exp_gdt = torch.exp(gdt)
        expm1_gdt = _expm1(gdt)
        expm1_2gdt = _expm1(2.0 * gdt)

        v_mean = exp_gdt * x + 2.0 * expm1_gdt * score
        noise = torch.randn_like(x)
        v_new = v_mean + torch.sqrt(expm1_2gdt.abs()) * noise
        return v_new, v_mean

    @classmethod
    def is_compatible(cls, corruption) -> bool:
        return isinstance(corruption, KineticLangevinSDE)


class TDMDDIMPredictor(Predictor):
    r"""Deterministic (DDPM-style) predictor for TDM velocity.

    Uses the ratio :math:`\alpha_s/\alpha_t` to form the one-step optimal
    reverse predictor given the **full velocity score**:

    .. math::
        v_s = r\,v_t + (r\,\sigma_t - \sigma_s)\,\sigma_t\,s_\theta

    where :math:`r = \alpha_s/\alpha_t`, :math:`s = t - \Delta t`, and
    :math:`s_\theta` is the **full** reconstructed velocity score (not the
    raw simplified network output).

    This matches the DDPM predictor in Appendix H of the KLDM paper.

    Parameters
    ----------
    corruption : KineticLangevinSDE
        The kinetic Langevin SDE.
    score_fn : callable or None
        Score function ``(x, t, batch_idx) -> score``.

    """

    def __init__(
        self,
        corruption: KineticLangevinSDE,
        score_fn=None,
    ) -> None:
        """Initialise the DDPM predictor."""
        super().__init__(corruption=corruption, score_fn=score_fn)
        if not isinstance(corruption, KineticLangevinSDE):
            msg = f"TDMDDIMPredictor requires KineticLangevinSDE, got {type(corruption)}"
            raise TypeError(msg)

    def update_given_score(
        self,
        *,
        x: Tensor,
        t: Tensor,
        dt: float,
        batch_idx: B = None,
        score: Tensor,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> SampleAndMean:
        """DDPM deterministic reverse step on velocity.

        Expects ``score`` to be the **full** reconstructed velocity score
        :math:`s^\\text{full} = \\nabla_{v_t} \\log p(v_t)`.

        Args:
            x: Current velocity ``(N, 3)``.
            t: Current time ``(B, 1)``.
            dt: Reverse time step (positive scalar, going backward).
            batch_idx: Atom -> graph mapping.
            score: Full velocity score ``(N, 3)``.

        Returns:
            ``(v_new, v_new)`` -- deterministic so sample == mean.

        """
        gamma = self.corruption.gamma
        t_exp = maybe_expand(t, batch_idx, x)
        s = (t_exp - dt).clamp(min=0.0)

        alpha_t = torch.exp(-gamma * t_exp)
        alpha_s = torch.exp(-gamma * s)
        sigma_t = torch.sqrt((1.0 - alpha_t**2).clamp(min=1e-12))
        sigma_s = torch.sqrt((1.0 - alpha_s**2).clamp(min=1e-12))

        # r = alpha_s / alpha_t
        # v_s = r * v_t + (r * sigma_t - sigma_s) * sigma_t * score
        r = alpha_s / alpha_t.clamp(min=1e-8)
        v_new = r * x + (r * sigma_t - sigma_s) * sigma_t * score
        return v_new, v_new

    @classmethod
    def is_compatible(cls, corruption) -> bool:
        return isinstance(corruption, KineticLangevinSDE)
