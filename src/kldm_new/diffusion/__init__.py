"""Distributions and score utilities for wrapped-normal diffusion on the torus."""

from __future__ import annotations

import torch
from torch import Tensor


def d_log_p_wrapped_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    N: int = 10,
    T: float = 1.0,
) -> Tensor:
    """Score (∇_x log p) of a wrapped normal distribution.

    Computes the derivative of the log-probability of a wrapped normal
    distribution over the interval [0, T) by summing periodic images
    from -N to N.

    Args:
        x: Sample positions, arbitrary shape.
        mu: Mean of the (unwrapped) normal, same shape as *x*.
        sigma: Standard deviation, same shape as *x*.
        N: Number of periodic images on each side.
        T: Period of the torus (default 1.0 → fractional coords).

    Returns:
        Score tensor with the same shape as *x*.

    """
    # sigma² and inverse
    var = sigma**2
    # Broadcast over image index: shape = (2N+1, *x.shape)
    ns = torch.arange(-N, N + 1, device=x.device, dtype=x.dtype)
    for _ in range(x.ndim):
        ns = ns.unsqueeze(-1)

    # Shifted x for each periodic image
    shifted = x.unsqueeze(0) - mu.unsqueeze(0) - ns * T  # (2N+1, *x.shape)
    log_ps = -0.5 * shifted**2 / var.unsqueeze(0)  # un-normalised log-prob

    # Log-sum-exp for numerical stability
    log_norm = torch.logsumexp(log_ps, dim=0)  # (*x.shape)

    # ∂/∂x log Σ_n exp(log_p_n) = Σ_n [exp(log_p_n) * (∂ log_p_n / ∂x)] / Σ_n exp(log_p_n)
    #   = Σ_n softmax_n * (-(x - mu - nT) / sigma²)
    weights = torch.softmax(log_ps, dim=0)  # (2N+1, *x.shape)
    grad_per_image = -shifted / var.unsqueeze(0)
    score = (weights * grad_per_image).sum(dim=0)
    return score


def sigma_norm(
    sigma: Tensor,
    T: float = 1.0,
    N: int = 10,
    sn: int = 20_000,
) -> Tensor:
    r"""Expected squared L2-norm of the wrapped-normal score (per dimension).

    .. math::
        \sigma_{\text{norm}}(\sigma) = \mathbb{E}_{x \sim WN(0,\sigma,T)}
        \bigl[\|\nabla_x \log p(x)\|^2\bigr]

    Estimated via Monte-Carlo with *sn* samples.

    Args:
        sigma: Scalar or 1-D tensor of standard deviations.
        T: Period.
        N: Number of periodic images.
        sn: Number of Monte-Carlo samples.

    Returns:
        Tensor of the same shape as *sigma*.

    """
    original_shape = sigma.shape
    sigma_flat = sigma.reshape(-1)  # (K,)

    # Sample from wrapped normal: x = wrap(mu + sigma * eps), mu=0
    eps = torch.randn(sn, sigma_flat.shape[0], device=sigma.device)
    x = torch.remainder(sigma_flat.unsqueeze(0) * eps, T)  # (sn, K)

    mu = torch.zeros_like(sigma_flat).unsqueeze(0).expand(sn, -1)
    sigma_expanded = sigma_flat.unsqueeze(0).expand(sn, -1)

    scores = d_log_p_wrapped_normal(x, mu, sigma_expanded, N=N, T=T)
    # E[||score||^2] ≈ mean over samples
    sn_values = (scores**2).mean(dim=0)  # (K,)
    return sn_values.reshape(original_shape)


class DistributionGaussian:
    """Zero-centre-of-gravity Gaussian sampler for atom positions."""

    def __init__(self, dim: int = 3):
        self.dim = dim

    def sample(
        self,
        n_atoms: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Sample zero-CoG Gaussian noise for each crystal's atoms.

        Args:
            n_atoms: (B,) number of atoms per crystal.
            device: Target device.

        Returns:
            (total_atoms, dim) tensor with zero centre of gravity per crystal.

        """
        total = int(n_atoms.sum().item())
        z = torch.randn(total, self.dim, device=device)
        # Subtract per-crystal mean to enforce zero CoG
        batch_idx = torch.repeat_interleave(torch.arange(len(n_atoms), device=device), n_atoms)
        from torch_scatter import scatter_mean

        means = scatter_mean(z, batch_idx, dim=0)
        z = z - means[batch_idx]
        return z
