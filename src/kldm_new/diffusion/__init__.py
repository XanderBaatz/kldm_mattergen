"""Distributions and score utilities for wrapped-normal diffusion on the torus."""

from __future__ import annotations

import torch
from torch import Tensor


def d_log_p_wrapped_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    N: int = 10,  # noqa: N803
    T: float = 1.0,  # noqa: N803
) -> Tensor:
    r"""Derivative of log WN w.r.t. the **mean** (:math:`\partial_\mu \log p`).

    This returns the quantity needed for the KLDM velocity score.  By the
    chain rule through :math:`\mu_r(v_t)`:

    .. math::
        \nabla_{v_t} \log p(r_t \mid v_t, v_0)
          = \underbrace{\frac{\partial \mu_r}{\partial v_t}}_{\text{prefactor}}
            \cdot \underbrace{\frac{\partial \log \mathrm{WN}}{\partial \mu_r}}_{\text{this function}}

    Because :math:`\partial_\mu \log \mathrm{WN} = -\partial_r \log \mathrm{WN}`,
    this is the **negative** of the standard position-space score.

    Concretely:

    .. math::
        \frac{\partial \log p(x \mid \mu, \sigma^2)}{\partial \mu}
          = \sum_n w_n \frac{x - \mu + nT}{\sigma^2}, \quad
          w_n = \operatorname{softmax}_n\!\left(-\frac{(x-\mu+nT)^2}{2\sigma^2}\right)

    Args:
        x: Sample positions, arbitrary shape.
        mu: Mean of the (unwrapped) normal, same shape as *x*.
        sigma: Standard deviation, same shape as *x*.
        N: Number of periodic images on each side.
        T: Period of the torus (default 1.0 → fractional coords).

    Returns:
        :math:`\partial_\mu \log p` tensor, same shape as *x*.

    """
    var = sigma**2
    ns = torch.arange(-N, N + 1, device=x.device, dtype=x.dtype)
    for _ in range(x.ndim):
        ns = ns.unsqueeze(-1)

    # shifted[n] = x - mu - n*T.  Relabelling n → -n shows that summing
    # w_n * shifted_n / σ² over n ∈ [-N,N] is identical to summing
    # w_i * (x-mu+iT) / σ² over i ∈ [-N,N], i.e. ∂_μ log WN.
    shifted = x.unsqueeze(0) - mu.unsqueeze(0) - ns * T  # (2N+1, *x.shape)
    log_ps = -0.5 * shifted**2 / var.unsqueeze(0)

    weights = torch.softmax(log_ps, dim=0)
    # Clamp var to avoid 0/0 when sigma→0 and shifted→0 (limit is 0).
    grad_per_image = shifted / var.unsqueeze(0).clamp(min=1e-12)
    return (weights * grad_per_image).sum(dim=0)


def sigma_norm(
    sigma: Tensor,
    T: float = 1.0,
    N: int = 10,
    sn: int = 2_000,
    chunk_size: int = 100,
) -> Tensor:
    r"""Expected squared L2-norm of the wrapped-normal score (per dimension).

    .. math::
        \sigma_{\text{norm}}(\sigma) = \mathbb{E}_{x \sim WN(0,\sigma,T)}
        \bigl[\|\nabla_x \log p(x)\|^2\bigr]

    Estimated via Monte-Carlo with *sn* samples.

    The grid of sigma values is processed in chunks of *chunk_size* to bound
    peak memory.  Inside ``d_log_p_wrapped_normal`` an intermediate tensor of
    shape ``(2N+1, sn, chunk_size)`` is allocated; with the default values
    this stays well under 1 GB per chunk.

    Args:
        sigma: Scalar or 1-D tensor of standard deviations.
        T: Period.
        N: Number of periodic images.
        sn: Number of Monte-Carlo samples.
        chunk_size: Number of sigma values processed per chunk.

    Returns:
        Tensor of the same shape as *sigma*.

    """  # noqa: D401
    original_shape = sigma.shape
    sigma_flat = sigma.reshape(-1)  # (K,)
    K = sigma_flat.shape[0]  # noqa: N806

    sn_values = torch.empty(K, dtype=sigma.dtype, device=sigma.device)

    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        s_chunk = sigma_flat[start:end]  # (C,)

        eps = torch.randn(sn, end - start, device=sigma.device, dtype=sigma.dtype)
        x = torch.remainder(s_chunk.unsqueeze(0) * eps, T)  # (sn, C)
        mu = torch.zeros_like(x)
        s_exp = s_chunk.unsqueeze(0).expand(sn, -1)

        scores = d_log_p_wrapped_normal(x, mu, s_exp, N=N, T=T)  # (sn, C)
        sn_values[start:end] = (scores**2).mean(dim=0)

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
        return z - means[batch_idx]
