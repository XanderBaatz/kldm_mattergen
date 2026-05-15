import torch
from mattergen.diffusion.corruption.corruption import B  # noqa: TC002
from torch import Tensor
from torch_scatter import scatter_mean

# ---------------------------------------------------------------------------
# Wrapped-normal math helpers (period T, default 1.0)
# ---------------------------------------------------------------------------


def p_wrapped_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    N: int = 10,  # noqa: N803
    T: float = 1.0,  # noqa: N803
) -> Tensor:
    """Unnormalised density of the wrapped normal (sum of Gaussians over images)."""
    total = torch.zeros_like(x)
    for k in range(-N, N + 1):
        total = total + torch.exp(-((x - mu + T * k) ** 2) / (2.0 * sigma**2))
    return total


def d_log_p_wrapped_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    N: int = 10,  # noqa: N803
    T: float = 1.0,  # noqa: N803
) -> Tensor:
    """Gradient of log p_WN w.r.t. x (positive-direction convention as in kldm_frnct)."""
    numerator = torch.zeros_like(x)
    for k in range(-N, N + 1):
        numerator = numerator + (x - mu + T * k) / sigma**2 * torch.exp(-((x - mu + T * k) ** 2) / (2.0 * sigma**2))
    return numerator / p_wrapped_normal(x, mu, sigma, N, T)


@torch.no_grad()
def sigma_norm(
    sigma: Tensor,
    T: float = 1.0,  # noqa: N803
    N: int = 10,  # noqa: N803
    sn: int = 20_000,
) -> Tensor:
    """E[||d_log_p_WN(x, 0, sigma)||^2] over x ~ WN(0, sigma), for each sigma.

    Returns a 1-D tensor with the same length as *sigma*.
    """
    sigmas_2d = sigma[None].expand(sn, -1)  # [sn, n_sigmas]
    x_sample = sigma * torch.randn_like(sigmas_2d)
    # wrap into principal branch [-T/2, T/2)
    x_sample = torch.remainder(x_sample + T / 2.0, T) - T / 2.0
    score = d_log_p_wrapped_normal(x_sample, torch.zeros_like(x_sample), sigma, N=N, T=T)
    return (score**2).mean(dim=0)


def _scatter_center(
    x: Tensor,
    batch_idx: B,
) -> Tensor:
    """Subtract per-crystal mean to enforce zero center of gravity."""
    return x - scatter_mean(src=x, index=batch_idx, dim=0)[batch_idx]
