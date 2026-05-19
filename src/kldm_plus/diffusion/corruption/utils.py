import torch
from torch import Tensor

from kldm_plus.nn.utils import scatter_center

# Canonical implementation lives in kldm_plus.nn.utils; re-exported here so
# diffusion code can import from a single location without depending on nn directly.
__all__ = ["scatter_center"]

# ---------------------------------------------------------------------------
# Wrapped-normal math helpers (period T, default 1.0)
# ---------------------------------------------------------------------------


def p_wrapped_normal(
    x: Tensor,  # r_t
    mu: Tensor,  # mu_r_t
    sigma: Tensor,  # sigma_r_t
    N: int = 10,  # noqa: N803
    T: float = 1.0,  # noqa: N803
) -> Tensor:
    """Unnormalised density of the wrapped normal (sum of Gaussians over images).

    The wrapped normal with period T is defined as a sum of 2K+1 Gaussian images:

        WN_K(r_t ; mu_r_t, sigma_r_t^2, T)
            = sum_{k=-K}^{K}  N(r_t + k*T ; mu_r_t, sigma_r_t^2)
            = sum_{k=-K}^{K}  exp( -(r_t + k*T - mu_r_t)^2
                                    / (2 * sigma_r_t^2) )

    where the normalizing constant (1 / sqrt(2 pi sigma_r_t^2)) cancels in the log-derivative and is omitted.
    """
    k = torch.arange(
        -N,
        N + 1,
        device=x.device,
        dtype=x.dtype,
    ).view(-1, *([1] * x.ndim))  # broadcast
    delta = x - mu + T * k  # [2N+1, ...]
    return torch.exp(-(delta**2) / (2.0 * sigma**2)).sum(dim=0)


def d_log_p_wrapped_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
    N: int = 10,  # noqa: N803
    T: float = 1.0,  # noqa: N803
) -> Tensor:
    """Gradient of log p_WN w.r.t. x (positive-direction convention as in kldm_frnct)."""
    k = torch.arange(
        -N,
        N + 1,
        device=x.device,
        dtype=x.dtype,
    ).view(-1, *([1] * x.ndim))  # broadcast
    delta = x - mu + T * k  # [2N+1, ...]
    gauss = torch.exp(-(delta**2) / (2.0 * sigma**2))  # [2N+1, ...]
    return (delta / sigma**2 * gauss).sum(dim=0) / gauss.sum(dim=0)


@torch.no_grad()
def sigma_norm(
    sigma: Tensor,
    T: float = 1.0,  # noqa: N803
    N: int = 10,  # noqa: N803
    sn: int = 20_000,
    chunk_size: int = 50,
) -> Tensor:
    """E[||d_log_p_WN(x, 0, sigma)||^2] over x ~ WN(0, sigma), for each sigma.

    Returns a 1-D tensor with the same length as *sigma*.

    Computation is chunked over the sigma dimension to bound peak memory.
    With the default chunk_size=50 and sn=20000, each chunk allocates
    [2N+1, sn, chunk_size] = [27, 20000, 50] ~ 108 MB.
    """
    result = torch.empty(len(sigma), device=sigma.device, dtype=sigma.dtype)
    for start in range(0, len(sigma), chunk_size):
        sigma_c = sigma[start : start + chunk_size]  # [C]
        x = sigma_c[None] * torch.randn(sn, len(sigma_c), device=sigma.device, dtype=sigma.dtype)
        x = torch.remainder(x + T / 2.0, T) - T / 2.0  # [sn, C]
        score = d_log_p_wrapped_normal(
            x=x,
            mu=torch.zeros_like(x),
            sigma=sigma_c[None],
            N=N,
            T=T,
        )  # [sn, C]
        result[start : start + chunk_size] = (score**2).mean(dim=0)
    return result


if __name__ == "__main__":
    x = torch.tensor([0.5, 1.5, 2.0])
    mu = torch.tensor([1.1, 8.2, 3.2])
    sigma = torch.tensor([0.3, 8.22, 9.9])
    N = 10
    T = 1.0
    print(p_wrapped_normal(x, mu, sigma, N, T))  # noqa: T201
    print(sigma_norm(sigma, T, N, sn=100))  # noqa: T201
