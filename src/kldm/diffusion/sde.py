import math

import torch
from mattergen.diffusion.corruption.corruption import B, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE, BaseVPSDE
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch_scatter import scatter_add


# Copied from MatterGen implementation of Song et al. (2021)
class VESDE(SDE):
    """Variance Exploding SDE. See equation (9) of Song et al.

    The drift is zero and the diffusion is designed such that the variance of the marginal distribution at time t is:
        sigma_min^2 * (sigma_max / sigma_min)^(2t)
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0) -> None:
        """Variance exploding SDE with diffusion coefficient changing exponentially over time."""
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @property
    def T(self) -> float:  # noqa: N802
        """The end time of the corruption process is 1.0, corresponding to the maximum noise level."""
        return 1.0

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance exploding SDE."""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = maybe_expand(
            sigma * torch.sqrt(torch.tensor(2 * (math.log(self.sigma_max) - math.log(self.sigma_min)), device=t.device)),
            batch_idx,
            x,
        )
        return drift, diffusion

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and standard deviation of the marginal distribution at time t."""
        mean = x
        std = maybe_expand(
            self.sigma_min * (self.sigma_max / self.sigma_min) ** t,
            batch_idx,
            x,
        )
        return mean, std

    def prior_sampling(
        self,
        shape: torch.Size | tuple[int, ...],
        conditioning_data: BatchedData | None = None,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Generate samples from the prior distribution."""
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(
        self,
        z: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        """Compute the log probability of the given noise under the prior distribution."""
        shape = z.shape
        N = torch.prod(shape[1:])  # noqa: N806
        if batch_idx is not None:
            return -N / 2.0 * math.log(2 * torch.pi * self.sigma_max**2) - scatter_add(torch.sum(z**2, dim=1), batch_idx) / (2 * self.sigma_max**2)
        return -N / 2.0 * math.log(2 * torch.pi * self.sigma_max**2) - torch.sum(z**2, dim=tuple(range(1, z.ndim))) / (2 * self.sigma_max**2)


# Copied from MatterGen implementation of Song et al. (2021)
class VPSDE(BaseVPSDE):
    """Variance Preserving SDE. See equation (11) of Song et al.

    The drift and diffusion are designed such that the variance of the marginal distribution at time t is:
        1 - alpha_t^2
    where alpha_t is the mean coefficient.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20) -> None:
        """Variance-preserving SDE with drift coefficient changing linearly over time."""
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta scheduler."""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean_coeff)


class SubVPSDE(BaseVPSDE):
    """Sub Variance Preserving SDE. See equation (12) of Song et al.

    The drift is identical to VP, but the diffusion is scaled to keep the variance strictly lower than the VP SDE.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20) -> None:
        """Variance-preserving SDE with drift coefficient changing linearly over time."""
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear beta scheduler."""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean_coeff)

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance preserving SDE."""
        beta_t = self.beta(t)
        drift = -0.5 * maybe_expand(beta_t, batch_idx, x) * x
        mean_coeff = self._marginal_mean_coeff(t)  # alpha_t
        discount = 1.0 - torch.pow(mean_coeff, 4)  # ensure variance is strictly less than 1, so that this is a "sub" VPSDE
        diffusion = maybe_expand(torch.sqrt(beta_t * discount), batch_idx, x)
        return drift, diffusion

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and standard deviation of the marginal distribution at time t."""
        mean_coeff = self._marginal_mean_coeff(t)  # alpha_t
        mean = maybe_expand(mean_coeff, batch_idx, x) * x
        std = maybe_expand(
            x=1.0 - torch.pow(mean_coeff, 2),  # ensure variance is strictly less than 1, so that this is a "sub" VPSDE
            batch=batch_idx,
            like=x,
        )
        return mean, std
