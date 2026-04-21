"""Lattice corruption - re-exports MatterGen's LatticeVPSDE for cell 3x3."""

from typing import TYPE_CHECKING

import torch

from kldm_new.diffusion.sde import SubVPSDE
from mattergen.common.diffusion.corruption import (
    LatticeVPSDE,
    expand,
    make_noise_symmetric_preserve_variance,
)
from mattergen.diffusion.corruption.corruption import B, maybe_expand

if TYPE_CHECKING:
    from mattergen.diffusion.data.batched_data import BatchedData

__all__ = ["LatticeVPSDE", "make_noise_symmetric_preserve_variance"]


class LatticeSubVPSDE(SubVPSDE):
    @staticmethod
    def from_subvpsde_config(subvpsde_config: dict) -> "LatticeSubVPSDE":  # noqa: UP037
        """Construct a LatticeSubVPSDE from a SubVPSDE config."""
        return LatticeSubVPSDE(
            **subvpsde_config,
        )

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        limit_density: float = 0.05,
        limit_var_scaling_constant: float = 0.25,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        super().__init__()
        self.beta_0 = beta_min
        self.beta_1 = beta_max

        self.limit_density = limit_density
        self.limit_var_scaling_constant = limit_var_scaling_constant

        self._limit_info_key = "num_atoms"

    @property
    def limit_info_key(self) -> str:
        return self._limit_info_key

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_0 + t * (self.beta_1 - self.beta_0)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:  # alpha
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        return torch.exp(log_mean_coeff)

    def marginal_prob(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None  # noqa: S101

        mean_coeff = self._marginal_mean_coeff(t)

        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)

        mean_coeff_expanded = maybe_expand(mean_coeff, batch_idx, x)

        mean = mean_coeff_expanded * x + (1 - mean_coeff_expanded) * limit_mean
        std = torch.sqrt((1.0 - mean_coeff_expanded**2) * limit_var)
        return mean, std

    def mean_coeff_and_std(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean coefficient and standard deviation of marginal distribution at time t."""
        mean_coeff = self._marginal_mean_coeff(t)
        std = self.marginal_prob(x, t, batch_idx, batch)[1]
        return maybe_expand(mean_coeff, batch=None, like=x), std

    def get_limit_mean(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        n_atoms = batch[self.limit_info_key]

        return torch.pow(
            torch.eye(3, device=x.device).expand(len(n_atoms), 3, 3) * n_atoms[:, None, None] / self.limit_density,
            1.0 / 3,
        ).to(x.device)

    def get_limit_var(self, x: torch.Tensor, batch: BatchedData) -> torch.Tensor:
        n_atoms = batch[self.limit_info_key]

        n_atoms_expanded = expand(n_atoms, x.shape)

        n_atoms_expanded = torch.tile(n_atoms_expanded, (1, 3, 3))

        return torch.pow(n_atoms_expanded, 2.0 / 3).to(x.device) * self.limit_var_scaling_constant

    def sample_marginal(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,
    ) -> torch.Tensor:
        mean, std = self.marginal_prob(x=x, t=t, batch=batch)
        z = torch.randn_like(x)
        z = make_noise_symmetric_preserve_variance(z)
        return mean + expand(std, z.shape) * z

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,
        batch_idx: B = None,  # noqa: ARG002
    ) -> torch.Tensor:
        x_sample = torch.randn(*shape)
        x_sample = make_noise_symmetric_preserve_variance(x_sample)

        assert conditioning_data is not None  # noqa: S101

        limit_info = conditioning_data[self.limit_info_key]
        x_sample = x_sample.to(limit_info.device)
        limit_mean = self.get_limit_mean(x=x_sample, batch=conditioning_data)
        limit_var = self.get_limit_var(x=x_sample, batch=conditioning_data)

        return x_sample * limit_var.sqrt() + limit_mean

    def sde(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch is not None  # noqa: S101

        # same mean as VPSDE
        limit_mean = self.get_limit_mean(x=x, batch=batch)
        limit_var = self.get_limit_var(x=x, batch=batch)

        beta_t = self.beta(t)
        drift = -0.5 * expand(beta_t, x.shape) * (x - limit_mean)

        mean_coeff = self._marginal_mean_coeff(t)
        discount = 1.0 - torch.pow(mean_coeff, 4)
        diffusion = torch.sqrt(expand(beta_t, limit_var.shape) * limit_var * discount)
        # or diffusion = torch.sqrt(expand(beta_t * discount, limit_var.shape) * limit_var)

        return maybe_expand(drift, batch_idx), maybe_expand(diffusion, batch_idx)
