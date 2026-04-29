import torch  # noqa: I001, RUF100
from mattergen.diffusion.corruption.corruption import B, maybe_expand
from mattergen.diffusion.corruption.sde_lib import VESDE, VPSDE, BaseVPSDE
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: RUF100, TC001, TC002

from kldm_new.diffusion.schedule import LinearSchedule, NoiseSchedule

__all__ = ["VESDE", "VPSDE", "SubVPSDE"]


class SubVPSDE(BaseVPSDE):
    """Sub Variance Preserving SDE. See equation (12) of Song et al.

    The drift is identical to VP, but the diffusion is scaled to keep the variance strictly lower than the VP SDE.

    Parameters
    ----------
    schedule:
        Noise schedule that provides ``beta(t)`` and ``mean_coeff(t)``.
        Defaults to a :class:`~kldm_new.diffusion.schedule.LinearSchedule`.
    beta_min, beta_max:
        Used only when ``schedule=None`` to construct the default
        :class:`~kldm_new.diffusion.schedule.LinearSchedule`.

    """

    def __init__(
        self,
        schedule: NoiseSchedule | None = None,
        beta_min: float = 0.1,
        beta_max: float = 20,
    ) -> None:
        """Sub variance-preserving SDE with a pluggable noise schedule."""
        super().__init__()
        if schedule is None:
            schedule = LinearSchedule(beta_min=beta_min, beta_max=beta_max)
        self.schedule = schedule

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate β(t) delegated to the schedule."""
        return self.schedule.beta(t)

    def _marginal_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:  # alpha
        return self.schedule.mean_coeff(t)

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
