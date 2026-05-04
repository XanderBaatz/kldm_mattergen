import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE
from torch import Tensor


class KineticLangevinSDE(SDE):
    """Kinetic Langevin diffusion on the 3D torus.

    The forward SDE for velocity and (unwrapped) displacement reads:

        dr = v dt
        dv = -γ v dt + sqrt(2γ) dW_v

    with wrapped positions pos_t = wrap(pos_0 + r_t).

    At time t, marginal distributions are:

    * ``v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)``
    * ``r_t | v_0 ~ N(mu_r(v_0, t), sigma_r(t)² I)``   (see code for mu_r, sigma_r)
    * ``pos_t = wrap(pos_0 + wrap(r_t))``  where ``wrap`` is mod *scale_pos*.

    The model predicts the score of the wrapped-normal for r,
    which is equivalent to the joint score w.r.t. (v, pos).
    A simplified-parameterization target is returned by training_target.
    """  # noqa: RUF002

    def __init__(
        self,
        scale_pos: float = 1.0,
        tf: float = 2.0,
        gamma: float = 1.0,
        k_wn_score: int = 13,
        n_sigmas: int = 2000,
    ) -> None:
        """Initialize the Kinetic Langevin SDE."""
        super().__init__()

        self.scale_pos = scale_pos
        self.tf = tf
        self.gamma = gamma
        self.k_wn_score = k_wn_score
        self.n_sigmas = n_sigmas

        if gamma <= 0.0:  # else we have imaginary diffusion coefficients
            msg = "gamma must be positive"
            raise ValueError(msg)

    @staticmethod
    def _wrap_pos(
        x: Tensor,
        period: float = 1.0,
    ) -> Tensor:
        """Wrap group element (position) into [0, period). Implements expm on torus."""
        return torch.remainder(x, period)

    @staticmethod
    def _wrap_disp(
        x: Tensor,
        period: float = 1.0,
    ) -> Tensor:
        """Wrap Lie algebra element (displacement) into [-period/2, period/2). Implements principal logm."""
        return torch.remainder(x + period / 2, period) - period / 2

    @property
    def T(self) -> float:  # noqa: N802
        """The end time of the diffusion process."""
        return self.tf

    def sde(
        self,
        x: Tensor,
        t: Tensor,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """OU SDE process for velocity.

            dv = -γ v dt + sqrt(2γ) dW_v

        Returns drift and diffusion for velocity.
        """  # noqa: RUF002
        drift = -self.gamma * x
        diffusion = torch.full_like(input=x, fill_value=(2.0 * self.gamma) ** 0.5)
        return drift, diffusion

    def marginal_prob(
        self,
        x: Tensor,  # v0
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std for velocity at time t (see Corollary 1 µ_xi and sigma²_xi in TDM paper).

            v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)

        Returns mean and std.
        """  # noqa: RUF002
        v0 = x
        t = maybe_expand(x=t, batch=batch_idx, like=v0)  # expand t

        mu_v_t = torch.exp(-self.gamma * t) * v0
        sigma_v_t = torch.sqrt((1.0 - torch.exp(-2.0 * self.gamma * t)).clamp(min=1e-12))

        return mu_v_t, sigma_v_t

    def _displacement_marginal(
        self,
        v0: Tensor,
        t: Tensor,
        vt: Tensor | None = None,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std for coordinate displacement r at time t (see Corollary 1 and 2 in TDM paper)."""
        gamma = self.gamma
        t = maybe_expand(x=t, batch=batch_idx, like=v0)  # expand t
        exp_vt = torch.exp(-gamma * t)

        mu_r_t = ((1.0 - exp_vt) / (gamma * (1.0 + exp_vt))) * (vt + v0)
        # sigma_r_t = torch.sqrt(torch.clamp((2.0 / gamma**2) * (gamma * t + (4 * gamma) / (exp_vt ** (-1.0) + 1.0) - 2 * gamma), min=1e-12))  # noqa: E501, ERA001
        sigma_r_t = torch.sqrt(torch.clamp((2.0 / gamma**2.0) * (gamma * t - 2.0 * torch.tanh((gamma * t) / 2.0)), min=1e-12))

        return mu_r_t, sigma_r_t

    def sample_marginal(
        self,
        x: Tensor,
        t: Tensor,
        batch_idx: B = None,
        batch: BatchedData | None = None,
    ) -> Tensor:
        """Sample noisy marginal for v_t given v_0.

        Returns:
            v_t

        """
        mean, std = self.marginal_prob(x, t, batch_idx, batch)
        z = torch.randn_like(x)

        return mean + std * z

    def sample_pos(
        self,
        x0: Tensor,  # pos_0
        v0: Tensor,
        vt: Tensor,
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample noisy torus position x_t given x_0."""
        mu_r_t, sigma_r_t = self._displacement_marginal(v0=v0, t=t, vt=vt, batch_idx=batch_idx)
        z = torch.randn_like(x0)

        r_t = self._wrap_disp(mu_r_t + sigma_r_t * z, self.scale_pos)  # r_t, displacement

        return self._wrap_pos(x0 + r_t, period=self.scale_pos)  # x_t, position

    def prior_sampling(
        self,
        shape: torch.Size | tuple,
        conditioning_data: BatchedData | None = None,  # noqa: ARG002
        batch_idx: B = None,  # noqa: ARG002
    ) -> Tensor:
        """Sample velocity from the prior distribution p(x_T) = N(0, I)."""
        return torch.randn(*shape)

    def prior_logp(
        self,
        z: Tensor,
        batch_idx: B = None,  # noqa: ARG002
        batch: BatchedData | None = None,  # noqa: ARG002
    ) -> Tensor:
        """Log-probability under the velocity prior log(p(x_T))."""
        # d = v.shape[-1]
        # return -0.5 * (v**2).sum(dim=-1) - 0.5 * d * math.log(2 * math.pi)
        d = z.shape[-1]
        logp = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        return logp - 0.5 * (z**2).sum(dim=-1)
