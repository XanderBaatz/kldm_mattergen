import torch
from mattergen.diffusion.corruption.corruption import B, BatchedData, maybe_expand
from mattergen.diffusion.corruption.sde_lib import SDE
from torch import Tensor
from torch_scatter import scatter_mean


def _scatter_center(
    x: Tensor,
    batch_idx: B,
) -> Tensor:
    """Subtract per-crystal mean to enforce zero center of gravity."""
    return x - scatter_mean(src=x, index=batch_idx, dim=0)[batch_idx]


# TODO(xba): maybe split up into ODE and SDE parts  # noqa: FIX002, TD003
class KineticLangevinSDE(SDE):
    """Kinetic Langevin diffusion on the 3D torus.

    The forward SDE for velocity and (unwrapped) displacement reads:

        dr = v dt
        dv = -γ v dt + sqrt(2γ) dW_v

    with wrapped positions x_t = wrap(x_0 + r_t).

    At time t, marginal distributions are:

    * ``v_t | v_0 ~ N(exp(-γt) v_0, (1 - exp(-2γt)) I)``
    * ``r_t | v_0 ~ N(mu_r(v_0, t), sigma_r(t)² I)``   (Corollary 1 and 2)

    The model predicts the score of the wrapped-normal for r,
    which is equivalent to the joint score w.r.t. (v, x).
    A simplified-parameterization target is returned by training_target.
    """  # noqa: RUF002

    def __init__(
        self,
        scale_pos: float = 1.0,  # used in wrapping and in zero CoG
        tf: float = 2.0,
        gamma: float = 1.0,
    ) -> None:
        """Initialize the Kinetic Langevin SDE."""
        super().__init__()

        self.scale_pos = scale_pos
        self.tf = tf
        self.gamma = gamma

        if gamma <= 0.0:  # else we have imaginary diffusion coefficients
            msg = "gamma must be positive"
            raise ValueError(msg)

    @property
    def T(self) -> float:  # noqa: N802
        """The end time of the diffusion process."""
        return 1.0

    def _t_internal(self, t: Tensor) -> Tensor:
        """Map external scheduler time t ∈ [0, 1] → internal time τ ∈ [0, tf]."""
        return t * self.tf

    @staticmethod
    def wrap_pos(
        x: Tensor,
        period: float = 1.0,
    ) -> Tensor:
        """Wrap group element (position) into [0, period).

        Implements the torus group action (expm): R -> [0, period) ≅ T.
        """
        return torch.remainder(x, period)

    @staticmethod
    def wrap_disp(
        x: Tensor,
        period: float = 1.0,
    ) -> Tensor:
        """Wrap Lie algebra element (displacement) into [-period/2, period/2).

        Implements the principal branch of logm: R -> [-period/2, period/2) ≅ g.

        Among all pre-images of expm(x), selects the one closest to zero.
        """
        return torch.remainder(x + period / 2, period) - period / 2

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
        drift = -self.gamma * self.tf * x
        diffusion = torch.full_like(input=x, fill_value=(2.0 * self.gamma * self.tf) ** 0.5)
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
        t = self._t_internal(t)  # τ = t * tf
        t = maybe_expand(x=t, batch=batch_idx, like=v0)  # expand t

        mu_v_t = torch.exp(-self.gamma * t) * v0
        # -expm1(-2γt) = 1 - e^{-2γt}: avoids cancellation for small t
        sigma_v_t = torch.sqrt((-torch.expm1(-2.0 * self.gamma * t)).clamp(min=1e-12))

        return mu_v_t, sigma_v_t

    def displacement_marginal(
        self,
        v0: Tensor,
        t: Tensor,
        vt: Tensor,
        batch_idx: B = None,
    ) -> tuple[Tensor, Tensor]:
        """Marginal mean and std for coordinate displacement r at time t (see Corollary 1 and 2 in TDM paper).

            r_t | v_0, v_t ~ N(mu_r_t, sigma_r_t² I)

        Uses the numerically stable tanh form for sigma_r_t to avoid cancellation at small t in float32.
        """
        gamma = self.gamma
        t = self._t_internal(t)  # τ = t * tf
        t = maybe_expand(x=t, batch=batch_idx, like=v0)  # expand t

        # (1 - e^{-γt}) / (γ(1 + e^{-γt})) = tanh(γt/2) / γ
        mu_r_t = (torch.tanh(gamma * t / 2.0) / gamma) * (vt + v0)
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

        Noise is centered per crystal (zero CoG) before scaling, which matches the periodic
        translation-invariance constraint of the model.

        Returns:
            v_t

        """
        mean, std = self.marginal_prob(x, t, batch_idx, batch)
        z = torch.randn_like(x)

        if batch_idx is not None:
            z = _scatter_center(x=z, batch_idx=batch_idx)  # zero-CoG per crystal

        return mean + std * z

    def sample_pos(
        self,
        x0: Tensor,  # initial position on torus
        v0: Tensor,  # initial velocity
        vt: Tensor,  # velocity at time t
        t: Tensor,
        batch_idx: B = None,
    ) -> Tensor:
        """Sample noisy torus position x_t given x_0.

        Displacement noise is centered per crystal (zero CoG) before wrapping, which matches the periodic
        translation-invariance constraint.

        Steps:
            1. Draw zero-CoG Gaussian noise z
            2. Form r_t = wrap_disp(mu_r_t + sigma_r_t * z), Lie algebra element
            3. Return x_t = wrap_pos(x0 + r_t), group element

        Returns:
            x_t

        """
        mu_r_t, sigma_r_t = self.displacement_marginal(v0=v0, t=t, vt=vt, batch_idx=batch_idx)
        z = torch.randn_like(x0)

        if batch_idx is not None:
            z = _scatter_center(x=z, batch_idx=batch_idx)

        r_t = self.wrap_disp(mu_r_t + sigma_r_t * z, self.scale_pos)  # r_t ∈ g, displacement

        return self.wrap_pos(x0 + r_t, period=self.scale_pos)  # x_t ∈ G, position

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
        # d = v.shape[-1]  # noqa: ERA001
        # return -0.5 * (v**2).sum(dim=-1) - 0.5 * d * math.log(2 * math.pi)  # noqa: ERA001
        d = z.shape[-1]
        logp = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        return logp - 0.5 * (z**2).sum(dim=-1)
