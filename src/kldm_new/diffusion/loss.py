import torch
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch import Tensor, nn
from torch_scatter import scatter_mean

from kldm_new.diffusion import d_log_p_wrapped_normal, sigma_norm
from kldm_new.diffusion.corruption import KLDMMultiCorruption  # noqa: TC001
from kldm_new.diffusion.tdm import KineticLangevinSDE  # noqa: TC001
from kldm_new.nn.utils import scatter_center


class KLDMLoss(nn.Module):
    r"""Combined loss for KLDM: TDM (pos/vel) + DSM (cell)."""

    def __init__(  # noqa: PLR0913
        self,
        weight_vel: float = 1.0,
        weight_cell: float = 1.0,
        *,
        simplified_parameterization: bool = True,
        k_wn_score: int = 13,
        n_sigmas: int = 2000,
        scale_pos: float = 1.0,
    ) -> None:
        """Initialize the KLDM loss."""
        super().__init__()
        self.weight_vel = weight_vel
        self.weight_cell = weight_cell
        self.simplified_parameterization = simplified_parameterization
        self.k_wn_score = k_wn_score
        self.scale_pos = scale_pos

        if simplified_parameterization:
            sigma_grid = torch.logspace(
                start=-6,
                end=0,
                steps=n_sigmas,  # steps
                dtype=torch.float64,
            )
            sn_grid = sigma_norm(
                sigma=sigma_grid,
                T=scale_pos,
                N=k_wn_score,
                sn=20000,
            ).float()
            self.register_buffer("_sn_log_sigma", sigma_grid.log().float())
            self.register_buffer("_sn_values", sn_grid)

    def _prefactor_t(
        self,
        gamma: float,
        t_exp: Tensor,  # expanded t
    ) -> Tensor:
        """See KLDM p. 24 and Eq. 26.

        This can be simplified to:

            1 / gamma * tanh(gamma * t / 2).
        """
        return torch.tanh(gamma * t_exp / 2.0) / gamma

    def _lookup_sigma_norm(
        self,
        sigma: Tensor,
    ) -> Tensor:
        """Interpolate sigma_norm from precomputed lookup table."""
        log_sigma = sigma.clamp(min=1e-7).log()
        idx = torch.searchsorted(self._sn_log_sigma, log_sigma).clamp(1, len(self._sn_log_sigma) - 1)
        lo, hi = idx - 1, idx
        log_s_lo = self._sn_log_sigma[lo]
        log_s_hi = self._sn_log_sigma[hi]
        w = ((log_sigma - log_s_lo) / (log_s_hi - log_s_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
        return self._sn_values[lo] * (1.0 - w) + self._sn_values[hi] * w

    def _pos_training_target(  # noqa: PLR0913
        self,
        pos_sde: KineticLangevinSDE,
        x0: Tensor,  # position/coordinate on manifold
        xt: Tensor,  # position/coordinate on manifold
        v0: Tensor,  # velocity in Lie algebra
        vt: Tensor,  # velocity in Lie algebra
        t: Tensor,
        batch_idx: torch.LongTensor | None,
        scale_pos: float,
    ) -> Tensor:
        """Compute the position training target for the score network.

        The target is the score of the wrapped-normal displacement:

            s_c = prefactor * score_wn

        where:

            prefactor = 1 / gamma * tanh(gamma * t / 2)
            score_wn = ∇_{mu_r_t} log(WN(r_t ; mu_r_t, sigma²_r_t I))
        """
        t_exp = maybe_expand(x=t, batch=batch_idx, like=x0)

        mu_r_t, sigma_r_t = pos_sde.displacement_marginal(v0=v0, t=t, vt=vt, batch_idx=batch_idx)

        # Displacement wrapped to [-scale/2, scale/2)
        r_t = pos_sde.wrap_disp(x=xt - x0, period=scale_pos)

        score_wn = d_log_p_wrapped_normal(
            x=r_t,
            mu=mu_r_t,
            sigma=sigma_r_t,
            N=self.k_wn_score,
            T=scale_pos,
        )

        if self.simplified_parameterization:
            # Simplified target: WN_score / sqrt(sigma_norm)
            sigma_flat = sigma_r_t.reshape(-1)
            sigma_unique, inv_idx = torch.unique(sigma_flat, return_inverse=True)
            sn_unique = self._lookup_sigma_norm(sigma=sigma_unique)
            sn = sn_unique[inv_idx].reshape(sigma_r_t.shape)
            target = score_wn / torch.sqrt(sn.clamp(min=1e-8))
        else:
            prefactor = self._prefactor_t(pos_sde.gamma, t_exp)
            target = prefactor * score_wn

        # Center the target per crystal — the network is constrained to output
        # zero CoG, so the target must also have zero CoG.  Failure to do this
        # creates an impossible optimization objective.  This correction is
        # noted in kldm_jonas: "it is indeed a mistake in the original KLDM
        # paper appendix" (centering should apply to the target, not f_t).
        if batch_idx is not None:
            target = scatter_center(target, index=batch_idx)

        return target

    def forward(  # noqa: PLR0913
        self,
        *,
        multi_corruption: KLDMMultiCorruption,
        batch: BatchedData,
        noisy_batch: BatchedData,
        score_model_output: BatchedData,
        t: Tensor,
        node_is_unmasked: torch.LongTensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the total loss.

        Args:
            multi_corruption: The KLDM multi-corruption instance.
            batch: **Clean** batch (before corruption).
            noisy_batch: **Noisy** batch (after corruption at time *t*).
            score_model_output: Model output with ``vel`` and ``cell`` fields.
            t: Diffusion time ``(B, 1)``.
            node_is_unmasked: Optional per-atom mask.

        Returns:
            ``(total_loss, loss_dict)``

        """
        pos_batch_idx = noisy_batch.get_batch_idx("pos")

        if pos_batch_idx is None:
            msg = "pos_batch_idx cannot be None"
            raise ValueError(msg)

        batch_size = noisy_batch.get_batch_size()
        pos_sde = multi_corruption.pos_sde
        scale_pos = pos_sde.scale_pos

        vel_target = self._pos_training_target(
            pos_sde=pos_sde,
            x0=batch["pos"],
            xt=noisy_batch["pos"],
            v0=batch["vel"],
            vt=noisy_batch["vel"],
            t=t,
            batch_idx=pos_batch_idx,
            scale_pos=scale_pos,
        )
        vel_pred = score_model_output["vel"]
        vel_loss_per_atom = ((vel_pred - vel_target) ** 2).sum(dim=-1)  # (N,)

        if node_is_unmasked is not None:
            vel_loss_per_atom = vel_loss_per_atom * node_is_unmasked

        vel_loss = scatter_mean(vel_loss_per_atom, pos_batch_idx, dim=0, dim_size=batch_size).mean()

        cell_0 = batch["cell"]
        cell_t = noisy_batch["cell"]
        cell_pred = score_model_output["cell"]

        mean, std = multi_corruption.cell_sde.marginal_prob(cell_0, t, batch_idx=None, batch=noisy_batch)
        noise = (cell_t - mean) / std.clamp(min=1e-8)
        cell_target = -noise

        cell_loss = ((cell_pred - cell_target) ** 2).mean()

        total_loss = self.weight_vel * vel_loss + self.weight_cell * cell_loss

        loss_dict = {
            "vel": vel_loss.item(),
            "cell": cell_loss.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def reconstruct_velocity_score(
        self,
        pred: Tensor,
        vt: Tensor,
        t: Tensor,
        pos_sde: KineticLangevinSDE,
        batch_idx: torch.LongTensor | None = None,
    ) -> Tensor:
        r"""Convert raw network output to the full velocity score.

        Inverts the simplified parameterisation applied during training and
        appends the analytic Gaussian velocity term:

        .. math::
            s^\text{full} = \text{pf}(t)\,\sqrt{\sigma_{\text{norm}}(t)}\,\hat{s}_\theta
                            \;-\; \frac{v_t}{\sigma_v^2(t)}

        Args:
            pred: Raw network output ``(N, 3)`` — the simplified target
                  predicted during training.
            vt: Current noisy velocity ``(N, 3)``.
            t: Diffusion time ``(B, 1)``.
            pos_sde: :class:`~kldm_new.diffusion.tdm.KineticLangevinSDE`.
            batch_idx: Atom -> graph mapping ``(N,)``.

        Returns:
            Full velocity score ``(N, 3)``.

        """
        gamma = pos_sde.gamma
        t_exp = maybe_expand(x=t, batch=batch_idx, like=vt)

        if self.simplified_parameterization:
            sigma_r_t = torch.sqrt(torch.clamp((2.0 / gamma**2.0) * (gamma * t_exp - 2.0 * torch.tanh((gamma * t_exp) / 2.0)), min=1e-12))
            sigma_flat = sigma_r_t.reshape(-1)
            sigma_unique, inv_idx = torch.unique(sigma_flat, return_inverse=True)
            sn_unique = self._lookup_sigma_norm(sigma_unique)
            sn = sn_unique[inv_idx].reshape(sigma_r_t.shape)
            # pred = WN_score / sqrt(sn)  →  WN contribution = pf(t) * sqrt(sn) * pred
            prefactor = self._prefactor_t(gamma, t_exp)
            wn_contribution = prefactor * torch.sqrt(sn.clamp(min=1e-8)) * pred
        else:
            wn_contribution = pred

        sigma_v_sq = (-torch.expm1(-2.0 * gamma * t_exp)).clamp(min=1e-12)
        gaussian_term = -vt / sigma_v_sq

        return wn_contribution + gaussian_term
