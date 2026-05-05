"""KLDM Predictor–Corrector sampler.

Orchestrates reverse-time sampling for the coupled kinetic-Langevin
(pos + vel) and lattice-cell diffusion.

The standard MatterGen :class:`~mattergen.diffusion.sampling.predictors_correctors.PredictorCorrector`
treats each field independently, but KLDM requires that position updates
depend on the new velocity at each step.  This module reimplements the PC
loop with that coupling.

Predictors:

* :class:`~kldm_new.diffusion.predictors.TDMDDIMPredictor` (default) —
  DDIM-like deterministic velocity update.
* :class:`~kldm_new.diffusion.predictors.TDMEMPredictor` —
  stochastic exponential-integrator velocity update.

Correctors:

* :class:`~kldm_new.diffusion.correctors.TDMLangevinCorrector` — adaptive
  Langevin correction on velocity.
"""

from __future__ import annotations

import torch
from mattergen.common.diffusion.corruption import make_noise_symmetric_preserve_variance
from mattergen.diffusion.data.batched_data import BatchedData
from torch import Tensor

from kldm_new.diffusion.correctors import TDMLangevinCorrector
from kldm_new.diffusion.corruption import KLDMMultiCorruption
from kldm_new.diffusion.loss import KLDMLoss
from kldm_new.diffusion.predictors import TDMDDIMPredictor, TDMEMPredictor


class KLDMSampler:
    """Predictor–Corrector sampler for KLDM.

    Parameters
    ----------
    multi_corruption : KLDMMultiCorruption
        The multi-corruption holding TDM + cell SDEs.
    score_fn : callable
        ``(batch, t) → batch`` — returns model predictions for ``vel`` and
        ``cell`` fields given a noisy batch and time.
    loss_fn : KLDMLoss
        The loss module — used to reconstruct the full velocity score from
        the raw (simplified) network output.
    N : int
        Number of reverse-time discretisation steps.
    n_corrector_steps : int
        Langevin corrector steps per predictor step (velocity).
    n_corrector_steps_cell : int
        Langevin corrector steps per predictor step (cell).
    eps_t : float
        Minimum diffusion time (avoid exact zero).
    corrector_tau : float
        SNR parameter for TDM Langevin corrector.
    corrector_snr_cell : float
        SNR parameter for cell Langevin corrector.
    use_ddim : bool
        If ``True`` (default), use :class:`~kldm_new.diffusion.predictors.TDMDDIMPredictor`;
        otherwise use :class:`~kldm_new.diffusion.predictors.TDMEMPredictor`.

    """

    def __init__(
        self,
        multi_corruption: KLDMMultiCorruption,
        score_fn,
        loss_fn: KLDMLoss,
        N: int = 1000,
        n_corrector_steps: int = 1,
        n_corrector_steps_cell: int = 1,
        eps_t: float = 1e-3,
        corrector_tau: float = 0.5,
        corrector_snr_cell: float = 0.2,
        use_ddim: bool = True,
    ) -> None:
        """Initialise the KLDM sampler."""
        self.multi_corruption = multi_corruption
        self.score_fn = score_fn
        self.loss_fn = loss_fn
        self.N = N
        self.n_corrector_steps = n_corrector_steps
        self.n_corrector_steps_cell = n_corrector_steps_cell
        self.eps_t = eps_t
        self.corrector_tau = corrector_tau
        self.corrector_snr_cell = corrector_snr_cell

        pos_sde = multi_corruption.pos_sde
        PredictorCls = TDMDDIMPredictor if use_ddim else TDMEMPredictor  # noqa: N806
        self.tdm_predictor = PredictorCls(corruption=pos_sde)
        self.tdm_corrector = TDMLangevinCorrector(
            corruption=pos_sde,
            n_steps=n_corrector_steps,
            tau=corrector_tau,
        )

    @torch.no_grad()
    def sample(
        self,
        conditioning_data: BatchedData,
    ) -> BatchedData:
        """Run the full reverse-time PC sampling loop.

        Args:
            conditioning_data: A batch with ``num_atoms`` (and optionally
                ``atomic_numbers``) used for prior sampling.

        Returns:
            Denoised batch with ``pos``, ``vel``, ``cell`` fields.

        """
        device = conditioning_data["pos"].device
        batch_size = conditioning_data.get_batch_size()
        pos_batch_idx = conditioning_data.get_batch_idx("pos")
        n_atoms_total = conditioning_data["pos"].shape[0]

        pos_sde = self.multi_corruption.pos_sde
        T = self.multi_corruption.T
        dt = (T - self.eps_t) / self.N

        # ---- Prior sampling ----
        vel = pos_sde.prior_sampling(shape=(n_atoms_total, 3)).to(device)
        pos = torch.rand(n_atoms_total, 3, device=device) * pos_sde.scale_pos
        cell = self.multi_corruption.cell_sde.prior_sampling(
            shape=(batch_size, 3, 3),
            conditioning_data=conditioning_data,
        ).to(device)

        batch = conditioning_data.replace(pos=pos, vel=vel, cell=cell)
        timesteps = torch.linspace(T, self.eps_t, self.N + 1, device=device)

        for i in range(self.N):
            t_curr = timesteps[i]
            t_vec = torch.full((batch_size, 1), t_curr.item(), device=device)

            # -- Get raw model outputs --
            score_batch = self.score_fn(batch, t_vec)
            vel_pred_raw = score_batch["vel"]  # simplified network output (N_atoms, 3)
            cell_score = score_batch["cell"]  # score*std output (B, 3, 3)

            # -- Reconstruct full velocity score from simplified target --
            vel_score = self.loss_fn.reconstruct_velocity_score(
                pred=vel_pred_raw,
                v_t=vel,
                t=t_vec,
                pos_sde=pos_sde,
                batch_idx=pos_batch_idx,
            )

            # -- Corrector: Langevin on velocity (position unchanged) --
            for _ in range(self.n_corrector_steps):
                vel, _ = self.tdm_corrector.step_given_score(
                    x=vel,
                    batch_idx=pos_batch_idx,
                    score=vel_score,
                    t=t_vec,
                    dt=dt,
                )

            # -- Corrector: Langevin on cell --
            cell_score_actual = self._model_out_to_score(cell_score, t_vec, batch)
            for _ in range(self.n_corrector_steps_cell):
                cell = self._cell_langevin_step(cell, cell_score_actual, dt)

            # -- Predictor: velocity update --
            vel, _ = self.tdm_predictor.update_given_score(
                x=vel,
                t=t_vec,
                dt=dt,
                batch_idx=pos_batch_idx,
                score=vel_score,
                batch=batch,
            )

            # -- Predictor: position update (deterministic, driven by new vel) --
            pos = pos_sde.wrap_pos(pos - dt * vel, pos_sde.scale_pos)

            # -- Predictor: cell update (ancestral sampling) --
            cell_score_actual2 = self._model_out_to_score(cell_score, t_vec, batch)
            cell = self._cell_ancestral_step(cell, cell_score_actual2, t_vec, dt, batch)

            batch = batch.replace(pos=pos, vel=vel, cell=cell)

        return batch

    # ---- Cell helpers -------------------------------------------------------

    def _model_out_to_score(self, model_out: Tensor, t: Tensor, batch: BatchedData) -> Tensor:
        """Convert cell model output (score × std) to actual score."""
        _, std = self.multi_corruption.cell_sde.marginal_prob(
            x=torch.ones_like(model_out),
            t=t,
            batch_idx=None,
            batch=batch,
        )
        return model_out / std.clamp(min=1e-8)

    def _cell_ancestral_step(
        self,
        cell: Tensor,
        score: Tensor,
        t: Tensor,
        dt: float,
        batch: BatchedData,
    ) -> Tensor:
        """Ancestral sampling predictor step for the 3×3 cell field."""
        sde = self.multi_corruption.cell_sde
        s = t + (-dt)  # going backward: s = t - dt

        alpha_t, sigma_t = sde.mean_coeff_and_std(x=cell, t=t, batch_idx=None, batch=batch)
        alpha_s, sigma_s = sde.mean_coeff_and_std(x=cell, t=s.clamp(min=0.0), batch_idx=None, batch=batch)

        # Zero out sigma_s at the final step
        is_done = (s <= 0).float()
        while is_done.ndim < sigma_s.ndim:
            is_done = is_done.unsqueeze(-1)
        sigma_s = sigma_s * (1.0 - is_done)

        sigma2_t_given_s = sigma_t**2 - sigma_s**2 * alpha_t**2 / alpha_s**2
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s.clamp(min=1e-12))
        std = sigma_t_given_s * sigma_s / sigma_t.clamp(min=1e-12) * (1.0 - is_done)

        alpha_t_given_s = (alpha_t / alpha_s).clamp(min=0.001)
        score_coeff = sigma2_t_given_s / alpha_t_given_s
        x_coeff = 1.0 / alpha_t_given_s

        noise = make_noise_symmetric_preserve_variance(torch.randn_like(cell))
        mean = x_coeff * cell + score_coeff * score
        return mean + std * noise

    def _cell_langevin_step(self, cell: Tensor, score: Tensor, dt: float) -> Tensor:  # noqa: ARG002
        """Single Langevin corrector step for the cell field."""
        score_norm_sq = (score**2).mean().clamp(min=1e-8)
        delta = self.corrector_snr_cell**2 / score_norm_sq
        noise = make_noise_symmetric_preserve_variance(torch.randn_like(cell))
        return cell + delta * score + (2.0 * delta).sqrt() * noise
