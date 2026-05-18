from __future__ import annotations

from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from mattergen.diffusion.corruption.corruption import maybe_expand
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from mattergen.diffusion.losses import SummedFieldLoss
from mattergen.diffusion.model_target import ModelTargets
from mattergen.diffusion.training.field_loss import (
    aggregate_per_sample,
    compute_noise_given_sample_and_corruption,
)
from torch import Tensor

from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE  # noqa: TC001
from kldm_plus.diffusion.corruption.utils import _scatter_center, d_log_p_wrapped_normal
from kldm_plus.diffusion.training.model_target import ModelTarget

# ---------------------------------------------------------------------------
# Kinetic Langevin position loss helper
# ---------------------------------------------------------------------------


def kinetic_pos_loss(
    *,
    sde: KineticLangevinSDE,
    score_model_output: Tensor,
    t: Tensor,
    batch_idx: Tensor,
    batch_size: int,
    pos_0: Tensor,
    pos_t: Tensor,
    v_t: Tensor,
    reduce: Literal["sum", "mean"] = "mean",
) -> Tensor:
    """Per-sample MSE loss for the position field.

    Under the simplified parameterisation (v_0 = 0) the training target is::

        target = scatter_center(d_log_p_WN(r, mu_r, sigma_r) / sqrt(sigma_norm_t))

    where *r* is the wrapped displacement pos_t - pos_0 in the Lie algebra,
    *mu_r* is the expected displacement conditioned on v_t, and
    *sigma_norm_t = E[||score_WN||^2]* is precomputed on the SDE.
    The ``scatter_center`` projection removes the per-crystal mean, enforcing
    the zero-CoG constraint on the training target — matching kldm_frnct.

    Returns a 1-D loss tensor of shape (batch_size,).
    """
    r = sde.wrap_disp(pos_t - pos_0, sde.scale_pos)

    mu_r, sigma_r = sde.displacement_marginal(
        v0=torch.zeros_like(v_t),
        t=t,
        vt=v_t,
        batch_idx=batch_idx,
    )
    mu_r = sde.wrap_disp(mu_r, sde.scale_pos)

    # Scale displacement and mean to the loss coordinate system (T = loss_pos_scale,
    # default 2π).  This matches kldm_frnct where positions live in [0, 2π), keeping
    # sigma/T ≤ 0.155 at t=1 instead of 0.977.  Both r and mu_r are already in
    # (-scale_pos/2, scale_pos/2); multiplying by (loss_pos_scale / scale_pos) maps
    # them to (-loss_pos_scale/2, loss_pos_scale/2) as required by d_log_p_WN(T=loss_pos_scale).
    pos_rescale = sde.loss_pos_scale / sde.scale_pos
    r_loss = r * pos_rescale
    mu_r_loss = mu_r * pos_rescale

    target = d_log_p_wrapped_normal(
        r_loss,
        mu_r_loss,
        sigma_r,
        N=sde.k_wn,
        T=sde.loss_pos_scale,
    )

    sigma_norm_t = sde._sigma_norm_t(t)  # [B]  — built with T=loss_pos_scale
    sigma_norm_atom = maybe_expand(x=sigma_norm_t, batch=batch_idx, like=pos_0)
    target = target / sigma_norm_atom.sqrt().clamp(min=1e-6)

    # Project target onto the zero-CoG manifold: matches kldm_frnct where
    # scatter_center is applied to target_pos_t before dividing by sigma_norm_t.
    # The true score lives on this manifold (zero-sum per crystal); removing
    # the per-crystal mean eliminates spurious drift and reduces target variance.
    target = _scatter_center(target, batch_idx)

    losses = (score_model_output - target).square()
    return aggregate_per_sample(
        losses,
        batch_idx=batch_idx,
        reduce=reduce,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Masking cross-entropy loss helper
# ---------------------------------------------------------------------------


def masking_cross_entropy_loss(
    *,
    score_model_output: Tensor,
    x: Tensor,
    noisy_x: Tensor,
    t: Tensor,
    batch_idx: Tensor,
    batch_size: int,
    vocab_size: int,
    reduce: Literal["sum", "mean"] = "mean",
    simple_loss: bool = False,
    dgamma_times_alpha: Tensor | None = None,
    **_,
) -> Tensor:
    """Per-sample cross-entropy for masked atom types.

    Only penalises atoms whose type was replaced by the mask token
    (noisy_x == vocab_size) in the forward process.

    Returns a 1-D loss tensor of shape (batch_size,).
    """
    log_p = torch.log_softmax(score_model_output, dim=-1)  # [num_atoms, vocab_size]
    one_hot = F.one_hot(x.long(), vocab_size).float()  # [num_atoms, vocab_size]

    neg_ce = (one_hot * log_p).sum(dim=-1, keepdim=True)  # [num_atoms, 1]
    mask = (noisy_x == vocab_size).float().unsqueeze(-1)  # [num_atoms, 1]
    masked_neg_ce = mask * neg_ce

    if simple_loss or dgamma_times_alpha is None:
        losses = -masked_neg_ce.squeeze(-1)
    else:
        losses = -(dgamma_times_alpha.unsqueeze(-1) * masked_neg_ce).squeeze(-1)

    return aggregate_per_sample(
        losses,
        batch_idx=batch_idx,
        reduce=reduce,
        batch_size=batch_size,
    )


def cell_continuous_loss(
    *,
    corruption,
    score_model_output: Tensor,
    t: Tensor,
    batch_idx: Tensor,
    batch_size: int,
    x: Tensor,
    noisy_x: Tensor,
    batch: BatchedData,
    reduce: Literal["sum", "mean"] = "mean",
    model_target: ModelTarget | str = ModelTarget.x0,
    **_,
) -> Tensor:
    """Per-sample MSE for continuous lattice field with configurable target.

    Targets:
    - ``x0``: clean value (matches original kldm_frnct lattice objective)
    - ``eps``: raw corruption noise ε
    - ``score_times_std``: -ε (MatterGen default)
    """
    target_kind = ModelTarget.from_any(model_target)
    if target_kind == ModelTarget.x0:
        target = x
    else:
        raw_noise = compute_noise_given_sample_and_corruption(
            x=x,
            x_noisy=noisy_x,
            corruption=corruption,
            t=t,
            batch_idx=batch_idx,
            batch=batch,
        )
        if target_kind == ModelTarget.eps:
            target = raw_noise
        elif target_kind == ModelTarget.score_times_std:
            target = -raw_noise
        else:
            msg = f"Unknown model_target {target_kind}"
            raise ValueError(msg)

    losses = (score_model_output - target).square()
    return aggregate_per_sample(
        losses,
        batch_idx=batch_idx,
        reduce=reduce,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# FieldLoss callables
# ---------------------------------------------------------------------------


class KineticFieldLoss:
    """FieldLoss callable for the kinetic Langevin position field.

    Plugs into SummedFieldLoss for the "pos" key.  Retrieves v_t from
    noisy_batch["vel"], which KineticLoss.__call__ adds to the broadcast dict
    so that all other FieldLoss callables are unaffected.
    """

    def __init__(
        self,
        sde: KineticLangevinSDE,
        reduce: Literal["sum", "mean"] = "mean",
    ) -> None:
        self.sde = sde
        self.reduce = reduce

    def __call__(
        self,
        *,
        score_model_output: Tensor,
        t: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        x: Tensor,  # clean pos  (batch["pos"])
        noisy_x: Tensor,  # noisy pos  (noisy_batch["pos"])
        noisy_batch: BatchedData,  # broadcast by KineticLoss; provides v_t
        corruption=None,
        **_,
    ) -> Tensor:
        return kinetic_pos_loss(
            sde=self.sde,
            score_model_output=score_model_output,
            t=t,
            batch_idx=batch_idx,
            batch_size=batch_size,
            pos_0=x,
            pos_t=noisy_x,
            v_t=noisy_batch["vel"],
            reduce=self.reduce,
        )


class MaskingFieldLoss:
    """FieldLoss callable for masked atom types.

    Plugs into SummedFieldLoss for the "atomic_numbers" key.
    Retrieves dgamma_times_alpha from the corruption object when available.
    """

    def __init__(
        self,
        vocab_size: int,
        simple_loss: bool = False,
        reduce: Literal["sum", "mean"] = "mean",
    ) -> None:
        self.vocab_size = vocab_size
        self.simple_loss = simple_loss
        self.reduce = reduce

    def __call__(
        self,
        *,
        score_model_output: Tensor,
        t: Tensor,
        batch_idx: Tensor,
        batch_size: int,
        x: Tensor,
        noisy_x: Tensor,
        batch: BatchedData,
        corruption=None,
        **_,
    ) -> Tensor:
        dgamma_times_alpha: Tensor | None = None
        if not self.simple_loss and corruption is not None and hasattr(corruption, "masking_schedule"):
            t_atom = maybe_expand(x=t, batch=batch.get_batch_idx("pos"), like=x.float())
            dgamma_times_alpha = corruption.masking_schedule.dgamma_times_alpha(t_atom)

        return masking_cross_entropy_loss(
            score_model_output=score_model_output,
            x=x,
            noisy_x=noisy_x,
            t=t,
            batch_idx=batch_idx,
            batch_size=batch_size,
            vocab_size=self.vocab_size,
            reduce=self.reduce,
            simple_loss=self.simple_loss,
            dgamma_times_alpha=dgamma_times_alpha,
        )


# ---------------------------------------------------------------------------
# KineticLoss — SummedFieldLoss with noisy_batch broadcast
# ---------------------------------------------------------------------------


class KineticLoss(SummedFieldLoss):
    """KLDM training loss: kinetic_sde pos + VP-SDE cell + (optionally) masking atomic_numbers.

    Set ``include_atomic_numbers=False`` for CSP runs where atom types are fixed
    inputs and are not diffused or predicted by the model.
    """

    def __init__(
        self,
        kinetic_sde: KineticLangevinSDE,
        vocab_size: int,
        w_pos: float = 1.0,
        w_cell: float = 1.0,
        w_h: float = 1.0,
        simple_loss: bool = False,
        reduce: Literal["sum", "mean"] = "mean",
        include_atomic_numbers: bool = True,
        cell_model_target: ModelTarget | str = "x0",
    ) -> None:
        cell_target = ModelTarget.from_any(cell_model_target)
        # Expose the configured cell target so the diffusion module can apply
        # the matching model-output-to-score conversion during sampling.
        self.cell_model_target = cell_target
        model_targets: ModelTargets = {
            "pos": ModelTarget.score_times_std,
            "cell": cell_target,
        }
        loss_fns: dict = {
            "pos": KineticFieldLoss(kinetic_sde, reduce),
            "cell": partial(
                cell_continuous_loss,
                reduce=reduce,
                model_target=cell_target,
            ),
        }
        weights: dict = {"pos": w_pos, "cell": w_cell}
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            loss_fns["atomic_numbers"] = MaskingFieldLoss(vocab_size, simple_loss, reduce)
            weights["atomic_numbers"] = w_h
        super().__init__(loss_fns=loss_fns, model_targets=model_targets, weights=weights)

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption,
        batch: BatchedData,
        noisy_batch: BatchedData,
        score_model_output: BatchedData,
        t: Tensor,
        node_is_unmasked=None,
        **_,
    ) -> tuple[Tensor, dict[str, float]]:
        batch_idx = {k: batch.get_batch_idx(k) for k in self.loss_fns}
        node_is_unmasked_dict = dict.fromkeys(self.loss_fns, node_is_unmasked)

        loss_per_sample_per_field = apply(
            fns=self.loss_fns,
            corruption=multi_corruption.corruptions,
            x=batch,
            noisy_x=noisy_batch,
            score_model_output=score_model_output,
            batch_idx=batch_idx,
            broadcast=dict(
                t=t,
                batch_size=batch.get_batch_size(),
                batch=batch,
                noisy_batch=noisy_batch,  # extra: KineticFieldLoss needs v_t
            ),
            node_is_unmasked=node_is_unmasked_dict,
        )

        assert set(v.shape for v in loss_per_sample_per_field.values()) == {(batch.get_batch_size(),)}, "All losses should have shape (batch_size,)."

        scalar_loss_per_field = {k: v.mean() for k, v in loss_per_sample_per_field.items()}
        agg_loss = torch.stack(
            [self.loss_weights[k] * v for k, v in loss_per_sample_per_field.items()],
            dim=0,
        ).sum(0)

        return agg_loss.mean(), scalar_loss_per_field
