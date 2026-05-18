"""Cross-check training targets and losses between kldm_frnct and kldm_plus.

kldm_frnct is the original research implementation (module internally named
src_kldm, renamed to kldm_frnct).  kldm_plus re-implements the same physics
inside the mattergen framework.

Tests
-----
1. sigma_r formula          — same closed-form formula, same numbers.
2. mu_r formula             — tanh-based drift, same numbers.
3. d_log_p_wrapped_normal   — both functions are byte-for-byte the same formula.
4. sigma_norm               — stochastic but must agree to within 5 %.
5. training target          — given identical (r, mu_r, sigma_r, sigma_norm),
                              the normalised target d_log_p / sqrt(sigma_norm)
                              must match.
6. kldm_frnct smoke test    — TDM.training_targets produces finite tensors,
                              TDM.loss_diffusion returns a finite scalar.
7. kldm_plus smoke test     — KineticDiffusionModule.calc_loss returns a finite
                              scalar on a tiny synthetic ChemGraph batch.

Run with:
    uv run pytest tests/test__kldm_loss_equivalence.py -v
"""

from __future__ import annotations

import importlib
import sys

import torch

# ---------------------------------------------------------------------------
# Monkeypatch: make "src_kldm.*" importable as "kldm_frnct.*"
# kldm_frnct was written when the package was called src_kldm; the directory
# was renamed but internal imports still use the old prefix.
# ---------------------------------------------------------------------------
import kldm_frnct as _frnct_top

sys.modules.setdefault("src_kldm", _frnct_top)

for _sub in [
    "data",
    "data.datamodule",
    "data.dataset",
    "data.transforms",
    "data.utils",
    "model",
    "model.base",
    "model.continuous",
    "model.discrete",
    "model.distributions",
    "model.kldm",
    "model.tdm",
    "nn",
    "nn.arch",
    "nn.embedding",
    "nn.utils",
    "lit",
    "lit.module",
]:
    _src = f"src_kldm.{_sub}"
    _kldm = f"kldm_frnct.{_sub}"
    if _src not in sys.modules:
        try:
            sys.modules[_src] = importlib.import_module(_kldm)
        except ImportError:
            pass

# ---------------------------------------------------------------------------
# Imports — both implementations
# ---------------------------------------------------------------------------
from kldm_frnct.model.distributions import (  # noqa: E402
    d_log_p_wrapped_normal as frnct_d_log_p,
)
from kldm_frnct.model.distributions import (
    sigma_norm as frnct_sigma_norm,
)
from kldm_frnct.model.tdm import TDM  # noqa: E402
from kldm_plus.diffusion.corruption.sde import KineticLangevinSDE  # noqa: E402
from kldm_plus.diffusion.corruption.utils import (  # noqa: E402
    d_log_p_wrapped_normal as plus_d_log_p,
)
from kldm_plus.diffusion.corruption.utils import (
    sigma_norm as plus_sigma_norm,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------
SCALE_POS = 1.0  # run both on the same torus period
TF = 2.0
GAMMA = 1.0
K_WN = 5  # small for speed; 13 in production
N_SIGMAS = 200  # small for speed
BATCH = 2  # crystals
N_ATOMS = 6  # atoms per crystal
N_TOTAL = BATCH * N_ATOMS


def make_tdm() -> TDM:
    """TDM with scale_pos=1.0 so both models share the same period."""
    return TDM(
        scale_pos=SCALE_POS,
        k_wn_score=K_WN,
        tf=TF,
        simplified_parameterization=True,
        n_sigmas=N_SIGMAS,
    )


def make_sde() -> KineticLangevinSDE:
    # loss_pos_scale must match SCALE_POS so the sigma_norm table is built
    # with the same torus period T as kldm_frnct (which uses T=scale_pos).
    return KineticLangevinSDE(
        scale_pos=SCALE_POS,
        tf=TF,
        gamma=GAMMA,
        k_wn=K_WN,
        n_sigmas=N_SIGMAS,
        loss_pos_scale=SCALE_POS,
    )


def make_batch_index(batch: int = BATCH, n_atoms: int = N_ATOMS) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(batch), n_atoms)


# ---------------------------------------------------------------------------
# 1. sigma_r formula
# ---------------------------------------------------------------------------


def test_sigma_r_equivalence():
    """Both implementations must give identical sigma_r at the same internal time.

    Note: at very small tau (< 0.05) kldm_frnct adds eps=1e-6 inside sqrt while
    kldm_plus uses clamp(min=1e-12).  The formulas are algebraically identical but
    the numerical stabilizers differ.  We test tau >= 0.05 where sigma_r^2 >> 1e-6.
    """
    tdm = make_tdm()
    sde = make_sde()

    # Avoid near-zero tau where different eps/clamp stabilizers diverge.
    tau = torch.tensor([0.1, 0.5, 1.0, 1.5, TF])

    frnct_val = tdm._sigma_r_t(tau)
    plus_val = sde._sigma_r_tau(tau)

    assert torch.allclose(frnct_val, plus_val, atol=1e-3), f"sigma_r mismatch:\n  frnct: {frnct_val}\n  plus:  {plus_val}"


# ---------------------------------------------------------------------------
# 2. mu_r (displacement mean) formula
# ---------------------------------------------------------------------------


def test_mu_r_equivalence():
    """Both implementations must give identical displacement mean mu_r."""
    tdm = make_tdm()
    sde = make_sde()

    torch.manual_seed(0)
    v0 = torch.zeros(N_TOTAL, 3)  # simplified param: v0 = 0
    vt = torch.randn(N_TOTAL, 3)
    index = make_batch_index()

    # kldm_frnct: internal time τ ∈ [0, tf]
    tau = torch.full((N_TOTAL, 1), 1.0)  # τ = 1.0 for every atom

    frnct_mu = tdm._mu_r_t(tau, v0, vt)

    # kldm_plus: external time t ∈ [0, 1], tau = tf * t → t = 1.0/tf = 0.5
    t_ext = torch.full((BATCH,), 1.0 / TF)  # external time
    plus_mu, _ = sde.displacement_marginal(v0=v0, t=t_ext, vt=vt, batch_idx=index)

    assert torch.allclose(frnct_mu, plus_mu, atol=1e-5), f"mu_r mismatch:\n  max abs diff = {(frnct_mu - plus_mu).abs().max():.2e}"


# ---------------------------------------------------------------------------
# 3. d_log_p_wrapped_normal
# ---------------------------------------------------------------------------


def test_d_log_p_wrapped_normal_equivalence():
    """Both d_log_p_wrapped_normal functions must return the same values."""
    torch.manual_seed(42)
    x = torch.randn(20, 3) * 0.2
    mu = torch.zeros_like(x)
    sigma = torch.full_like(x, 0.3)

    frnct_val = frnct_d_log_p(x, mu, sigma, N=K_WN, T=SCALE_POS)
    plus_val = plus_d_log_p(x, mu, sigma, N=K_WN, T=SCALE_POS)

    assert torch.allclose(frnct_val, plus_val, atol=1e-6), f"d_log_p mismatch: max abs diff = {(frnct_val - plus_val).abs().max():.2e}"


# ---------------------------------------------------------------------------
# 4. sigma_norm (stochastic — allow 5% relative tolerance)
# ---------------------------------------------------------------------------


def test_sigma_norm_equivalence():
    """Both sigma_norm implementations must agree within 5% relative error."""
    tau = torch.tensor([0.5, 1.0, TF])
    sde = make_sde()
    tdm = make_tdm()

    sigma_r = sde._sigma_r_tau(tau)

    # Use a generous sn for stable estimates.
    torch.manual_seed(7)
    frnct_val = frnct_sigma_norm(sigma_r, T=SCALE_POS, N=K_WN, sn=50_000)
    torch.manual_seed(7)
    plus_val = plus_sigma_norm(sigma_r, T=SCALE_POS, N=K_WN, sn=50_000)

    # Relative tolerance: |frnct - plus| / (|frnct| + 1e-8) < 5 %
    rel_err = (frnct_val - plus_val).abs() / (frnct_val.abs() + 1e-8)
    assert (rel_err < 0.05).all(), f"sigma_norm relative error > 5%:\n  frnct: {frnct_val}\n  plus:  {plus_val}\n  rel:   {rel_err}"


# ---------------------------------------------------------------------------
# 5. Training target equivalence
# ---------------------------------------------------------------------------


def test_training_target_equivalence():
    """Given identical (r, mu_r, sigma_r, sigma_norm), both implementations must
    compute the same normalised training target d_log_p_WN / sqrt(sigma_norm).
    """
    tdm = make_tdm()
    sde = make_sde()

    torch.manual_seed(99)
    index = make_batch_index()
    t_int = torch.full((N_TOTAL, 1), 0.8)  # internal τ = 0.8
    t_ext = torch.full((BATCH,), 0.8 / TF)  # external t = τ / tf

    # --- kldm_frnct path ---
    v0 = torch.zeros(N_TOTAL, 3)
    vt = torch.randn(N_TOTAL, 3)
    # Scatter-center vt to zero CoG per crystal
    from torch_scatter import scatter_mean

    vt = vt - scatter_mean(vt, index=index, dim=0)[index]

    # Sample eps_r with zero CoG
    eps_r = torch.randn(N_TOTAL, 3)
    eps_r = eps_r - scatter_mean(eps_r, index=index, dim=0)[index]

    sigma_r = tdm._sigma_r_t(t_int)
    mu_r = tdm._mu_r_t(t_int, v0, vt)
    r = mu_r + sigma_r * eps_r

    # Wrap both displacement r and mean mu_r onto [-T/2, T/2)
    r_w = tdm._wrap(r)
    mu_r_w = tdm._wrap(mu_r)

    prefactor = tdm._prefactor_t(t_int)
    # target_pos_t = prefactor * d_log_p_WN (includes prefactor)
    target_pos_t = frnct_d_log_p(r_w, mu_r_w, sigma_r, N=K_WN, T=SCALE_POS)
    target_pos_t = prefactor * target_pos_t
    # simplified_parameterization: divide by prefactor * sqrt(sigma_norm)
    sigma_norm_t = torch.sqrt(tdm._sigma_norm_t(t_int))  # sqrt of lookup
    frnct_target = target_pos_t / prefactor / sigma_norm_t  # = d_log_p / sqrt(sigma_norm)
    # kldm_frnct applies scatter_center to enforce the zero-CoG manifold constraint
    from torch_scatter import scatter_mean as _scatter_mean

    frnct_target = frnct_target - _scatter_mean(frnct_target, index=index, dim=0)[index]

    # --- kldm_plus path ---
    # Reuse same r_w, mu_r_w, sigma_r, sigma_norm by direct call.
    sigma_norm_plus = sde._sigma_norm_t(t_ext)  # shape [B]
    # Expand to [N_TOTAL, 1] for broadcasting against [N_TOTAL, 3]
    sigma_norm_atom = sigma_norm_plus[index].unsqueeze(-1)  # [N_TOTAL, 1]

    target_plus = plus_d_log_p(r_w, mu_r_w, sigma_r, N=K_WN, T=SCALE_POS)
    target_plus = target_plus / sigma_norm_atom.sqrt().clamp(min=1e-6)
    # kldm_plus applies scatter_center (matching kldm_frnct)
    from kldm_plus.diffusion.corruption.utils import _scatter_center

    target_plus = _scatter_center(target_plus, index)

    # The sigma_norm table is a Monte Carlo estimate (sn=N_SIGMAS=200 samples),
    # so we allow 5% absolute tolerance — both values are O(1).
    assert torch.allclose(frnct_target, target_plus, atol=5e-2), (
        f"training target mismatch: max abs diff = {(frnct_target - target_plus).abs().max():.2e}"
    )


# ---------------------------------------------------------------------------
# 6. kldm_frnct smoke test — TDM forward pass
# ---------------------------------------------------------------------------


def test_kldm_frnct_tdm_forward():
    """TDM.training_targets + loss_diffusion must produce finite outputs."""
    tdm = make_tdm()
    index = make_batch_index()

    torch.manual_seed(0)
    pos01 = torch.rand(N_TOTAL, 3)  # fractional coords ∈ [0, 1)
    t01 = torch.rand(BATCH, 1)  # external time per crystal, broadcast below
    t_atom = t01[index]  # shape [N_TOTAL, 1]

    (v_t, pos_t), target = tdm.training_targets(t01=t_atom, pos01=pos01, index=index)

    assert torch.isfinite(target).all(), "TDM training target contains NaN/Inf"
    assert torch.isfinite(v_t).all(), "TDM v_t contains NaN/Inf"
    assert torch.isfinite(pos_t).all(), "TDM pos_t contains NaN/Inf"

    # Compute loss with a zero-prediction (worst case, just checking no crash).
    pred = torch.zeros_like(target)
    loss = tdm.loss_diffusion(pred, target)
    assert torch.isfinite(loss), f"TDM loss is not finite: {loss}"
    assert loss >= 0, f"TDM MSE loss is negative: {loss}"


# ---------------------------------------------------------------------------
# 7. kldm_plus smoke test — KineticDiffusionModule.calc_loss
# ---------------------------------------------------------------------------


def _make_kldm_plus_module():
    """Return a tiny KineticDiffusionModule for smoke testing."""
    from mattergen.common.diffusion.corruption import LatticeVPSDE

    from kldm_plus.diffusion.corruption.kinetic_multi_corruption import KineticMultiCorruption
    from kldm_plus.diffusion.diffusion_module import KineticDiffusionModule
    from kldm_plus.diffusion.score_models.base import KineticScoreModel
    from kldm_plus.diffusion.training.loss import KineticLoss
    from kldm_plus.nn.net import CSPVNet

    kinetic_sde = make_sde()
    cell_sde = LatticeVPSDE.from_vpsde_config(vpsde_config=dict(beta_min=0.1, beta_max=20, limit_density=0.05, limit_var_scaling_constant=0.25))
    corruption = KineticMultiCorruption(
        kinetic_sde=kinetic_sde,
        sdes={"cell": cell_sde},
        discrete_corruptions={},
    )
    loss_fn = KineticLoss(
        kinetic_sde=kinetic_sde,
        vocab_size=8,  # tiny
        w_pos=1.0,
        w_cell=1.0,
        w_h=0.0,
        simple_loss=False,
        reduce="mean",
    )
    # KineticLoss always includes an atomic_numbers head but our corruption has
    # no D3PM masking (discrete_corruptions={}).  Remove it so log_softmax
    # is not called on raw Long-type atomic_numbers.
    del loss_fn.loss_fns["atomic_numbers"]
    del loss_fn.model_targets["atomic_numbers"]
    loss_fn.weights = {"pos": 1.0, "cell": 1.0}
    net = CSPVNet(
        hidden_dim=8,  # tiny
        time_dim=8,
        num_layers=1,
        atom_dim=8,
        num_freqs=2,
        ln=False,
        pred_vel=True,
        pred_cell=True,
        pred_atom=False,
        zero_cog=True,
        lattice_dim=9,
    )
    score_model = KineticScoreModel(net=net)
    return KineticDiffusionModule(
        model=score_model,
        corruption=corruption,
        loss_fn=loss_fn,
    )


def _make_synthetic_batch(batch: int = BATCH, n_atoms: int = N_ATOMS):
    """Return a small batched KineticChemGraph on CPU."""
    import torch_geometric.data as pyg_data

    from kldm_plus.data.kineticchemgraph import KineticChemGraph

    graphs = []
    for _ in range(batch):
        n = n_atoms
        # Fractional coords ∈ [0, 1)
        pos = torch.rand(n, 3)
        vel = torch.zeros(n, 3)
        # Simple identity-like lattice (scaled to ~5 Å)
        cell = 5.0 * torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        atomic_numbers = torch.randint(1, 8, (n,))

        # Fully-connected edges (including self-loops for simplicity)
        rows = torch.arange(n).repeat(n)
        cols = torch.arange(n).repeat_interleave(n)
        edge_node_index = torch.stack([rows, cols], dim=0)  # (2, n*n)

        g = KineticChemGraph(
            atomic_numbers=atomic_numbers,
            pos=pos,
            cell=cell,
            vel=vel,
            edge_node_index=edge_node_index,
            num_atoms=torch.tensor(n),  # required by LatticeVPSDE for density scaling
        )
        graphs.append(g)

    # PyG batching — follow_batch ensures vel_batch is set.
    return pyg_data.Batch.from_data_list(graphs, follow_batch=["vel"])


def test_kldm_plus_calc_loss():
    """KineticDiffusionModule.calc_loss must return a finite scalar."""
    torch.manual_seed(123)
    module = _make_kldm_plus_module()
    module.eval()

    batch = _make_synthetic_batch()
    loss, metrics = module.calc_loss(batch)

    assert loss.ndim == 0, f"loss should be a scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"calc_loss returned non-finite loss: {loss.item()}"
    for k, v in metrics.items():
        assert torch.isfinite(v), f"metric '{k}' is not finite: {v}"


def test_kldm_plus_gradient_flows():
    """A backward pass on calc_loss must produce finite gradients."""
    torch.manual_seed(456)
    module = _make_kldm_plus_module()
    module.train()

    batch = _make_synthetic_batch()
    loss, _ = module.calc_loss(batch)
    loss.backward()

    for name, param in module.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in parameter '{name}'"
