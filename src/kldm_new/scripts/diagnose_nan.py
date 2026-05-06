"""NaN diagnostic script for kldm_new.

Run on the HPC to identify exactly where NaN originates on GPU:

    python -m kldm_new.scripts.diagnose_nan

Or with a specific dataset:

    python -m kldm_new.scripts.diagnose_nan --data mp_20
    python -m kldm_new.scripts.diagnose_nan --data carbon_24
"""

from __future__ import annotations

import argparse
import warnings

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def chk(label: str, t: torch.Tensor, *, warn_inf: bool = True) -> bool:
    """Print NaN/Inf stats for tensor *t*.  Returns True if clean."""
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    ok = n_nan == 0 and (n_inf == 0 if warn_inf else True)
    tag = PASS if ok else FAIL
    print(
        f"  {tag} {label}: shape={tuple(t.shape)} dtype={t.dtype}"
        f"  NaN={n_nan}  Inf={n_inf}"
        f"  min={t.float().min().item():.3e}  max={t.float().max().item():.3e}"
    )
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 – sigma_norm lookup table built at KLDMLoss init
# ──────────────────────────────────────────────────────────────────────────────


def test_sigma_norm_table(device: torch.device) -> None:
    section("TEST 1: sigma_norm lookup table (KLDMLoss init)")
    from kldm_new.diffusion.loss import KLDMLoss

    loss_fn = KLDMLoss().to(device)
    sn = loss_fn._sn_values
    log_s = loss_fn._sn_log_sigma

    print(f"  {INFO} table size: {len(sn)}  device: {sn.device}")
    chk("_sn_values", sn, warn_inf=True)
    chk("_sn_log_sigma", log_s, warn_inf=False)

    # Check for zeros (zero sn → 1/sqrt(0) → Inf in target)
    n_zero = (sn == 0).sum().item()
    n_tiny = (sn < 1e-8).sum().item()
    print(f"  {INFO} entries == 0: {n_zero}   entries < 1e-8: {n_tiny}")
    if n_tiny > 0:
        print(f"  {FAIL} small sn_values will cause target = score_wn / sqrt(~0) → huge or NaN")
    else:
        print(f"  {PASS} all sn_values >= 1e-8")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 – d_log_p_wrapped_normal on GPU
# ──────────────────────────────────────────────────────────────────────────────


def test_d_log_wrapped_normal(device: torch.device) -> None:
    section("TEST 2: d_log_p_wrapped_normal on GPU")
    from kldm_new.diffusion import d_log_p_wrapped_normal

    for sigma_val in [0.0, 1e-13, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        sigma = torch.full((8, 3), sigma_val, device=device)
        x = torch.zeros_like(sigma)
        mu = torch.zeros_like(sigma)
        out = d_log_p_wrapped_normal(x, mu, sigma)
        n_nan = torch.isnan(out).sum().item()
        status = PASS if n_nan == 0 else FAIL
        print(f"  {status} sigma={sigma_val:.0e}: NaN={n_nan}  sample={out[0, 0].item():.3e}")

    # Test with typical runtime values (sigma_r_t range at small t)
    print()
    for t_val in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]:
        t = torch.tensor([[t_val]], device=device)
        gamma = 1.0
        base_var = (2.0 / gamma**2) * (gamma * t - 2.0 * torch.tanh(gamma * t / 2.0))
        sigma_rt = torch.sqrt(base_var.clamp(min=1e-12)).expand(8, 3)
        x = sigma_rt * torch.randn_like(sigma_rt) * 0.1  # near-zero samples
        mu = torch.zeros_like(x)
        out = d_log_p_wrapped_normal(x, mu, sigma_rt)
        n_nan = torch.isnan(out).sum().item()
        status = PASS if n_nan == 0 else FAIL
        print(f"  {status} t={t_val:.0e}: sigma_rt={sigma_rt[0, 0].item():.3e}  NaN={n_nan}  sample={out[0, 0].item():.3e}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 – Score model forward pass
# ──────────────────────────────────────────────────────────────────────────────


def test_score_model_forward(device: torch.device) -> None:
    section("TEST 3: Score model forward pass (step-by-step)")
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate
    from torch_geometric.utils import dense_to_sparse

    from kldm_new.data import add_velocity
    from kldm_new.diffusion.corruption import KLDMMultiCorruption
    from kldm_new.diffusion.lattice_sde import LatticeSubVPSDE
    from kldm_new.diffusion.tdm import KineticLangevinSDE
    from kldm_new.model import KLDMScoreModel
    from kldm_new.nn.arch import CSPVCellNet

    net = CSPVCellNet(hidden_dim=128, time_dim=128, num_layers=4, h_dim=100, num_freqs=16, ln=True)
    score_model = KLDMScoreModel(net).to(device)
    multi_corruption = KLDMMultiCorruption(KineticLangevinSDE(), LatticeSubVPSDE())

    # Build a realistic batch with a mix of atom counts (including 1-atom crystals)
    n_atoms_list = [1, 2, 5, 8, 12, 3]
    graphs = []
    for n in n_atoms_list:
        fc_edges, _ = dense_to_sparse(torch.ones(n, n) - torch.eye(n))
        g = ChemGraph(
            pos=torch.rand(n, 3),
            cell=(torch.eye(3) + torch.randn(3, 3) * 0.1).unsqueeze(0) * 3,
            atomic_numbers=torch.randint(1, 95, (n,)),
            num_atoms=torch.tensor(n),
            num_nodes=torch.tensor(n),
        )
        g = g.replace(edge_node_index=fc_edges)
        graphs.append(g)

    batch = collate(graphs)
    batch = batch.to(device)
    batch = add_velocity(batch)
    B = len(n_atoms_list)
    t = torch.rand(B, 1, device=device) * 0.99 + 1e-3

    # Corrupt
    noisy = multi_corruption.sample_marginal(batch, t)

    # Check inputs to score model
    print("  Inputs to score model:")
    chk("pos", noisy["pos"])
    chk("vel", noisy["vel"])
    chk("cell", noisy["cell"])
    chk("edge_node_index", noisy["edge_node_index"].float(), warn_inf=False)
    chk("atomic_numbers", noisy["atomic_numbers"].float(), warn_inf=False)

    # Check edge_node_index range
    max_idx = noisy["edge_node_index"].max().item()
    N_nodes = noisy["pos"].shape[0]
    print(f"  {INFO} edge_node_index max={max_idx}  N_nodes={N_nodes}")
    if max_idx >= N_nodes:
        print(f"  {FAIL} edge_node_index out of bounds!")
    else:
        print(f"  {PASS} edge_node_index in bounds")

    # Run the score model with hooks to find first NaN layer
    nan_found_in = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor) and torch.isnan(output).any():
                nan_found_in.append(name)
                print(f"  {FAIL} NaN FIRST APPEARED after layer: {name}")

        return hook

    handles = []
    for name, module in net.named_modules():
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        score_out = score_model(noisy, t)
    except Exception as e:
        print(f"  {FAIL} Exception during forward: {e}")
    finally:
        for h in handles:
            h.remove()

    if not nan_found_in:
        print(f"  {PASS} No NaN in any layer")
    else:
        print(f"  {FAIL} NaN first appeared in: {nan_found_in[0]}")

    print("\n  Score model outputs:")
    if "vel" in score_out:
        chk("vel output", score_out["vel"])
    if "cell" in score_out:
        chk("cell output", score_out["cell"])


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 – Loss computation
# ──────────────────────────────────────────────────────────────────────────────


def test_loss_computation(device: torch.device) -> None:
    section("TEST 4: Loss computation (step-by-step)")
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate
    from torch_geometric.utils import dense_to_sparse

    from kldm_new.data import add_velocity
    from kldm_new.diffusion import d_log_p_wrapped_normal
    from kldm_new.diffusion.corruption import KLDMMultiCorruption
    from kldm_new.diffusion.lattice_sde import LatticeSubVPSDE
    from kldm_new.diffusion.loss import KLDMLoss
    from kldm_new.diffusion.tdm import KineticLangevinSDE
    from kldm_new.model import KLDMScoreModel
    from kldm_new.nn.arch import CSPVCellNet

    net = CSPVCellNet(hidden_dim=64, time_dim=64, num_layers=2, h_dim=100, num_freqs=8, ln=True)
    score_model = KLDMScoreModel(net).to(device)
    pos_sde = KineticLangevinSDE()
    cell_sde = LatticeSubVPSDE()
    multi_corruption = KLDMMultiCorruption(pos_sde, cell_sde)
    loss_fn = KLDMLoss().to(device)

    n_atoms_list = [3, 5, 7, 4]
    graphs = []
    for n in n_atoms_list:
        fc_edges, _ = dense_to_sparse(torch.ones(n, n) - torch.eye(n))
        g = ChemGraph(
            pos=torch.rand(n, 3),
            cell=(torch.eye(3) + torch.randn(3, 3) * 0.1).unsqueeze(0) * 3,
            atomic_numbers=torch.randint(1, 95, (n,)),
            num_atoms=torch.tensor(n),
            num_nodes=torch.tensor(n),
        )
        g = g.replace(edge_node_index=fc_edges)
        graphs.append(g)

    batch = collate(graphs).to(device)
    batch = add_velocity(batch)
    B = len(n_atoms_list)
    t = torch.rand(B, 1, device=device) * 0.99 + 1e-3
    batch_idx = batch.get_batch_idx("pos")

    noisy = multi_corruption.sample_marginal(batch, t)
    score_out = score_model(noisy, t)

    print("  Intermediate loss tensors:")

    # Velocity branch
    mu_r_t, sigma_r_t = pos_sde.displacement_marginal(v0=batch["vel"], t=t, vt=noisy["vel"], batch_idx=batch_idx)
    chk("sigma_r_t", sigma_r_t)
    chk("mu_r_t", mu_r_t)

    r_t = pos_sde.wrap_disp(x=noisy["pos"] - batch["pos"], period=1.0)
    chk("r_t (wrapped displacement)", r_t)

    score_wn = d_log_p_wrapped_normal(x=r_t, mu=mu_r_t, sigma=sigma_r_t, N=loss_fn.k_wn_score)
    chk("score_wn", score_wn)

    # sigma_norm lookup
    sigma_flat = sigma_r_t.reshape(-1)
    sigma_unique, inv_idx = torch.unique(sigma_flat, return_inverse=True)
    sn_unique = loss_fn._lookup_sigma_norm(sigma=sigma_unique)
    chk("sn_unique (sigma_norm lookup result)", sn_unique)
    n_tiny = (sn_unique < 1e-8).sum().item()
    print(f"  {INFO} sn_unique < 1e-8: {n_tiny}/{len(sn_unique)}")

    sn = sn_unique[inv_idx].reshape(sigma_r_t.shape)
    target_vel = score_wn / torch.sqrt(sn.clamp(min=1e-8))
    chk("vel target (before center)", target_vel)

    # Check amplification factor
    amp = (1.0 / torch.sqrt(sn.clamp(min=1e-8))).float()
    print(f"  {INFO} amplification 1/sqrt(sn) range: {amp.min().item():.2e} – {amp.max().item():.2e}")

    # Cell branch
    mean, std = cell_sde.marginal_prob(batch["cell"], t, batch_idx=None, batch=noisy)
    chk("cell marginal mean", mean)
    chk("cell marginal std", std)
    n_tiny_std = (std < 1e-6).sum().item()
    print(f"  {INFO} cell std < 1e-6: {n_tiny_std}/{std.numel()}")
    noise = (noisy["cell"] - mean) / std.clamp(min=1e-8)
    chk("cell noise (cell_target = -noise)", noise)

    # Full loss
    total, d = loss_fn(
        multi_corruption=multi_corruption,
        batch=batch,
        noisy_batch=noisy,
        score_model_output=score_out,
        t=t,
    )
    print(f"\n  Loss: total={total.item():.4f}  vel={d['vel']:.4f}  cell={d['cell']:.4f}")
    if torch.isnan(total):
        print(f"  {FAIL} total loss is NaN")
    else:
        print(f"  {PASS} total loss is finite")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 – Load real data from disk and run one full training step
# ──────────────────────────────────────────────────────────────────────────────


def test_real_data_step(device: torch.device, data_root: str, dataset_name: str) -> None:
    section(f"TEST 5: Real data from disk ({dataset_name})")
    from pathlib import Path

    from mattergen.common.data.collate import collate
    from mattergen.common.data.dataset import CrystalDataset

    from kldm_new.data import add_velocity
    from kldm_new.data.transform import FullyConnectedGraph
    from kldm_new.diffusion.corruption import KLDMMultiCorruption
    from kldm_new.diffusion.lattice_sde import LatticeSubVPSDE
    from kldm_new.diffusion.loss import KLDMLoss
    from kldm_new.diffusion.tdm import KineticLangevinSDE
    from kldm_new.model import KLDMScoreModel
    from kldm_new.nn.arch import CSPVCellNet

    data_path = Path(data_root) / dataset_name / "processed" / "train"
    if not data_path.exists():
        print(f"  {INFO} Data path not found: {data_path}  (skipping)")
        return

    transforms = [FullyConnectedGraph(key="edge_node_index", len_from="pos")]
    ds = CrystalDataset.from_cache_path(str(data_path), transforms=transforms)
    print(f"  {INFO} Dataset size: {len(ds)}")

    # Load a small batch (indices 0–7)
    samples = [ds[i] for i in range(min(8, len(ds)))]
    batch = collate(samples).to(device)
    batch = add_velocity(batch)
    B = batch.get_batch_size()
    t = torch.rand(B, 1, device=device) * 0.99 + 1e-3

    net = CSPVCellNet(hidden_dim=128, time_dim=128, num_layers=4, h_dim=100, num_freqs=16, ln=True)
    score_model = KLDMScoreModel(net).to(device)
    multi_corruption = KLDMMultiCorruption(KineticLangevinSDE(), LatticeSubVPSDE())
    loss_fn = KLDMLoss().to(device)
    opt = torch.optim.Adam(score_model.parameters(), lr=5e-4)

    print(f"  {INFO} Batch: {B} graphs, {batch['pos'].shape[0]} atoms")
    print(f"  {INFO} num_atoms range: {batch['num_atoms'].min().item()} – {batch['num_atoms'].max().item()}")
    chk("pos (real data)", batch["pos"])
    chk("cell (real data)", batch["cell"])
    chk("atomic_numbers", batch["atomic_numbers"].float(), warn_inf=False)

    # Check for out-of-bounds atomic numbers
    max_z = batch["atomic_numbers"].max().item()
    print(f"  {INFO} max atomic_number: {max_z}  (embedding size: 101)")
    if max_z > 100:
        print(f"  {FAIL} atomic_numbers > 100 will cause Embedding index OOB!")

    # Run with anomaly detection to get the exact operation that produces NaN
    torch.autograd.set_detect_anomaly(True)
    try:
        noisy = multi_corruption.sample_marginal(batch, t)
        chk("noisy pos", noisy["pos"])
        chk("noisy vel", noisy["vel"])
        chk("noisy cell", noisy["cell"])

        score_out = score_model(noisy, t)
        chk("score vel output", score_out["vel"])
        chk("score cell output", score_out["cell"])

        loss, d = loss_fn(
            multi_corruption=multi_corruption,
            batch=batch,
            noisy_batch=noisy,
            score_model_output=score_out,
            t=t,
        )
        print(f"\n  Loss: total={loss.item():.4f}  vel={d['vel']:.4f}  cell={d['cell']:.4f}")

        opt.zero_grad()
        loss.backward()

        nan_params = [(n, p.grad) for n, p in score_model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
        if nan_params:
            print(f"  {FAIL} NaN gradients in: {[n for n, _ in nan_params[:5]]}")
        else:
            print(f"  {PASS} All gradients finite")

    except Exception as e:
        import traceback

        print(f"  {FAIL} Exception: {e}")
        traceback.print_exc()
    finally:
        torch.autograd.set_detect_anomaly(False)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 – Backward pass NaN reproduction at small t
# ──────────────────────────────────────────────────────────────────────────────


def test_backward_small_t(device: torch.device) -> None:
    section("TEST 6: Backward pass with very small t (most numerically challenging)")
    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate
    from torch_geometric.utils import dense_to_sparse

    from kldm_new.data import add_velocity
    from kldm_new.diffusion.corruption import KLDMMultiCorruption
    from kldm_new.diffusion.lattice_sde import LatticeSubVPSDE
    from kldm_new.diffusion.loss import KLDMLoss
    from kldm_new.diffusion.tdm import KineticLangevinSDE
    from kldm_new.model import KLDMScoreModel
    from kldm_new.nn.arch import CSPVCellNet

    net = CSPVCellNet(hidden_dim=128, time_dim=128, num_layers=4, h_dim=100, num_freqs=16, ln=True)
    score_model = KLDMScoreModel(net).to(device)
    pos_sde = KineticLangevinSDE()
    cell_sde = LatticeSubVPSDE()
    multi_corruption = KLDMMultiCorruption(pos_sde, cell_sde)
    loss_fn = KLDMLoss().to(device)
    opt = torch.optim.Adam(score_model.parameters(), lr=5e-4)

    n_atoms_list = [3, 5, 7, 4, 6, 2, 8, 3]
    B = len(n_atoms_list)

    for t_val in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0]:
        graphs = []
        for n in n_atoms_list:
            fc_edges, _ = dense_to_sparse(torch.ones(n, n) - torch.eye(n))
            g = ChemGraph(
                pos=torch.rand(n, 3),
                cell=(torch.eye(3) + torch.randn(3, 3) * 0.1).unsqueeze(0) * 3,
                atomic_numbers=torch.randint(1, 95, (n,)),
                num_atoms=torch.tensor(n),
                num_nodes=torch.tensor(n),
            )
            g = g.replace(edge_node_index=fc_edges)
            graphs.append(g)

        batch = collate(graphs).to(device)
        batch = add_velocity(batch)
        t = torch.full((B, 1), t_val, device=device)

        noisy = multi_corruption.sample_marginal(batch, t)
        score_out = score_model(noisy, t)
        loss, d = loss_fn(multi_corruption=multi_corruption, batch=batch, noisy_batch=noisy, score_model_output=score_out, t=t)

        opt.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            nan_g = sum(1 for p in score_model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        else:
            nan_g = -1  # didn't do backward

        tag = PASS if not torch.isnan(loss) and nan_g == 0 else FAIL
        print(f"  {tag} t={t_val:.0e}: loss={loss.item():.4f}  vel={d['vel']:.4f}  cell={d['cell']:.4f}  nan_grad_params={nan_g}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def test_stress_training_steps(device: torch.device, data_root: str | None, dataset_name: str) -> None:
    """Run 20 training steps and report the first NaN with anomaly detection."""
    section("TEST 7: 20-step stress test (first NaN with stack trace)")
    from pathlib import Path

    from mattergen.common.data.chemgraph import ChemGraph
    from mattergen.common.data.collate import collate
    from torch_geometric.utils import dense_to_sparse

    from kldm_new.data import add_velocity
    from kldm_new.data.transform import FullyConnectedGraph
    from kldm_new.diffusion.corruption import KLDMMultiCorruption
    from kldm_new.diffusion.lattice_sde import LatticeSubVPSDE
    from kldm_new.diffusion.loss import KLDMLoss
    from kldm_new.diffusion.tdm import KineticLangevinSDE
    from kldm_new.model import KLDMScoreModel
    from kldm_new.nn.arch import CSPVCellNet

    # Build data source
    samples = None
    if data_root:
        data_path = Path(data_root) / dataset_name / "processed" / "train"
        if data_path.exists():
            from mattergen.common.data.dataset import CrystalDataset

            transforms = [FullyConnectedGraph(key="edge_node_index", len_from="pos")]
            ds = CrystalDataset.from_cache_path(str(data_path), transforms=transforms)
            samples = [ds[i] for i in range(min(16, len(ds)))]
            print(f"  {INFO} Using real data from {dataset_name} ({len(samples)} samples)")

    if samples is None:
        print(f"  {INFO} Using synthetic data (no data_root found)")

    net = CSPVCellNet(hidden_dim=128, time_dim=128, num_layers=4, h_dim=100, num_freqs=16, ln=True)
    score_model = KLDMScoreModel(net).to(device)
    multi_corruption = KLDMMultiCorruption(KineticLangevinSDE(), LatticeSubVPSDE())
    loss_fn = KLDMLoss().to(device)
    opt = torch.optim.Adam(score_model.parameters(), lr=1e-4)

    n_atoms_list = [3, 5, 7, 4, 6, 2, 8, 3, 5, 4, 6, 7, 3, 4, 2, 5]

    nan_step = None
    torch.autograd.set_detect_anomaly(False)  # fast mode until NaN found

    for step in range(20):
        if samples is not None:
            batch = collate(samples).to(device)
        else:
            graphs = []
            for n in n_atoms_list:
                fc_edges, _ = dense_to_sparse(torch.ones(n, n) - torch.eye(n))
                g = ChemGraph(
                    pos=torch.rand(n, 3),
                    cell=(torch.eye(3) + torch.randn(3, 3) * 0.1).unsqueeze(0) * 3,
                    atomic_numbers=torch.randint(1, 95, (n,)),
                    num_atoms=torch.tensor(n),
                    num_nodes=torch.tensor(n),
                )
                g = g.replace(edge_node_index=fc_edges)
                graphs.append(g)
            batch = collate(graphs).to(device)

        batch = add_velocity(batch)
        B = batch.get_batch_size()
        t = torch.rand(B, 1, device=device) * 0.99 + 1e-3

        try:
            noisy = multi_corruption.sample_marginal(batch, t)
            score_out = score_model(noisy, t)
            loss, d = loss_fn(
                multi_corruption=multi_corruption,
                batch=batch,
                noisy_batch=noisy,
                score_model_output=score_out,
                t=t,
            )
            is_nan = torch.isnan(loss)
            tag = PASS if not is_nan else FAIL
            print(
                f"  {tag} step {step:02d}: total={loss.item():.4f}  vel={d['vel']:.4f}  cell={d['cell']:.4f}  t_range=[{t.min().item():.3f},{t.max().item():.3f}]"
            )

            if is_nan and nan_step is None:
                nan_step = step
                # Re-run with anomaly detection to get stack trace
                print(f"\n  {FAIL} NaN found at step {step}! Re-running with anomaly detection...")
                torch.autograd.set_detect_anomaly(True)
                try:
                    noisy2 = multi_corruption.sample_marginal(batch, t)
                    score_out2 = score_model(noisy2, t)
                    loss2, _ = loss_fn(
                        multi_corruption=multi_corruption,
                        batch=batch,
                        noisy_batch=noisy2,
                        score_model_output=score_out2,
                        t=t,
                    )
                    loss2.backward()
                except Exception as e:
                    print(f"  {FAIL} Anomaly detected: {e}")
                finally:
                    torch.autograd.set_detect_anomaly(False)
                break

            opt.zero_grad()
            if not is_nan:
                loss.backward()
                opt.step()

        except Exception as e:
            import traceback

            print(f"  {FAIL} step {step:02d}: Exception: {e}")
            traceback.print_exc()
            break

    if nan_step is None:
        print(f"\n  {PASS} All 20 steps passed without NaN.")
    else:
        print(f"\n  {FAIL} NaN first appeared at step {nan_step}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NaN diagnostics for kldm_new")
    parser.add_argument("--data", default="mp_20", help="Dataset name (mp_20, carbon_24, ...)")
    parser.add_argument("--data-root", default=None, help="Path to data root directory")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\n{INFO} PyTorch version: {torch.__version__}")
    print(f"{INFO} Running diagnostics on: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"{INFO} GPU: {p.name}  (SM {p.major}.{p.minor}, {p.total_memory // 1024**2} MB)")
        print(f"{INFO} CUDA version: {torch.version.cuda}")
        print(f"{INFO} cuDNN version: {torch.backends.cudnn.version()}")
        print(f"{INFO} TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}  TF32 cudnn: {torch.backends.cudnn.allow_tf32}")

    warnings.filterwarnings("ignore", category=UserWarning)

    test_sigma_norm_table(device)
    test_d_log_wrapped_normal(device)
    test_score_model_forward(device)
    test_loss_computation(device)
    test_backward_small_t(device)

    # Real data test (if data root is known)
    data_root = args.data_root
    if data_root is None:
        # Try common HPC paths
        import os

        for candidate in [
            "/zhome/a5/1/205686/kldm_mattergen/data",
            "/workspace/data",
            os.path.expanduser("~/kldm_mattergen/data"),
        ]:
            if os.path.isdir(candidate):
                data_root = candidate
                break

    if data_root:
        test_real_data_step(device, data_root, args.data)
    else:
        print(f"\n{INFO} No data root found, skipping real-data test.")
        print(f"{INFO} Pass --data-root /path/to/data to enable it.")

    test_stress_training_steps(device, data_root, args.data)

    print(f"\n{'=' * 70}")
    print("  Diagnostics complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
