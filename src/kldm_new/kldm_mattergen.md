# KLDM in the MatterGen Framework — `kldm_new`

## Overview

`kldm_new` is a **full reimplementation** of the Kinetic Langevin Diffusion Model (KLDM) using the [MatterGen](https://github.com/microsoft/mattergen) diffusion framework as its backbone. It does **not** wrap MatterGen modules — instead, it implements KLDM's unique coupled position–velocity physics as native extensions of MatterGen's `SDE`, `Corruption`, `Predictor`, `Corrector`, and `Loss` interfaces.

### Key design choices

| Aspect | kldm_frnct (original) | kldm_new (this package) |
|---|---|---|
| **Lattice representation** | 6-dim vector `l = [log_a, log_b, log_c, tan_α, tan_β, tan_γ]` | 3×3 cell matrix via MatterGen's `LatticeVPSDE` |
| **Position diffusion** | Custom `TDM` class with kinetic Langevin | `KineticLangevinSDE(SDE)` — MatterGen-native interface |
| **Multi-field corruption** | Manual orchestration in `KLDM.loss_diffusion` | `KLDMMultiCorruption(MultiCorruption)` — standard dispatch for cell, custom override for pos+vel |
| **Reverse-time sampling** | Hand-written PC loop in `KLDM.sample()` | `KLDMSampler` with `TDMPredictor` + `TDMLangevinCorrector` |
| **Data objects** | PyG `Data`/`Batch` with custom fields | MatterGen's `ChemGraph` (frozen PyG Data) with `replace()` |
| **Structure decoding** | Inverse transforms on 6-dim `l` → `Lattice.from_parameters` | Direct `Lattice(cell_matrix)` — no inverse transforms needed |

---

## Package structure

```
src/kldm_new/
├── __init__.py
├── data/
│   └── __init__.py           # add_velocity transform
├── diffusion/
│   ├── __init__.py            # distributions: d_log_p_wrapped_normal, sigma_norm, DistributionGaussian
│   ├── tdm.py                 # KineticLangevinSDE(SDE) — core TDM physics
│   ├── lattice_sde.py         # Re-export of MatterGen's LatticeVPSDE
│   ├── corruption.py          # KLDMMultiCorruption — ties TDM + LatticeVPSDE
│   ├── loss.py                # KLDMLoss — TDM wrapped-normal target + cell DSM
│   ├── predictors.py          # TDMPredictor — exponential integrator / DDIM
│   ├── correctors.py          # TDMLangevinCorrector — adaptive Langevin on velocity
│   └── sampling.py            # KLDMSampler — full PC reverse-time loop
├── nn/
│   ├── __init__.py            # SinEmbedding, FourierEmbedding
│   ├── utils.py               # scatter_center, wrap
│   └── arch.py                # CSPVCellNet — GNN score network
└── model/
    ├── __init__.py            # KLDMScoreModel wrapper
    └── lit_module.py          # LitKLDM — PyTorch Lightning module
```

---

## Diffusion pipeline

### Forward process (corruption)

The KLDM forward process corrupts three fields **simultaneously**:

1. **Velocity** (per-atom, 3D) — Ornstein–Uhlenbeck process:
   ```
   v_t = exp(-t) · v_0 + sqrt(1 - exp(-2t)) · ε_v
   ```
   At t → ∞, velocity converges to N(0, I).

2. **Position** (per-atom, fractional coords on [0,1)³) — Wrapped displacement:
   ```
   μ_r = (1 - exp(-t)) · v_0
   σ_r² = 2t - 3 + 4·exp(-t) - exp(-2t)
   pos_t = wrap(pos_0 + wrap(μ_r + σ_r · ε_r))
   ```
   Position depends on the initial velocity — this **coupling** is what makes TDM special and incompatible with standard per-field independent corruption.

3. **Cell** (per-graph, 3×3 matrix) — Variance-Preserving SDE:
   ```
   cell_t = α(t) · cell_0 + (1-α(t)) · μ_limit + σ(t) · ε_cell
   ```
   Where `μ_limit ∝ n_atoms^{1/3} · I` and `σ ∝ n_atoms^{1/3}` (from MatterGen's `LatticeVPSDE`). Noise is symmetrised to preserve lattice symmetry.

### Training target

- **Velocity target**: Score of the wrapped-normal displacement distribution:
  ```
  target = ∇_r log p_WN(r | μ_r, σ_r) · prefactor / sqrt(σ_norm)
  ```
  where `σ_norm = E[||∇ log p_WN||²]` (precomputed via Monte Carlo), and `prefactor = σ_r / σ_v`. This simplified parameterisation stabilises training.

- **Cell target**: Standard denoising score matching (predict −noise):
  ```
  target = -(cell_t - mean) / std
  ```

### Reverse process (sampling)

The `KLDMSampler` runs a Predictor–Corrector loop from t = T down to t = ε:

1. **Score computation**: Run the model once per step to get `(vel_score, cell_score)`.

2. **Corrector** (Langevin):
   - **Velocity**: Adaptive step size `δ = τ / mean(||score||²)`, then:
     ```
     v ← v + δ · score + sqrt(2δ) · z
     ```
   - **Cell**: Standard Langevin with `δ = snr² / mean(||score||²)` and symmetric noise.

3. **Predictor**:
   - **Velocity** (DDIM-like):
     ```
     v̂_0 = (v - σ_t · score) / α_t
     v_new = α_s · v̂_0 + σ_s · score
     ```
   - **Position**: `pos_new = wrap(pos - dt · v_new)` — deterministic propagation using updated velocity.
   - **Cell**: Ancestral sampling (standard MatterGen predictor).

---

## Score network: CSPVCellNet

The GNN architecture follows the original CSPVNet but adapted for the 3×3 cell representation:

```
Input: (t, pos, vel, h, cell, node_index, edge_index)
                │
    ┌───────────▼───────────┐
    │ Node embedding:       │
    │   h → Embed → [h, t]  │
    │   → Linear(hidden)    │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐  ×num_layers
    │ CSPVCellLayer:        │
    │  Edge: [h_i, h_j,     │
    │    cell_flat(9),       │  ← was 6-dim l in original
    │    v_proj(v_j-v_i),   │
    │    sin_emb(pos_diff)] │
    │  → MLP → edge_feat    │
    │  Node: [h, agg(edge)] │
    │  → MLP + residual     │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │ Readouts:             │
    │  vel: Linear(3)       │  zero-CoG enforced
    │  cell: scatter_mean → │
    │    Linear(9) → (3,3)  │
    └───────────────────────┘
```

The edge model takes 9-dim flattened cell (vs 6-dim `l` in the original), giving the network direct access to the full cell geometry.

---

## Usage example

```python
import torch
from mattergen.common.data.chemgraph import ChemGraph
from torch_geometric.data import Batch

from kldm_new.diffusion.tdm import KineticLangevinSDE
from kldm_new.diffusion.lattice_sde import LatticeVPSDE
from kldm_new.diffusion.corruption import KLDMMultiCorruption
from kldm_new.diffusion.loss import KLDMLoss
from kldm_new.nn.arch import CSPVCellNet
from kldm_new.model import KLDMScoreModel
from kldm_new.model.lit_module import LitKLDM

# Build corruption
pos_sde = KineticLangevinSDE(scale_pos=1.0, tf=2.0)
cell_sde = LatticeVPSDE(beta_min=0.1, beta_max=20.0)
corruption = KLDMMultiCorruption(pos_sde=pos_sde, cell_sde=cell_sde)

# Build score model
net = CSPVCellNet(hidden_dim=128, time_dim=128, num_layers=4, h_dim=100)
score_model = KLDMScoreModel(net=net, cutoff=0.5, max_neighbors=20)

# Build Lightning module
lit = LitKLDM(
    score_model=score_model,
    multi_corruption=corruption,
    loss_fn=KLDMLoss(weight_vel=1.0, weight_cell=1.0),
)

# Training: lit.training_step(batch, 0)
# Sampling: structures = lit.sample(batch)
```

---

## Relationship to MatterGen

| MatterGen component | KLDM usage |
|---|---|
| `SDE` (base class) | Extended by `KineticLangevinSDE` for TDM physics |
| `LatticeVPSDE` | Used directly for cell diffusion |
| `MultiCorruption` | Extended by `KLDMMultiCorruption` to handle coupled pos+vel |
| `ChemGraph` / `BatchedData` | Used as the data container (with added `vel` field) |
| `Predictor` | Extended by `TDMPredictor` for exponential integrator |
| `AncestralSamplingPredictor` | Used for cell predictor steps |
| `make_noise_symmetric_preserve_variance` | Used to symmetrise cell noise |

The key insight is that TDM's coupled position–velocity dynamics **cannot** be expressed as a standard MatterGen per-field SDE. Position sampling requires knowing the clean velocity (`v_0`), and the wrapped-normal score target is fundamentally different from Gaussian score matching. All custom components (`KineticLangevinSDE`, `TDMPredictor`, `TDMLangevinCorrector`, `KLDMLoss`) exist because of this coupling.

---

## Mathematical reference

### Wrapped normal score

The score of a wrapped normal distribution with period T:

$$\nabla_x \log p_{WN}(x \mid \mu, \sigma) = \frac{\sum_{n=-N}^{N} \frac{-(x - \mu - nT)}{\sigma^2} \exp\left(-\frac{(x-\mu-nT)^2}{2\sigma^2}\right)}{\sum_{n=-N}^{N} \exp\left(-\frac{(x-\mu-nT)^2}{2\sigma^2}\right)}$$

### Sigma norm

$$\sigma_{\text{norm}}(\sigma) = \mathbb{E}_{x \sim WN(0,\sigma)} \left[\|\nabla_x \log p_{WN}(x)\|^2\right]$$

Estimated via Monte Carlo (20,000 samples by default).

### Kinetic Langevin forward SDE

$$dv = -v\,dt + \sqrt{2}\,dW_v, \quad dr = v\,dt$$

### Exponential integrator (reverse)

$$v_{t-\Delta t} = e^{\Delta t} v_t + 2(e^{\Delta t}-1) s_\theta + \sqrt{e^{2\Delta t}-1}\, z$$
$$\text{pos}_{t-\Delta t} = \text{wrap}(\text{pos}_t - \Delta t \cdot v_{t-\Delta t})$$
