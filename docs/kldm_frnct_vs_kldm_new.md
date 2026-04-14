# KLDM: `kldm_frnct` vs `kldm_new` — Comparative Analysis

> **Audience**: Readers familiar with the `kldm_frnct` codebase who need to understand
> how `kldm_new` reimplements the same physics inside the MatterGen framework.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Lattice Representation](#3-lattice-representation)
4. [Coordinate System & Wrapping](#4-coordinate-system--wrapping)
5. [SDE Framework](#5-sde-framework)
6. [TDM (Kinetic Langevin) — The Core Diffusion](#6-tdm-kinetic-langevin--the-core-diffusion)
7. [Lattice Diffusion (VP-SDE)](#7-lattice-diffusion-vp-sde)
8. [Multi-Modal Orchestration](#8-multi-modal-orchestration)
9. [Graph Neural Network](#9-graph-neural-network)
10. [Training Pipeline](#10-training-pipeline)
11. [Sampling (Reverse-Time)](#11-sampling-reverse-time)
12. [Data Pipeline](#12-data-pipeline)
13. [HPC / Deployment](#13-hpc--deployment)
14. [Quick Reference Table](#14-quick-reference-table)

---

## 1. Executive Summary

`kldm_new` is a **full reimplementation** of the KLDM crystal diffusion model inside
the MatterGen framework. It is **not** a wrapper — every module was rebuilt to use MatterGen's
native abstractions (`SDE`, `MultiCorruption`, `Predictor`, `Corrector`, `BatchedData`,
`CrystalDataset`).

The physics and mathematics are equivalent, but several design choices differ:

| Aspect | `kldm_frnct` | `kldm_new` |
|--------|-------------|------------|
| Lattice | 6 params (3 lengths + 3 angles) | 3×3 cell matrix |
| Coordinates | Scaled to $[0, 2\pi)$ | Fractional $[0, 1)$ |
| Graph | Fully-connected | Radius graph |
| SDE classes | Custom standalone | Inherits MatterGen `SDE` |
| Data format | `.pt` files (PyG `Data`) | Numpy cache (MatterGen `CrystalDataset`) |
| Config | Hydra (custom) | Hydra (MatterGen-compatible) |

---

## 2. Architecture at a Glance

### `kldm_frnct`

```
lit/module.py  →  LitKLDM (Lightning)
                    ↓
model/kldm.py  →  KLDM (nn.Module)
                    ├── net: CSPVNet          (GNN)
                    └── diffusions: ModuleDict
                         ├── "v": TDM                (kinetic Langevin)
                         ├── "l": ContinuousDiffusion (VP-SDE on 6-param lattice)
                         └── "h": DiscreteDiffusion   (optional, for DNG tasks)

nn/arch.py     →  CSPVNet → CSPVLayer (message-passing)
data/          →  Dataset (.pt) + transforms + DataModule
```

### `kldm_new`

```
model/lit_module.py  →  LitKLDM (Lightning)
                          ├── score_model: KLDMScoreModel
                          ├── multi_corruption: KLDMMultiCorruption
                          └── loss_fn: KLDMLoss

model/__init__.py    →  KLDMScoreModel (wraps CSPVCellNet + radius_graph)
nn/arch.py           →  CSPVCellNet → CSPVCellLayer (message-passing)

diffusion/
  ├── tdm.py         →  KineticLangevinSDE(SDE)   — MatterGen SDE subclass
  ├── lattice_sde.py →  Re-exports MatterGen's LatticeVPSDE
  ├── corruption.py  →  KLDMMultiCorruption(MultiCorruption)
  ├── loss.py        →  KLDMLoss
  ├── sampling.py    →  KLDMSampler (PC loop)
  ├── predictors.py  →  TDMPredictor(Predictor)
  └── correctors.py  →  TDMLangevinCorrector

scripts/             →  Hydra entry-point + DataModule (MatterGen CrystalDataset)
```

The key structural difference: `kldm_frnct` uses a single `KLDM` orchestrator class that
holds the GNN and per-modality diffusions in a `ModuleDict`. `kldm_new` separates these
concerns into MatterGen's abstractions: the GNN is wrapped in a `ScoreModel`, the
diffusions live in a `MultiCorruption`, and the loss is a standalone callable.

---

## 3. Lattice Representation

### `kldm_frnct`: 6-parameter vector `l`

The lattice is decomposed into **3 lengths** $(a, b, c)$ and **3 angles** $(\alpha, \beta, \gamma)$,
each normalised via learned transforms:

$$
\ell_{\text{abc}} = \frac{\log(a, b, c) - \mu_n}{\sigma_n} \qquad
\ell_{\text{angles}} = \frac{\tan(\alpha - \pi/2) - \mu}{\sigma}
$$

where $\mu_n, \sigma_n$ are **per-atom-count** statistics for lengths. The final
6-dimensional vector is:

```python
# data/transforms.py
l = torch.cat([log_lengths_normalised, tan_angles_normalised], dim=-1)  # (B, 6)
```

**Inversion** at sampling time requires the same transform objects to undo the normalisation:

```python
# lit/module.py — structures_from_tensors
log_abc, tan_angles = li[:3], li[3:]
a, b, c = transform_lengths.invert_one(log_abc, n)
alpha, beta, gamma = transform_angles.invert_one(tan_angles)
lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
```

### `kldm_new`: 3×3 cell matrix `cell`

The lattice is stored directly as a **3×3 matrix** — the three row vectors of the unit cell:

$$
\text{cell} = \begin{pmatrix} \mathbf{a} \\ \mathbf{b} \\ \mathbf{c} \end{pmatrix} \in \mathbb{R}^{3 \times 3}
$$

This is the native representation in MatterGen's `ChemGraph`:

```python
# CrystalDataset stores cell as (B, 3, 3) numpy arrays
# ChemGraph.cell is a (1, 3, 3) tensor per structure, batched to (B, 3, 3)
```

**Implications:**

| | 6-param `l` | 3×3 `cell` |
|-|-------------|-------------|
| Normalisation | Required (per-atom-count stats) | Not needed |
| Invertibility | Requires transforms at sampling | Direct: `Lattice(cell[i])` |
| Degrees of freedom | 6 (minimal) | 9 (redundant — orientation info) |
| MatterGen compat | ✗ (custom) | ✓ (native) |
| Diffusion | VP-SDE on $\mathbb{R}^6$ | VP-SDE on $\mathbb{R}^{3 \times 3}$ |

The 3×3 representation is over-parameterised (rotational degrees of freedom), but
MatterGen's `LatticeVPSDE` handles this with **symmetric noise** to preserve the
physical symmetry:

```python
# MatterGen's LatticeVPSDE.sample_marginal
noise = make_noise_symmetric_preserve_variance(noise)
```

---

## 4. Coordinate System & Wrapping

### `kldm_frnct`: Scaled coordinates on $[0, 2\pi)$

Positions are **fractional** internally but are **scaled by $2\pi$** before diffusion:

```python
# model/tdm.py — TDM.__init__
self.scale_pos = 2.0 * math.pi   # = 2π

# model/tdm.py — training_targets
pos = self.scale_pos * wrap(pos01)  # pos01 ∈ [0,1) → pos ∈ [0, 2π)
```

Wrapping uses `atan2`:

```python
# nn/utils.py
def wrap(x, x_range=(2.0 * torch.pi)):
    return torch.arctan2(torch.sin(x_range * x), torch.cos(x_range * x)) / x_range
```

The network receives **scaled-down** latents: `v_t / scale_pos` and `pos_t / scale_pos`,
so the GNN sees values in $[-0.5, 0.5]$.

### `kldm_new`: Fractional coordinates on $[0, 1)$

Positions are kept in the natural fractional range:

```python
# diffusion/tdm.py — KineticLangevinSDE.__init__
self.scale_pos = 1.0   # fractional coords

# diffusion/tdm.py
def _wrap(x, period=1.0):
    return torch.remainder(x, period)
```

No scaling is applied. The GNN sees **raw fractional coordinates** $\in [0, 1)$.

Wrapping uses `torch.remainder` instead of `atan2` — simpler, cheaper, but not
differentiable at the boundary (not needed in practice since we never backprop through wrapping).

**Edge-level minimum-image convention:** The GNN applies the minimum image convention
to displacement vectors before embedding:

```python
# nn/arch.py — CSPVCellNet.forward
pos_diff = pos[edge_node_index[1]] - pos[edge_node_index[0]]
pos_diff = pos_diff - pos_diff.round()  # minimum image convention
```

This maps each component to $[-0.5, 0.5]$ — essential for fractional coordinates
where atom 0 at $x{=}0.01$ and atom 1 at $x{=}0.99$ are only 0.02 apart, not 0.98.

> **`kldm_frnct` does NOT apply minimum-image convention.** This is because it uses
> a fully-connected graph where `pos_diff` is just an input feature (not a physical
> distance), and the $2\pi$ scaling + `atan2` wrapping partially handles periodicity.

---

## 5. SDE Framework

### `kldm_frnct`: Custom standalone classes

Each modality has its own hand-rolled diffusion class:

```python
# model/continuous.py — custom SDE hierarchy
class Schedule(ABC, nn.Module): ...
class LinearSchedule(Schedule): ...   # β(t) = β_min + (β_max - β_min) t
class SDE(ABC, nn.Module): ...
class VPSDE(SDE): ...                 # loc-scale parameterization
class ContinuousDiffusion(BaseContinuousDiffusion): ...

# model/tdm.py — standalone TDM
class TDM(nn.Module): ...             # kinetic Langevin, not an SDE subclass

# model/discrete.py
class DiscreteDiffusion(nn.Module): ... # masking diffusion for atom types
```

These are **independent implementations** — `TDM` does **not** subclass `SDE`; it has
its own `training_targets()`, `reverse_step_em()`, `reverse_step_predictor()`, etc.

### `kldm_new`: Inherits MatterGen's `SDE` interface

```python
# diffusion/tdm.py
class KineticLangevinSDE(SDE):  # MatterGen's SDE base
    def sde(self, x, t, ...) -> (drift, diffusion): ...
    def marginal_prob(self, x, t, ...) -> (mean, std): ...
    def sample_marginal(self, x, t, ...) -> x_t: ...
    def prior_sampling(self, shape, ...) -> z: ...
    def prior_logp(self, z, ...) -> logp: ...
    # + custom methods for displacement marginal, training target, reverse steps

# diffusion/lattice_sde.py
from mattergen.common.diffusion.corruption import LatticeVPSDE  # re-exported
```

By conforming to MatterGen's `SDE` interface, `KineticLangevinSDE` plugs into:
- `MultiCorruption.sample_marginal()` for training
- `Predictor` / `Corrector` abstractions for sampling
- MatterGen's built-in `PredictorCorrector` loop (for the cell field)

---

## 6. TDM (Kinetic Langevin) — The Core Diffusion

Both implementations model the same physics:

$$
\mathrm{d}v = -v\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}W, \qquad
\mathrm{d}r = v\,\mathrm{d}t
$$

with wrapped positions $\text{pos}_t = \text{wrap}(\text{pos}_0 + r_t)$.

### Velocity marginal (identical in both)

$$
v_t \mid v_0 \sim \mathcal{N}\!\bigl(e^{-t} v_0,\; (1 - e^{-2t})\,I\bigr)
$$

```python
# kldm_frnct — TDM
def _mu_v_t_coeff(self, t):   return torch.exp(-t)
def _sigma_v_t(self, t):      return torch.sqrt(1.0 - torch.exp(-2.0 * t))
```

```python
# kldm_new — KineticLangevinSDE.marginal_prob
mean = torch.exp(-t) * x
std  = torch.sqrt(1.0 - torch.exp(-2.0 * t))
```

### Displacement marginal — **key mathematical difference**

| | `kldm_frnct` | `kldm_new` |
|--|-------------|------------|
| $\mu_r$ | $\frac{1 - e^{-t}}{1 + e^{-t}}\,(v_t + v_0)$ | $(1 - e^{-t})\,v_0$ |
| Uses noisy $v_t$? | **Yes** | **No** |
| $\sigma_r^2$ | $2t + \frac{8}{1+e^t} - 4$ | $2t - 3 + 4e^{-t} - e^{-2t}$ |

#### `kldm_frnct` — displacement depends on noisy velocity

```python
# model/tdm.py
def _mu_r_t(self, t, v, v_t):
    prefactor = self._prefactor_t(t)      # (1-e^-t) / (1+e^-t)
    return prefactor * (v_t + v)          # uses NOISY v_t and CLEAN v_0

def _sigma_r_t(self, t, eps=1e-6):
    return torch.sqrt(2.0 * t + 8.0 / (1.0 + torch.exp(t)) - 4.0 + eps)
```

The formula $\mu_r = \frac{1-e^{-t}}{1+e^{-t}} (v_t + v_0)$ is a **conditional** mean
given $v_0$ **and** $v_t$, yielding a tighter distribution (lower variance $\sigma_r$).

#### `kldm_new` — displacement depends only on clean velocity

```python
# diffusion/tdm.py
def _displacement_marginal(self, v0, t, batch_idx=None):
    mu_r   = (1.0 - torch.exp(-t)) * v0
    sigma_r_sq = 2.0 * t - 3.0 + 4.0 * torch.exp(-t) - torch.exp(-2.0 * t)
    sigma_r = torch.sqrt(torch.clamp(sigma_r_sq, min=1e-12))
    return mu_r, sigma_r
```

This is the **marginal** (integrating out intermediate velocities), using only the clean $v_0$.
The variance $\sigma_r^2 = 2t - 3 + 4e^{-t} - e^{-2t}$ is larger than `kldm_frnct`'s, since
it accounts for velocity uncertainty.

> **Mathematically**, both are correct derivations of the KL diffusion marginals. They
> differ in **what they condition on**: `kldm_frnct` conditions on both $v_0$ and $v_t$ (the sampled
> noisy velocity), while `kldm_new` conditions only on $v_0$.

### Training target

Both use the **score of the wrapped-normal distribution** as the training target:

$$
\nabla_r \log p_{\text{WN}}(r \mid \mu_r, \sigma_r; T)
= \sum_{n=-N}^{N} w_n \cdot \frac{-(r - \mu_r - nT)}{\sigma_r^2}
$$

where $w_n = \text{softmax}_n\!\bigl(-\frac{(r - \mu_r - nT)^2}{2\sigma_r^2}\bigr)$.

Both use **simplified parameterization** to rescale the target. The rescaling factors differ
because the underlying marginals differ:

**`kldm_frnct`** — rescale by `prefactor * sqrt(sigma_norm)`:

```python
# target_pos_t is the WN score × prefactor
target = target_pos_t / prefactor_t / sqrt(sigma_norm_t)
```

**`kldm_new`** — rescale by `(sigma_r / vel_std) / sqrt(sigma_norm)`:

```python
prefactor = sigma_r / vel_std.clamp(min=1e-8)
target = score_wn * prefactor / sqrt(sigma_norm)
```

### Score construction at sampling time

At sampling time, the model output must be converted back to a **velocity score**.

**`kldm_frnct`** — `_construct_score_v_t`:

```python
# model/tdm.py
def _construct_score_v_t(self, t, v_t, pred_v_t, index):
    # Velocity part: -v_t / σ²_v
    term_v = -v_t / self._sigma_v_t(t)[index] ** 2
    # WN part: model_output × prefactor × √σ_norm
    prefactor = self._prefactor_t(t)[index]
    sigma_norm_t = torch.sqrt(self._sigma_norm_t(t))[index]
    term_wn = pred_v_t * prefactor * sigma_norm_t
    return term_v + term_wn
```

The velocity score has **two terms**: an analytic velocity-prior term and the WN score
from the network. This is because the target was only the WN part.

**`kldm_new`** — the model output **is** treated as the (rescaled) velocity score directly,
and the predictors/correctors use it as such. No separate decomposition into velocity
and WN terms is needed because the `_displacement_marginal` already marginalises out
the velocity coupling.

---

## 7. Lattice Diffusion (VP-SDE)

### `kldm_frnct`: Custom VP-SDE on $\mathbb{R}^6$

```python
class ContinuousDiffusion(BaseContinuousDiffusion):
    sde = VPSDE(LinearSchedule(beta_min=0.1, beta_max=20.0))
    dim = 6                # 3 log-lengths + 3 tan-angles
    parameterization = "eps"  # or "x0"
```

Forward marginal:
$$
\ell_t = \alpha_t \ell_0 + \sigma_t \epsilon, \quad
\alpha_t = e^{-\frac{1}{2}\int_0^t \beta(s)\,ds}, \quad
\sigma_t = \sqrt{1 - \alpha_t^2}
$$

### `kldm_new`: MatterGen's `LatticeVPSDE` on $\mathbb{R}^{3 \times 3}$

```python
from mattergen.common.diffusion.corruption import LatticeVPSDE
```

Same VP-SDE mathematics, but with two MatterGen-specific enhancements:

1. **Density-aware prior**: The stationary distribution mean is set to a cell
   corresponding to a target density (default 0.05 Å⁻³), with variance scaled
   by $n_{\text{atoms}}^{2/3}$.

2. **Symmetric noise**: Noise is made symmetric to preserve the lattice's
   physical symmetry under rotation (via `make_noise_symmetric_preserve_variance`).

```python
# MatterGen internally:
def sample_marginal(self, x, t, ...):
    limit_mean = self.get_limit_mean(x, batch)
    limit_var = self.get_limit_var(x, batch)
    noise = make_noise_symmetric_preserve_variance(torch.randn_like(x))
    ...
```

---

## 8. Multi-Modal Orchestration

### `kldm_frnct`: `KLDM` class with `ModuleDict`

```python
class KLDM(nn.Module):
    def __init__(self, net, diffusion_v, diffusion_l, diffusion_h):
        self.net = net
        self.diffusions = nn.ModuleDict({"v": diffusion_v, "l": diffusion_l, "h": diffusion_h})

    def loss_diffusion(self, t, batch):
        # 1. Compute targets for each modality
        latents, targets = self.training_targets(t, batch)
        # 2. Forward through GNN
        preds = self.net(t=t, **latents, ...)
        # 3. Per-modality losses
        losses = {key: self.diffusions[key].loss_diffusion(preds[key], targets[key], ...) for key in targets}
        return losses
```

The `KLDM` class is the **single orchestrator** — it computes targets, runs the GNN,
and gathers losses. The `LitKLDM` Lightning module does the weighted sum.

### `kldm_new`: Separate `MultiCorruption` + `Loss` + `ScoreModel`

```python
class LitKLDM(LightningModule):
    def _basic_step(self, batch):
        batch = add_velocity(batch)                 # zero-init vel
        t = self._sample_t(batch_size)

        noisy_batch = self.multi_corruption.sample_marginal(batch, t)  # corrupt ALL fields
        score_out   = self.score_model(noisy_batch, t)                 # predict scores
        total_loss  = self.loss_fn(..., batch, noisy_batch, score_out, t)
        return total_loss
```

Each concern is its own class:
- **Corruption** (`KLDMMultiCorruption`): knows how to corrupt pos/vel/cell together
- **Score model** (`KLDMScoreModel`): builds edges, runs GNN, returns predictions
- **Loss** (`KLDMLoss`): computes TDM loss + cell DSM loss

This matches MatterGen's design philosophy where `DiffusionModule.training_step` calls
`multi_corruption.sample_marginal()` → `score_model()` → `loss()` in sequence.

---

## 9. Graph Neural Network

Both use the same **CSPVLayer / CSPVCellLayer** architecture:

```
Edge model: [h_i, h_j, lattice_info, v_proj(v_j-v_i), sin_emb(pos_diff)] → MLP
Node model: [h_node, mean_agg(edges)] → MLP + residual
```

### Key differences

| Feature | `CSPVNet` | `CSPVCellNet` |
|---------|-----------|---------------|
| Lattice input | `l` (6-dim vector) | `cell_flat` (9-dim, 3×3 flattened) |
| Edge input dim | `2H + 2D + 6` | `2H + 2D + 9` |
| pos_diff handling | Raw difference (no minimum image) | `pos_diff - pos_diff.round()` |
| Graph construction | Fully-connected (precomputed) | `radius_graph` (on-the-fly) |
| Velocity output key | `"v"` | `"vel"` |
| Lattice output | `out_l` (B, 6) | `out_cell` (B, 3, 3) via reshape |

#### Graph construction

```python
# kldm_frnct — precomputed fully-connected graph in transforms
class FullyConnectedGraph(BaseTransform):
    def forward(self, data):
        n = len(data.pos)
        fc_graph = torch.ones(n, n) - torch.eye(n)
        fc_edges, _ = dense_to_sparse(fc_graph)
        data.edge_node_index = fc_edges
        return data
```

```python
# kldm_new — on-the-fly radius graph in KLDMScoreModel.forward
edge_index = radius_graph(
    pos,
    r=self.cutoff,        # default 0.5 (fractional units)
    batch=batch_idx,
    max_num_neighbors=self.max_neighbors,  # default 20
)
```

The fully-connected graph scales as $O(N^2)$ edges, limiting `kldm_frnct` to
small systems. The radius graph scales as $O(kN)$ where $k$ is the average
coordination, enabling larger systems.

#### Distance embedding

```python
# kldm_frnct — SinEmbedding operates on raw 3D vector
class SinEmbedding:
    def forward(self, x):
        # x is (E, 3), frequencies applied per component
        emb = x.unsqueeze(-1) * self.frequencies  # (E, 3, K)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)  # (E, 3K)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)  # (E, 6K)
```

```python
# kldm_new — SinEmbedding takes norm of displacement
class SinEmbedding:
    def forward(self, x):
        if x.ndim >= 2 and x.shape[-1] > 1:
            x = x.norm(dim=-1, keepdim=True)   # (E, 1) — scalar distance
        xf = x * self.freqs                     # (E, K)
        return torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)  # (E, 2K)
```

In `kldm_frnct`, the embedding is **per-component** (3 spatial dims × K frequencies → dim = 6K).
In `kldm_new`, it embeds the **scalar distance** (1 × K frequencies → dim = 2K). This is
consistent with radius-graph usage where the displacement direction is not as meaningful
in fractional space.

---

## 10. Training Pipeline

### Forward pass comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                          kldm_frnct                                 │
├─────────────────────────────────────────────────────────────────────┤
│  1.  t ~ U(1e-3, 1)              (stored as t01 ∈ (0,1])           │
│  2.  TDM.training_targets(t01*tf, pos, idx)                        │
│      → v_t, pos_t, target_v                                        │
│  3.  ContinuousDiffusion.training_targets(t, l)   → l_t, target_l  │
│  4.  CSPVNet(t, pos_t/2π, v_t/2π, h, l_t, idx, edges) → preds     │
│  5.  Σ_k  w_k · loss_k(preds[k], targets[k])                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          kldm_new                                   │
├─────────────────────────────────────────────────────────────────────┤
│  1.  t ~ U(ε, T]  where T=2.0                                      │
│  2.  batch = add_velocity(batch)          (vel_0 = 0)               │
│  3.  noisy_batch = multi_corruption.sample_marginal(batch, t)       │
│      → pos_t (wrapped frac), vel_t (Gaussian), cell_t (VP)         │
│  4.  score_out = KLDMScoreModel(noisy_batch, t)                     │
│      → builds radius_graph, runs CSPVCellNet                        │
│  5.  KLDMLoss:                                                      │
│      a. vel_target = KineticLangevinSDE.training_target(...)        │
│      b. cell_target = -(cell_t - mean) / std                        │
│      c. total = w_vel * MSE(vel) + w_cell * MSE(cell)               │
└─────────────────────────────────────────────────────────────────────┘
```

### Time sampling

| | `kldm_frnct` | `kldm_new` |
|--|-------------|------------|
| Range | $t \in (10^{-3},\, 1]$ | $t \in (10^{-3},\, 2.0]$ |
| Internal time | $t_{\text{internal}} = t_f \cdot t_{01} = 2.0 \cdot t_{01}$ | $t$ directly (no rescaling) |
| Applies to lattice | Same $t_{01}$ ∈ (0,1] (VP-SDE in [0,1]) | Same $t$ ∈ (ε, 2] but LatticeVPSDE has $T{=}1$ → clamped internally |

In `kldm_frnct`, the TDM operates in internal time $[0, t_f{=}2]$ while the VP-SDE
uses $t_{01} \in [0, 1]$. The mapping is $t_{\text{internal}} = 2 \cdot t_{01}$.

In `kldm_new`, both SDEs share the same time $t \in [\varepsilon, T]$ where $T = 2.0$.
The `LatticeVPSDE` (which has `T=1.0`) internally normalises $t$ accordingly via
MatterGen's infrastructure.

### Loss computation

**Velocity/position loss** — identical concept (MSE on simplified-parameterization target),
but computed differently due to the displacement marginal differences in §6.

**Lattice loss:**

```python
# kldm_frnct — eps or x0 parameterization
loss_l = F.mse_loss(pred_l, target_l)  # target is ε or l_0

# kldm_new — score_times_std (DSM) parameterization
noise = (cell_t - mean) / std
cell_target = -noise                    # = score × std
loss_cell = ((pred_cell - cell_target) ** 2).mean()
```

---

## 11. Sampling (Reverse-Time)

### `kldm_frnct`: Monolithic `KLDM.sample()`

```python
def sample(self, batch, method="pc", n_steps=1000, ts=1.0, tf=1e-3):
    ts = torch.linspace(ts, tf, n_steps + 1)           # time schedule
    pos_t, v_t, h_t, l_t = self.sample_prior(batch)    # prior sampling

    for i in range(n_steps):
        if method == "em":
            pos_t, v_t, h_t, l_t = self.reverse_step_em(...)
        elif method == "pc":
            pos_t, v_t, h_t, l_t = self.reverse_step_pc(...)

    return self.final_step(...)
```

The PC method calls:
1. **Corrector** (`n_correction_steps` Langevin steps on velocity → position)
2. **Predictor** (DDIM-like step for velocity + position, then VP-SDE step for lattice)

All modality-specific reverse steps are methods on `TDM` / `ContinuousDiffusion`.

### `kldm_new`: `KLDMSampler` with Predictor/Corrector objects

```python
class KLDMSampler:
    def __init__(self, multi_corruption, score_fn, N, ...):
        self.tdm_predictor = TDMPredictor(corruption=multi_corruption.pos_sde)
        self.tdm_corrector = TDMLangevinCorrector(corruption=multi_corruption.pos_sde)

    def sample(self, conditioning_data):
        # Prior sampling
        vel = pos_sde.prior_sampling((n_atoms, 3))
        pos = torch.rand(n_atoms, 3) * scale_pos
        cell = cell_sde.prior_sampling((B, 3, 3))

        for i in range(N):
            score_batch = self.score_fn(batch, t)

            # Corrector (velocity + cell)
            vel = self.tdm_corrector.step_given_score(x=vel, score=vel_score, ...)
            cell = self._cell_langevin_step(cell, cell_score, ...)

            # Predictor (velocity → position + cell)
            vel = self.tdm_predictor.update_given_score(x=vel, score=vel_score, ...)
            pos = wrap(pos - dt * vel)
            cell = self._cell_ancestral_step(cell, cell_score, ...)

        return batch
```

The key structural difference: `kldm_new` uses **Predictor** and **Corrector** objects
that conform to MatterGen's interfaces. The cell predictor step is implemented manually
as `_cell_ancestral_step` because MatterGen's built-in `AncestralSamplingPredictor`
assumes 1-D or 2-D tensors and fails on the (B, 3, 3) cell tensor.

### Velocity predictor step comparison

**`kldm_frnct`** — ratio-based update:

```python
# model/tdm.py — reverse_step_predictor
r = self._mu_v_t_coeff(t + dt) / self._mu_v_t_coeff(t)    # = exp(dt)
sigma_v_t = self._sigma_v_t(t)
prefactor = (r * sigma_v_t - self._sigma_v_t(t + dt)) * sigma_v_t
v_t = r[node_index] * v_t + prefactor[node_index] * score_v_t
pos_t = wrap(pos_t + dt * v_t)
```

**`kldm_new`** — DDIM-like update:

```python
# diffusion/tdm.py — reverse_step_pc_predictor
alpha_t, alpha_s = exp(-t), exp(-s)
sigma_t = sqrt(1 - alpha_t²)
sigma_s = sqrt(1 - alpha_s²)
v0_hat = (v - sigma_t * score) / alpha_t         # "predicted v_0"
v_new  = alpha_s * v0_hat + sigma_s * score       # interpolate
pos_new = wrap(pos - dt * v_new)
```

The DDIM formulation explicitly reconstructs $\hat{v}_0$ (the predicted clean velocity)
and interpolates between it and the score to arrive at the next step. This is equivalent
to the ratio-based update but expressed in the "x0-prediction" language common in the
diffusion literature.

---

## 12. Data Pipeline

### `kldm_frnct`: `.pt` files + PyG transforms

```python
# data/dataset.py
class Dataset(torchdata.Dataset):
    def __init__(self, path, transform=None):
        self.data = torch.load(path)     # list of dicts or PyG Data objects
        self.transform = transform

    def __getitem__(self, idx):
        data = self.data[idx]
        data = Data(pos=..., h=..., lengths=..., angles=...)
        return self.transform(data)

# data/datamodule.py
class DataModule(LightningDataModule):
    def __init__(self, transform, train_path, val_path, ...):
        self.train_dataset = Dataset(path=train_path, transform=transform)
```

Transforms include:
- `FullyConnectedGraph` — build $O(N^2)$ edge index
- `ContinuousIntervalLengths` — normalise lattice lengths
- `ContinuousIntervalAngles` — normalise lattice angles
- `ConcatFeatures` — concatenate to 6-dim `l`
- `OneHot` — encode atom types

### `kldm_new`: MatterGen's numpy-cached `CrystalDataset`

```python
# scripts/_datamodule.py
class KLDMNewDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.train_dataset = CrystalDataset.from_cache_path(str(data_path / "train"))
        ...

    def _make_loader(self, dataset, batch_size, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate,   # MatterGen's collate → ChemGraphBatch
            ...
        )
```

MatterGen's `CrystalDataset`:
- Stores pre-processed numpy arrays: `pos`, `cell (3×3)`, `atomic_numbers`, `num_atoms`
- `__getitem__` returns a `ChemGraph` (subclass of PyG `Data`)
- `collate` batches `ChemGraph` objects into `ChemGraphBatch` (which satisfies `BatchedData`)
- **No transforms needed** — the cell is already a 3×3 matrix, atom types are raw integers

> **Data conversion**: To use a dataset originally prepared for `kldm_frnct`, you need to
> convert `.pt` files to MatterGen's numpy cache format. The `CrystalDataset` expects
> a directory structure with `pos.npy`, `cell.npy`, `atomic_numbers.npy`, `num_atoms.npy`.

---

## 13. HPC / Deployment

### `kldm_frnct`

- `scripts/train.py` — Hydra entry-point with custom config
- `configs/train_csp_mp_20.yaml` — YAML config specifying all components
- Instantiation: Hydra resolves full component tree (model, optimizer, LitKLDM, …)

### `kldm_new`

- `scripts/train.py` — Hydra entry-point (`@hydra.main` with `config_path` pointing to `kldm_new/configs/`)
- `configs/train_mp_20.yaml` — Full config with MatterGen-compatible components
- `configs/experiment/debug.yaml` — Override config for quick local testing

```bash
# Default training
python -m kldm_new.scripts.train

# Debug (small model, CPU, no logging)
python -m kldm_new.scripts.train +experiment=debug

# HPC / multi-GPU
python -m kldm_new.scripts.train trainer.devices=4 trainer.strategy=ddp
```

Both use PyTorch Lightning + Hydra, so the deployment story is very similar.
The main difference is that `kldm_new`'s DataModule wraps `CrystalDataset` (numpy cache)
instead of `.pt` files, which is faster for large datasets.

---

## 14. Quick Reference Table

| Component | `kldm_frnct` | `kldm_new` | Notes |
|-----------|-------------|------------|-------|
| **Import prefix** | `src_kldm` | `kldm_new` | Set via `.pth` file |
| **Lattice repr** | 6D `l` (lengths+angles) | 3×3 `cell` | Over-parameterised but simpler |
| **Coord range** | $[0, 2\pi)$ | $[0, 1)$ | Fractional native |
| **Wrapping fn** | `atan2(sin, cos)` | `torch.remainder` | atan2 is differentiable |
| **Min image convention** | No | `pos_diff - pos_diff.round()` | Critical for frac coords |
| **Graph type** | Fully-connected | Radius ($r{=}0.5$, $k_{\max}{=}20$) | $O(N^2)$ vs $O(kN)$ |
| **TDM class** | `TDM(nn.Module)` | `KineticLangevinSDE(SDE)` | MatterGen inheritance |
| **Lattice SDE** | `ContinuousDiffusion` | `LatticeVPSDE` (MatterGen) | Symmetric noise + density prior |
| **$\mu_r$ formula** | $\frac{1-e^{-t}}{1+e^{-t}}(v_t+v_0)$ | $(1-e^{-t})v_0$ | Conditional vs marginal |
| **$\sigma_r^2$ formula** | $2t+\frac{8}{1+e^t}-4$ | $2t-3+4e^{-t}-e^{-2t}$ | Different conditioning |
| **Score construction** | 2-term (analytic + NN) | Direct (NN predicts score) | Simpler in kldm_new |
| **Orchestrator** | `KLDM(nn.Module)` | `KLDMMultiCorruption` + `KLDMLoss` | Separated concerns |
| **Score model** | `CSPVNet` inside `KLDM` | `KLDMScoreModel` (wraps `CSPVCellNet`) | + radius_graph |
| **Predictor** | Methods on `TDM` / `ContinuousDiffusion` | `TDMPredictor(Predictor)` | MatterGen interface |
| **Corrector** | Methods on `TDM` / `ContinuousDiffusion` | `TDMLangevinCorrector` | MatterGen interface |
| **Sampler** | `KLDM.sample()` | `KLDMSampler` | Standalone PC loop |
| **Data format** | `.pt` (PyG Data lists) | Numpy cache (`CrystalDataset`) | Faster for large data |
| **Transforms** | Length/angle normalisation | None needed | 3×3 cell is raw |
| **Lightning module** | `LitKLDM` (custom) | `LitKLDM` (MatterGen-style) | Similar structure |
| **Atom types** | `"h"` key | `"atomic_numbers"` key | MatterGen naming |
| **Velocity key** | `"v"` | `"vel"` | |
| **Lattice key** | `"l"` | `"cell"` | |
| **SinEmbedding** | Per-component (dim=6K) | Scalar norm (dim=2K) | Different encoding |
| **FourierEmbedding** | Fixed random weights | Random weights + linear layer | Extra learnable projection |
| **Atom type diffusion** | `DiscreteDiffusion` / `AnalogBitsContinuousDiffusion` | Not implemented | CSP-only in kldm_new |
| **EMA** | `AveragedModel` | `AveragedModel` | Identical |
| **Optimizer** | `AdamW(amsgrad, foreach)` | `AdamW(amsgrad)` | Nearly identical |
