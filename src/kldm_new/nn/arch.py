"""CSPVCellNet — Crystal Structure Prediction GNN with velocity and cell (3x3).

This is a reimplementation of kldm_frnct's ``CSPVNet`` adapted for the
MatterGen framework.  The key change is that the lattice is represented as
a 3x3 cell matrix (``cell``) instead of a 6-dim parameter vector (``l``).

Architecture
------------
1. Node embedding: ``Embedding(h_dim, hidden) + cat(time_emb) → Linear``
2. Message-passing layers (:class:`CSPVCellLayer`):
   * **Edge model**: ``[h_i, h_j, cell_flat, v_proj(v_j-v_i), sin_emb(pos_diff)] → MLP``
   * **Node model**: ``[h_node, agg(edge)] → MLP`` with residual
3. Readouts:
   * ``vel``: per-atom 3D (zero-CoG), used as velocity score
   * ``cell``: per-graph 3x3, used as cell score (score_times_std)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_scatter import scatter

from kldm_new.nn import FourierEmbedding, SinEmbedding
from kldm_new.nn.utils import scatter_center


class CSPVCellLayer(nn.Module):
    """Single message-passing layer for CSPVCellNet."""

    def __init__(
        self,
        dis_emb: SinEmbedding,
        hidden_dim: int = 128,
        act_fn: nn.Module | None = None,
        ln: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the CSPVCellLayer."""
        super().__init__()
        if act_fn is None:
            act_fn = nn.SiLU()

        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim

        # Edge input: h_i + h_j + cell_flat(9) + v_proj(dis_dim) + pos_emb(dis_dim)
        #           = 2*hidden + 9 + 2*dis_dim
        input_dim = hidden_dim * 2 + 2 * self.dis_dim + 9

        self.v_proj = nn.Linear(3, dis_emb.dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(  # noqa: PLR0913
        self,
        pos_diff: Tensor,
        v: Tensor,
        node_features: Tensor,
        cell_flat: Tensor,
        edge_node_index: Tensor,
        edge_graph_index: Tensor,
    ) -> Tensor:
        """Compute edge features."""
        hi = node_features[edge_node_index[0]]
        hj = node_features[edge_node_index[1]]
        vi, vj = v[edge_node_index[0]], v[edge_node_index[1]]
        vij = self.v_proj(vj - vi)

        pos_emb = self.dis_emb(pos_diff)
        cell_edge = cell_flat[edge_graph_index]  # (E, 9)

        edges_input = torch.cat([hi, hj, cell_edge, vij, pos_emb], dim=1)
        return self.edge_mlp(edges_input)

    def node_model(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_node_index: Tensor,
    ) -> Tensor:
        """Compute node features with mean aggregation."""
        agg = scatter(
            edge_features,
            edge_node_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        # Isolated nodes (0 edges) produce NaN from 0/0 on some builds; replace with 0
        agg = torch.nan_to_num(agg, nan=0.0)
        return self.node_mlp(torch.cat([node_features, agg], dim=1))

    def forward(  # noqa: PLR0913
        self,
        pos_diff: Tensor,
        v: Tensor,
        node_features: Tensor,
        cell_flat: Tensor,
        edge_node_index: Tensor,
        edge_graph_index: Tensor,
    ) -> Tensor:
        """Forward pass with residual connection."""
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(
            pos_diff=pos_diff,
            v=v,
            node_features=node_features,
            cell_flat=cell_flat,
            edge_node_index=edge_node_index,
            edge_graph_index=edge_graph_index,
        )
        node_output = self.node_model(
            node_features=node_features,
            edge_features=edge_features,
            edge_node_index=edge_node_index,
        )
        return node_input + node_output


class CSPVCellNet(nn.Module):
    """Crystal Structure Prediction GNN with velocity and cell 3x3.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension throughout the network.
    time_dim : int
        Dimension of the time embedding.
    num_layers : int
        Number of message-passing layers.
    h_dim : int
        Number of atom types (for discrete embedding) or continuous dim.
    num_freqs : int
        Number of sinusoidal frequencies for distance embedding.
    ln : bool
        Use layer normalization.
    smooth : bool
        If True, use a linear embedding for continuous atom features
        instead of discrete ``Embedding``.
    pred_vel : bool
        Output a velocity prediction (3D per atom).
    pred_cell : bool
        Output a cell prediction (3x3 per graph).
    pred_h : bool
        Output atom-type logits.
    zero_cog : bool
        Enforce zero centre-of-gravity on velocity output.

    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_dim: int = 128,
        time_dim: int = 128,
        num_layers: int = 4,
        h_dim: int = 100,
        num_freqs: int = 10,
        ln: bool = True,  # noqa: FBT001, FBT002
        smooth: bool = False,  # noqa: FBT001, FBT002
        pred_vel: bool = True,  # noqa: FBT001, FBT002
        pred_cell: bool = True,  # noqa: FBT001, FBT002
        pred_h: bool = False,  # noqa: FBT001, FBT002
        zero_cog: bool = True,  # noqa: FBT001, FBT002
        time_emb: nn.Module | None = None,
    ) -> None:
        """Initialize the CSPVCellNet."""
        super().__init__()
        self.act_fn = nn.SiLU()

        # Node embedding
        if smooth:
            self.node_embedding = nn.Linear(h_dim, hidden_dim, bias=False)
        else:
            self.node_embedding = nn.Embedding(h_dim + 1, hidden_dim)

        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim, hidden_dim)

        self.dis_emb = SinEmbedding(n_frequencies=num_freqs)

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)
        self.time_emb = time_emb

        # Message-passing layers
        self.layers = nn.ModuleList([CSPVCellLayer(self.dis_emb, hidden_dim=hidden_dim, act_fn=self.act_fn, ln=ln) for _ in range(num_layers)])

        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Readout heads
        if pred_vel:
            self.out_vel = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act_fn,
                nn.Linear(hidden_dim, 3, bias=False),
            )

        if pred_cell:
            self.out_cell = nn.Linear(hidden_dim, 9, bias=False)  # 3x3 flattened

        if pred_h:
            self.out_h = nn.Linear(hidden_dim, h_dim)

        self.ln = ln
        self.smooth = smooth
        self.pred_vel = pred_vel
        self.pred_cell = pred_cell
        self.pred_h = pred_h
        self.zero_cog = zero_cog

    def forward(  # noqa: PLR0913
        self,
        t: Tensor,
        pos: Tensor,
        vel: Tensor,
        h: Tensor,
        cell: Tensor,
        node_index: Tensor,
        edge_node_index: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            t: ``(B, 1)`` diffusion time.
            pos: ``(N, 3)`` fractional coordinates.
            vel: ``(N, 3)`` velocities.
            h: ``(N,)`` atom types (long) or ``(N, h_dim)`` continuous.
            cell: ``(B, 3, 3)`` lattice cell matrix.
            node_index: ``(N,)`` graph membership for each atom.
            edge_node_index: ``(2, E)`` edge indices.

        Returns:
            Dictionary with optional keys ``"vel"``, ``"cell"``, ``"h"``.

        """
        # Time embedding
        t_emb = self.time_emb(t)  # (B, time_dim)
        t_per_atom = t_emb[node_index]  # (N, time_dim)

        # Node features
        node_features = self.node_embedding(h)  # (N, hidden_dim)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # Edge metadata — wrap to minimum image on torus [0,1)
        pos_diff = pos[edge_node_index[1]] - pos[edge_node_index[0]]  # (E, 3)
        pos_diff = pos_diff - pos_diff.round()  # minimum image convention
        edge_graph_index = node_index[edge_node_index[0]]  # (E,)
        cell_flat = cell.reshape(-1, 9)  # (B, 9)

        # Message passing
        for layer in self.layers:
            node_features = layer(
                pos_diff=pos_diff,
                v=vel,
                node_features=node_features,
                cell_flat=cell_flat,
                edge_node_index=edge_node_index,
                edge_graph_index=edge_graph_index,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        out: dict[str, Tensor] = {}

        if self.pred_vel:
            out_vel = self.out_vel(node_features)
            if self.zero_cog:
                out_vel = scatter_center(out_vel, index=node_index)
            out["vel"] = out_vel

        if self.pred_cell:
            graph_features = scatter(node_features, node_index, dim=0, reduce="mean")
            out_cell = self.out_cell(graph_features)  # (B, 9)
            out["cell"] = out_cell.view(-1, 3, 3)

        if self.pred_h:
            out["h"] = self.out_h(node_features)

        return out
