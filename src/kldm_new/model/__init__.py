"""Score model wrapper for KLDM.

Wraps :class:`CSPVCellNet` into MatterGen's :class:`ScoreModel` interface
so it can be used inside :class:`DiffusionModule`.
"""

from __future__ import annotations

import torch
from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch import Tensor, nn
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse

from kldm_new.nn.arch import CSPVCellNet  # noqa: TC001


class KLDMScoreModel(nn.Module):
    """Score model for KLDM — wraps CSPVCellNet for the MatterGen pipeline.

    Supports two edge-construction strategies, selectable via *graph_mode*:

    ``"radius"`` (default)
        Build edges on-the-fly with a radius graph over fractional
        coordinates.  Scales to large crystals; may produce isolated atoms
        if the cutoff is too small.

    ``"full"``
        Build a fully-connected graph (every atom is connected to every
        other atom in the same crystal), matching the original
        ``kldm_frnct`` implementation.  Quadratic in the number of atoms
        per crystal — only practical for small unit cells (≲50 atoms).

    Parameters
    ----------
    net : CSPVCellNet
        The underlying GNN.
    cutoff : float
        Radius cutoff used when *graph_mode* = ``"radius"``.
    max_neighbors : int
        Maximum neighbours per atom used when *graph_mode* = ``"radius"``.
    graph_mode : str
        ``"radius"`` or ``"full"``.

    """

    def __init__(
        self,
        net: CSPVCellNet,
        cutoff: float = 0.5,
        max_neighbors: int = 20,
        graph_mode: str = "radius",
    ) -> None:
        """Initialize the KLDMScoreModel."""
        super().__init__()
        self.net = net
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        if graph_mode not in {"radius", "full"}:
            msg = f"graph_mode must be 'radius' or 'full', got {graph_mode!r}"
            raise ValueError(msg)
        self.graph_mode = graph_mode

    def _build_edges(self, pos: Tensor, batch_idx: Tensor) -> Tensor:
        """Build edge index according to *graph_mode*."""
        if self.graph_mode == "radius":
            return radius_graph(
                pos,
                r=self.cutoff,
                batch=batch_idx,
                max_num_neighbors=self.max_neighbors,
            )
        # "full": one fully-connected intra-crystal graph per structure
        n = pos.shape[0]
        # Block-diagonal adjacency: connect i↔j iff batch_idx[i]==batch_idx[j] and i≠j
        same_graph = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # (N, N)
        no_self = ~torch.eye(n, dtype=torch.bool, device=pos.device)
        adj = same_graph & no_self
        edge_index, _ = dense_to_sparse(adj.float())
        return edge_index

    def forward(self, x: BatchedData, t: Tensor) -> BatchedData:
        """Predict scores for all fields.

        Args:
            x: Noisy batch with ``pos``, ``vel``, ``cell``, ``atomic_numbers``.
            t: ``(B, 1)`` diffusion time.

        Returns:
            A batch-like object with predicted ``vel`` and ``cell`` scores.

        """
        pos = x["pos"]  # (N, 3)
        vel = x["vel"]  # (N, 3)
        cell = x["cell"]  # (B, 3, 3)
        h = x["atomic_numbers"]  # (N,) — integer atom types
        batch_idx = x.get_batch_idx("pos")  # (N,)

        edge_index = self._build_edges(pos, batch_idx)

        out = self.net(
            t=t,
            pos=pos,
            vel=vel,
            h=h,
            cell=cell,
            node_index=batch_idx,
            edge_node_index=edge_index,
        )

        return x.replace(**out)
