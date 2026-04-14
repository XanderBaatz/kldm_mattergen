"""Score model wrapper for KLDM.

Wraps :class:`CSPVCellNet` into MatterGen's :class:`ScoreModel` interface
so it can be used inside :class:`DiffusionModule`.
"""

from __future__ import annotations

from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch import Tensor, nn
from torch_geometric.nn import radius_graph

from kldm_new.nn.arch import CSPVCellNet  # noqa: TC001


class KLDMScoreModel(nn.Module):
    """Score model for KLDM — wraps CSPVCellNet for the MatterGen pipeline.

    This module builds edges on-the-fly via a radius graph and delegates
    to :class:`CSPVCellNet` for the forward pass.

    Parameters
    ----------
    net : CSPVCellNet
        The underlying GNN.
    cutoff : float
        Radius cutoff for edge construction (in fractional coordinate units).
    max_neighbors : int
        Maximum number of neighbors per atom.

    """

    def __init__(
        self,
        net: CSPVCellNet,
        cutoff: float = 0.5,
        max_neighbors: int = 20,
    ) -> None:
        """Initialize the KLDMScoreModel."""
        super().__init__()
        self.net = net
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors

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

        # Build edges via radius graph on fractional coordinates
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch_idx,
            max_num_neighbors=self.max_neighbors,
        )

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
