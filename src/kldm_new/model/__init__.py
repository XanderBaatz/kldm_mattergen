"""Score model wrapper for KLDM.

Wraps :class:`CSPVCellNet` into MatterGen's :class:`ScoreModel` interface
so it can be used inside :class:`DiffusionModule`.
"""

from __future__ import annotations

from mattergen.diffusion.data.batched_data import BatchedData  # noqa: TC002
from torch import Tensor, nn

from kldm_new.nn.arch import CSPVCellNet  # noqa: TC001


class KLDMScoreModel(nn.Module):
    """Score model for KLDM — wraps CSPVCellNet for the MatterGen pipeline.

    Edges are expected to be precomputed at dataset time as ``edge_node_index``
    (a ``(2, E)`` long tensor stored on each :class:`ChemGraph`).  Use the
    :class:`~kldm_new.data.transform.FullyConnectedGraph` dataset transform to
    add fully-connected intra-crystal edges during data loading.

    Parameters
    ----------
    net : CSPVCellNet
        The underlying GNN.

    """

    def __init__(self, net: CSPVCellNet) -> None:
        """Initialize the KLDMScoreModel."""
        super().__init__()
        self.net = net

    def forward(self, x: BatchedData, t: Tensor) -> BatchedData:
        """Predict scores for all fields.

        Args:
            x: Noisy batch with ``pos``, ``vel``, ``cell``, ``atomic_numbers``,
               and ``edge_node_index`` (precomputed fully-connected edges).
            t: ``(B, 1)`` diffusion time.

        Returns:
            A batch-like object with predicted ``vel`` and ``cell`` scores.

        """
        pos = x["pos"]  # (N, 3)
        vel = x["vel"]  # (N, 3)
        cell = x["cell"]  # (B, 3, 3)
        h = x["atomic_numbers"]  # (N,) — integer atom types
        batch_idx = x.get_batch_idx("pos")  # (N,)
        edge_node_index = x["edge_node_index"]  # (2, E) — precomputed at dataset time

        out = self.net(
            t=t,
            pos=pos,
            vel=vel,
            h=h,
            cell=cell,
            node_index=batch_idx,
            edge_node_index=edge_node_index,
        )

        return x.replace(**out)
