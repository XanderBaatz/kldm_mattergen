from mattergen.common.data.chemgraph import ChemGraph, ChemGraphBatch  # noqa: TC002
from mattergen.diffusion.score_models.base import ScoreModel
from torch import Tensor  # noqa: TC002

from kldm_plus.nn.net import CSPVNet  # noqa: TC001


class KineticScoreModel(ScoreModel):
    """Score model for KLDM. Uses CSPVCellNet for the MatterGen pipeline."""

    def __init__(
        self,
        net: CSPVNet,  # GNN
    ) -> None:
        """Construct a GNN object."""
        super().__init__()
        self.net = net

    def forward(
        self,
        x: ChemGraph | ChemGraphBatch,  # tuple containing a noisy ChemGraph (or ChemGraphBatch)
        t: Tensor,  # timestep per crystal
    ) -> ChemGraph | ChemGraphBatch:
        """Predict scores for all fields.

        Args:
            x (ChemGraph | ChemGraphBatch): noisy batch
            t (Tensor): diffusion timestep per crystal

        Returns:
            ChemGraph | ChemGraphBatch: A batch-like object with predict vel and cell scores

        """
        # Fetch data from ChemGraph or ChemGraphBatch
        pos = x["pos"]  # (N, 3) - fractional coordinates
        vel = x["vel"]  # (N, 3) - velocity
        cell = x["cell"]  # (B, 3, 3) - lattice
        num_atoms = x["atomic_numbers"]  # (N,) - integer atom types
        batch_idx = x.get_batch_idx("pos")  # (N,)
        edge_node_index = x["edge_node_index"]  # (2, E) — precomputed at dataset time

        # Pass to GNN
        # CSPVNet uses 'lattice' (not 'cell') as the parameter name.
        output = self.net(
            t=t,
            pos=pos,
            vel=vel,
            atom=num_atoms,  # atom types
            lattice=cell,
            node_index=batch_idx,
            edge_node_index=edge_node_index,
        )

        # The network predicts the kinetic target η in the "vel" output slot.
        # The mattergen loss (KineticFieldLoss) and score_fn both look up the
        # prediction under the "pos" key of the model-output ChemGraph.
        # Rename here so the rest of the pipeline can use the standard protocol.
        pos_pred = output.pop("vel", pos)  # η prediction; fallback to input pos
        return x.replace(pos=pos_pred, **output)
