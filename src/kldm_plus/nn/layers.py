import torch
from torch import Tensor, nn
from torch_scatter import scatter

from kldm_plus.nn.embedding import SinEmbedding  # noqa: TC001


class CSPVLayer(nn.Module):
    """Message-passing layer for crystal structure prediction with velocities (CSPV).

    Combines node features, relative positions, velocities, and lattice information
    to perform equivariant graph neural network updates.
    """

    def __init__(
        self,
        dis_emb: SinEmbedding,
        hidden_dim: int = 128,
        act_fn: nn.Module | None = None,
        ln: bool = True,  # noqa: FBT001, FBT002 layer normalization
        lattice_dim: int = 9,
    ) -> None:
        """Initialize the CSPV layer.

        Args:
            dis_emb: Distance embedding module.
            hidden_dim: Hidden feature dimension for internal MLPs.
            act_fn: Activation module used in MLP blocks. Defaults to SiLU.
            ln: Whether to apply layer normalization to node features.
            lattice_dim: Flattened dimension of the lattice feature passed to each
                edge.  Use 9 for a 3x3 cell matrix (mattergen default) or 6 for
                the KLDM 6D lengths-and-angles representation.

        """
        super().__init__()

        if lattice_dim not in (6, 9):
            msg = f"lattice_dim must be 6 (6D KLDM) or 9 (3x3 matrix), got {lattice_dim}"
            raise ValueError(msg)

        if act_fn is None:
            act_fn = nn.SiLU()

        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim

        input_dim = hidden_dim * 2 + 2 * self.dis_dim + lattice_dim

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

        # Layer normalization
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        pos_diff: Tensor,
        vel: Tensor,
        node_features: Tensor,
        lattices: Tensor,
        edge_indices: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Compute edge features from node features, velocities, and lattice information.

        Args:
            pos_diff: Relative position vectors for each edge.
            vel: Node-wise vector features or velocities.
            node_features: Node feature matrix.
            lattices: Graph-wise lattice tensors.
            edge_indices: Tuple of (edge_node_index, edge_graph_index).

        Returns:
            Edge feature tensor after MLP transformation.

        """
        edge_node_index, edge_graph_index = edge_indices
        hi, hj = node_features[edge_node_index[0]], node_features[edge_node_index[1]]
        vi, vj = vel[edge_node_index[0]], vel[edge_node_index[1]]
        vij = self.v_proj(vj - vi)

        pos_diff = self.dis_emb(pos_diff)
        cell_edge = lattices[edge_graph_index]  # (E, 9)

        edges_input = torch.cat([hi, hj, cell_edge, vij, pos_diff], dim=1)

        return self.edge_mlp(edges_input)

    def node_model(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_node_index: Tensor,
    ) -> Tensor:
        """Aggregate edge features into node features and update node states.

        Args:
            node_features: Node feature matrix.
            edge_features: Edge feature matrix.
            edge_node_index: Edge-to-node index tensor.

        Returns:
            Updated node feature tensor after aggregation and MLP transformation.

        """
        agg = scatter(
            src=edge_features,
            index=edge_node_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)

        return self.node_mlp(agg)

    def forward(  # noqa: PLR0913
        self,
        pos_diff: Tensor,
        vel: Tensor,
        node_features: Tensor,
        lattice: Tensor,
        edge_node_index: Tensor,
        edge_graph_index: Tensor,
    ) -> Tensor:
        """Run one CSPV message-passing update step.

        Args:
            pos_diff: Relative position vectors for edges.
            vel: Node-wise vector features or velocities (?).
            node_features: Node feature matrix.
            lattice: Graph-wise lattice tensors.
            edge_node_index: Edge-to-node index tensor.
            edge_graph_index: Edge-to-graph index tensor.

        Returns:
            Updated node feature tensor with residual connection.

        """
        node_input = node_features

        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(
            pos_diff=pos_diff,
            vel=vel,
            node_features=node_features,
            lattices=lattice,
            edge_indices=(edge_node_index, edge_graph_index),
        )

        node_output = self.node_model(
            node_features=node_features,
            edge_features=edge_features,
            edge_node_index=edge_node_index,
        )

        return node_input + node_output
