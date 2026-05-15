# cspell:words CSPV

import torch
from torch import Tensor, nn
from torch_scatter import scatter

from kldm_plus.nn.embedding import FourierEmbedding, SinEmbedding
from kldm_plus.nn.layers import CSPVLayer
from kldm_plus.nn.utils import scatter_center


class CSPVNet(nn.Module):
    """Crystal-structure message-passing network for joint property prediction.

    The network embeds atom features and time, performs CSPV message passing,
    and predicts per-atom velocities, per-graph cell updates, and optional atom
    outputs depending on configured prediction heads.
    """

    def __init__(  # noqa: PLR0913
        self,
        hidden_dim: int = 128,
        time_dim: int = 128,
        num_layers: int = 4,
        atom_dim: int = 100,
        num_freqs: int = 10,
        ln: bool = True,  # noqa: FBT001, FBT002
        smooth: bool = False,  # noqa: FBT001, FBT002
        pred_atom: bool = False,  # noqa: FBT001, FBT002
        pred_vel: bool = True,  # noqa: FBT001, FBT002
        pred_cell: bool = True,  # noqa: FBT001, FBT002
        zero_cog: bool = True,  # noqa: FBT001, FBT002
        time_emb: nn.Module | None = None,
        lattice_dim: int = 9,
    ) -> None:
        """Initialize CSPVNet.

        Args:
            hidden_dim: Dimensionality of hidden node features.
            time_dim: Dimensionality of the time embedding.
            num_layers: Number of CSPV message-passing layers.
            atom_dim: Number of atom types (or input atom feature size when smooth=True).
            num_freqs: Number of frequencies for the sinusoidal distance embedding.
            ln: Whether to apply layer normalization.
            smooth: If True, use a linear projection for atom features instead of an embedding lookup.
            pred_atom: Whether to predict per-atom outputs.
            pred_vel: Whether to predict per-atom velocities.
            pred_cell: Whether to predict per-graph cell updates.
            zero_cog: Whether to zero the center of gravity of predicted velocities.
            time_emb: Optional custom time-embedding module. Defaults to FourierEmbedding.
            lattice_dim: Flattened size of the lattice (``cell``) feature.  Use
                ``9`` for the mattergen 3x3 representation and ``6`` for the KLDM
                6D representation.

        """
        super().__init__()

        self.act_fn = nn.SiLU()

        # Node embedding
        if smooth:
            self.node_embedding = nn.Linear(
                in_features=atom_dim,
                out_features=hidden_dim,
                bias=False,
            )
        else:
            self.node_embedding = nn.Embedding(
                num_embeddings=atom_dim + 1,  # the +1 is for mask token
                embedding_dim=hidden_dim,
            )

        self.atom_latent_emb = nn.Linear(in_features=hidden_dim + time_dim, out_features=hidden_dim)

        self.dis_emb = SinEmbedding(n_frequencies=num_freqs)

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)

        self.time_emb = time_emb

        # Message-passing layers
        self.layers = nn.ModuleList(
            modules=[CSPVLayer(self.dis_emb, hidden_dim=hidden_dim, act_fn=self.act_fn, ln=ln, lattice_dim=lattice_dim) for _ in range(num_layers)]
        )

        # Layer normalization
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # Readout heads
        if pred_vel:
            self.out_vel = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), self.act_fn, nn.Linear(hidden_dim, 3, bias=False))

        if pred_cell:
            self.out_cell = nn.Linear(hidden_dim, out_features=lattice_dim, bias=False)

        if pred_atom:
            self.out_atom = nn.Linear(hidden_dim, atom_dim)

        self.ln = ln
        self.smooth = smooth
        self.pred_vel = pred_vel
        self.pred_cell = pred_cell
        self.pred_atom = pred_atom
        self.zero_cog = zero_cog
        self.lattice_dim = lattice_dim

    def forward(  # noqa: PLR0913
        self,
        t: Tensor,
        pos: Tensor,
        vel: Tensor,
        atom: Tensor,  # atom types
        lattice: Tensor,
        node_index: Tensor,
        edge_node_index: Tensor,
    ) -> dict[str, Tensor]:
        """Run forward pass of CSPVNet.

        Args:
            t: Time values of shape (B,).
            pos: Atom positions of shape (N, 3).
            vel: Atom velocities of shape (N, 3).
            atom: Atom type indices or features of shape (N,) or (N, atom_dim).
            lattice: Lattice tensor — either shape (B, 3, 3) or shape (B, 6)
                depending on the representation stored in ``batch.cell``.
            node_index: Graph index per node of shape (N,).
            edge_node_index: Edge connectivity of shape (2, E).

        Returns:
            Dictionary with optional keys ``'vel'`` (N, 3),
            ``'cell'`` (B, 3, 3) or (B, 6), and ``'h'`` (N, atom_dim).

        """
        # Time embedding — FourierEmbedding expects [..., in_features]; t is [B].
        t_emb = self.time_emb(t.unsqueeze(-1))  # (B, time_dim)
        t_per_atom = t_emb[node_index]  # (N, time_dim)

        # Node features
        node_features = self.node_embedding(atom)  # (N, hidden_dim)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # Edge metadata - wrap to minimum image on torus [0, 1)
        pos_diff = pos[edge_node_index[1]] - pos[edge_node_index[0]]  # (E, 3)
        pos_diff = pos_diff - pos_diff.round()  # minimum image convention, see p. 3 KLDM
        edge_graph_index = node_index[edge_node_index[0]]  # (E, )
        lattice_flat = lattice.reshape(-1, self.lattice_dim)  # (B, lattice_dim)

        # Message passing
        for layer in self.layers:
            node_features = layer(
                pos_diff=pos_diff,
                vel=vel,
                node_features=node_features,
                lattice=lattice_flat,
                edge_node_index=edge_node_index,
                edge_graph_index=edge_graph_index,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        out: dict[str, Tensor] = {}

        if self.pred_vel:
            out_vel = self.out_vel(node_features)
            if self.zero_cog:
                out_vel = scatter_center(
                    pos=out_vel,  # out velocities
                    index=node_index,  # node index
                )
            out["vel"] = out_vel

        if self.pred_cell:
            graph_features = scatter(
                src=node_features,
                index=node_index,
                dim=0,
                reduce="mean",
            )
            out_lattice = self.out_cell(graph_features)  # (B, lattice_dim)
            # Restore the original shape: (B, 3, 3) or (B, 6)
            out["cell"] = out_lattice.reshape(-1, *lattice.shape[1:])

        if self.pred_atom:
            out["atomic_numbers"] = self.out_atom(node_features)

        return out
