import torch
from mattergen.common.data.chemgraph import ChemGraph  # noqa: TC002
from mattergen.diffusion.score_models.base import ScoreModel
from torch import nn


class TinyScoreModel(ScoreModel):
    """A tiny score model for testing purposes."""

    def __init__(self, hidden_dim: int = 64) -> None:
        """Initialize a tiny score model for testing purposes."""
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
        )

        self.pos_mlp = nn.Sequential(
            nn.Linear(3 + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.cell_mlp = nn.Sequential(
            nn.Linear(9 + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x: ChemGraph, t: torch.Tensor) -> ChemGraph:
        """Predict the score for the given input and time step."""
        t = t.reshape(-1, 1)
        t_emb = self.time_embed(t)

        batch_idx = x.get_batch_idx("pos")
        t_pos = t_emb[batch_idx]

        pos_input = torch.cat([x["pos"], t_pos], dim=-1)
        pred_pos = self.pos_mlp(pos_input)

        cell_input = torch.cat(
            [x["cell"].reshape(x["cell"].shape[0], -1), t_emb],
            dim=-1,
        )
        pred_cell = self.cell_mlp(cell_input).reshape(-1, 3, 3)

        return x.replace(
            pos=pred_pos,
            cell=pred_cell,
        )
