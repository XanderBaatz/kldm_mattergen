from mattergen.common.data.chemgraph import ChemGraph
from torch import LongTensor  # noqa: TC002


class ChemGraph(ChemGraph):
    """ChemGraph class for representing crystal structures as graphs."""

    def get_batch_idx(self, field_name: str) -> LongTensor | None:
        """Diffusion library uses this to retrieve batch indices for a given field."""
        if field_name in ["cell", "l"]:
            return None
        return super().get_batch_idx(field_name=field_name)
