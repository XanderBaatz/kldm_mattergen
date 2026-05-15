from torch import Tensor  # noqa: TC002
from torch_scatter import scatter_mean


def scatter_center(pos: Tensor, index: Tensor) -> Tensor:
    """Center positions by subtracting the per-index mean position."""
    return pos - scatter_mean(pos, index=index, dim=0)[index]
