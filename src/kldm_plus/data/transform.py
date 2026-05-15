import json
from collections import defaultdict
from pathlib import Path

import torch
from mattergen.common.data.chemgraph import ChemGraph  # noqa: TC002
from mattergen.common.data.transform import Transform  # basically the same as BaseTransform, but immutable
from torch import Tensor
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.utils import dense_to_sparse, one_hot


class PlusOneAtomicNumbers(Transform):
    """Toy example of a Transform that adds 1 to the atomic numbers. Useful for testing."""

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Transform that adds a constant to the atomic numbers."""
        new_atomic_numbers = sample.atomic_numbers + 1
        return sample.replace(atomic_numbers=new_atomic_numbers)


@functional_transform("fully_connected_graph")
class FullyConnectedGraph(Transform):
    """Transform that creates a fully connected graph by adding an 'edge_node_index' attribute to the ChemGraph."""

    def __init__(
        self,
        key: str = "edge_node_index",
        len_from: str = "pos",
    ) -> None:
        """Initialize the FullyConnectedGraph transform."""
        self.key = key
        self.len_from = len_from

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Transform that creates a fully connected graph by adding an 'edge_node_index' attribute to the ChemGraph."""
        n = len(getattr(sample, self.len_from))
        fc_graph = torch.ones(n, n) - torch.eye(n)
        fc_edges, _ = dense_to_sparse(fc_graph)

        return sample.replace(**{self.key: fc_edges})


@functional_transform("continuous_interval_lengths")
class ContinuousIntervalLengths(Transform):
    """Transform the lengths of the lattice into a continuous interval by taking l --> log(l).

    Note: loc and scale should be given in transformed space.
    """

    def __init__(
        self,
        in_key: str = "lengths",
        out_key: str | None = None,
        normalize_by_num_atoms: bool = False,  # noqa: FBT001, FBT002
        cache_file: str | Path | None = None,
        quantile: float = 0.025,  # KLDM uses 0.025
    ) -> None:
        """Initialize the ContinuousIntervalLengths transform."""
        self.in_key = in_key
        self.out_key = out_key
        self.normalize_by_num_atoms = normalize_by_num_atoms
        self.cache_file = Path(cache_file) if cache_file is not None else None
        self.quantile = quantile
        self.loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # load cached loc and scale if available
        if self.cache_file and self.cache_file.exists():
            with self.cache_file.open("r") as f:
                loaded = json.load(f)

            # convert lists to torch tensors
            self.loc_scale = {int(k): (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in loaded.items()}

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Transform the lengths of the lattice into a continuous interval by taking l --> log(l)."""
        if not hasattr(sample, "cell"):
            msg = "ChemGraph must have a 'cell' attribute to use ContinuousIntervalLengths transform."
            raise ValueError(msg)

        n_atoms = int(sample.num_atoms)

        # Compute lattice lengths from the cell matrix
        cell_matrix = sample.cell.squeeze(0)  # shape [3, 3]
        lengths = torch.linalg.norm(cell_matrix, dim=1)  # shape [3]

        # Optional normalization by number of atoms
        if self.normalize_by_num_atoms:
            lengths = lengths / (n_atoms ** (1 / 3))

        # Log-transform
        log_lengths = torch.log(lengths)

        # Apply loc/scale normalization if available
        if n_atoms in self.loc_scale:
            loc, scale = self.loc_scale[n_atoms]
            log_lengths = (log_lengths - loc) / scale

        key = self.out_key if self.out_key is not None else self.in_key
        return sample.replace(**{key: log_lengths})

    def compute_loc_scale(self, samples: list[ChemGraph]) -> None:
        """Compute the loc and scale for the log-transformed lengths based on the given samples."""
        lengths_by_n = defaultdict(list)

        for sample in samples:
            n_atoms = int(sample.num_atoms)
            cell_matrix = sample.cell.squeeze(0)  # shape [3, 3]
            lengths = torch.linalg.norm(cell_matrix, dim=1)  # shape [3]
            if self.normalize_by_num_atoms:
                lengths = lengths / (n_atoms ** (1 / 3))
            log_lengths = torch.log(lengths)
            lengths_by_n[n_atoms].append(log_lengths)

        for n, vals in lengths_by_n.items():
            vals_stack = torch.stack(vals)
            q = int(vals_stack.shape[0] * self.quantile)
            vals_sorted, _ = torch.sort(vals_stack, dim=0)
            vals_trimmed = vals_sorted[q:-q] if q > 0 else vals_sorted  # trim the top and bottom quantiles
            loc = vals_trimmed.mean(dim=0)
            scale = vals_trimmed.std(dim=0)
            self.loc_scale[n] = (loc, scale)

        if self.cache_file:
            to_save = {n: [loc.tolist(), scale.tolist()] for n, (loc, scale) in self.loc_scale.items()}
            with self.cache_file.open("w") as f:
                json.dump(to_save, f, indent=2)

    def invert_one(
        self,
        log_lengths: Tensor,
        n_atoms: int,
    ) -> Tensor:
        """Invert the transformation for a single sample given its log-lengths and number of atoms."""
        lengths = log_lengths.clone()
        if n_atoms in self.loc_scale:
            loc, scale = self.loc_scale[n_atoms]
            lengths = lengths * scale + loc
        lengths = torch.exp(lengths)
        if self.normalize_by_num_atoms:
            lengths = lengths * (n_atoms ** (1 / 3))
        return lengths


@functional_transform("continuous_interval_lattice")
class ContinuousIntervalLattice(Transform):
    """Combined transform for lattice lengths and angles.

    Supports:
        - lengths: log(a,b,c) optionally normalized by n_atoms.
        - angles: φ --> tan(φ - π/2).

    Supports optional loc/scale normalization for both.

    Can cache computed loc/scale for lengths to disk.
    """

    def __init__(  # noqa: PLR0913
        self,
        lengths_in_key: str = "lengths",
        lengths_out_key: str | None = None,
        angles_in_key: str = "angles",
        angles_out_key: str | None = None,
        normalize_lengths_by_num_atoms: bool = False,  # noqa: FBT001, FBT002
        cache_file: str | Path | None = None,
        lengths_quantile: float = 0.025,  # KLDM uses 0.025
        angles_loc_scale: tuple[torch.Tensor, torch.Tensor] | None = None,
        angles_in_deg: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the ContinuousIntervalLattice transform."""
        self.lengths_in_key = lengths_in_key
        self.lengths_out_key = lengths_out_key or lengths_in_key
        self.angles_in_key = angles_in_key
        self.angles_out_key = angles_out_key or angles_in_key
        self.normalize_lengths_by_num_atoms = normalize_lengths_by_num_atoms
        self.cache_file = Path(cache_file) if cache_file else None
        self.lengths_quantile = lengths_quantile

        self.lengths_loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if self.cache_file and self.cache_file.exists():
            with self.cache_file.open("r") as f:
                loaded = json.load(f)
            self.lengths_loc_scale = {int(k): (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in loaded.items()}

        self.angles_loc_scale = angles_loc_scale
        self.angles_in_deg = angles_in_deg

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Apply the continuous interval transformation to both lengths and angles."""
        if not hasattr(sample, "cell"):
            msg = "ChemGraph must have a 'cell' attribute to use ContinuousIntervalLattice transform."
            raise ValueError(msg)

        n_atoms = int(sample.num_atoms)
        cell_matrix = sample.cell.squeeze(0)  # shape [3, 3]

        # Process lengths
        lengths = torch.linalg.norm(cell_matrix, dim=1)  # shape [3]
        if self.normalize_lengths_by_num_atoms:
            lengths = lengths / (n_atoms ** (1 / 3))
        log_lengths = torch.log(lengths)
        if n_atoms in self.lengths_loc_scale:
            loc, scale = self.lengths_loc_scale[n_atoms]
            log_lengths = (log_lengths - loc) / scale

        # Process angles
        alpha = torch.acos(torch.clamp(torch.dot(cell_matrix[1], cell_matrix[2]) / (lengths[1] * lengths[2]), -1.0, 1.0))
        beta = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[2]) / (lengths[0] * lengths[2]), -1.0, 1.0))
        gamma = torch.acos(torch.clamp(torch.dot(cell_matrix[0], cell_matrix[1]) / (lengths[0] * lengths[1]), -1.0, 1.0))
        angles_rad = torch.stack([alpha, beta, gamma])

        transformed_angles = torch.tan(angles_rad - torch.pi / 2)
        if self.angles_loc_scale is not None:
            loc, scale = self.angles_loc_scale
            transformed_angles = (transformed_angles - loc) / scale

        return sample.replace(**{self.lengths_out_key: log_lengths, self.angles_out_key: transformed_angles})


@functional_transform("one_hot")
class OneHot(Transform):
    """Transform that applies one-hot encoding to a specified key in the ChemGraph based on a provided mapping of values to indices."""

    def __init__(  # noqa: PLR0913
        self,
        values: list[int],
        key: str = "h",
        scale: float = 1.0,
        noise_std: float = 0.0,
        dtype: torch.dtype = torch.get_default_dtype(),  # noqa: B008
        expand_as_vector: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the OneHot transform with a mapping from values to indices and optional scaling and noise."""
        self.mapping = {v: i for (i, v) in enumerate(values)}
        self.key = key
        self.dtype = dtype
        self.noise_std = noise_std
        self.scale = scale
        self.expand_as_vector = expand_as_vector

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Apply one-hot encoding to the specified key in the ChemGraph."""
        data_key = getattr(sample, self.key)
        assert data_key.ndim == 1  # noqa: S101

        x = torch.as_tensor([self.mapping[xi.item()] for xi in data_key])
        if self.expand_as_vector:
            x = self.scale * one_hot(x, num_classes=len(self.mapping)).to(self.dtype)
            if self.noise_std > 0.0:
                x = x + torch.randn_like(x) * self.noise_std

        return sample.replace(**{self.key: x})

    def __repr__(self) -> str:
        """Return a string representation of the OneHot transform."""
        return f"{self.__class__.__name__}({self.mapping})"


@functional_transform("concat_features")
class ConcatFeatures(Transform):
    """Concatenate multiple feature tensors along a specified dimension and store in a new key."""

    def __init__(
        self,
        in_keys: list[str],
        out_key: str,
        dim: int = -1,
    ) -> None:
        """Initialize the ConcatFeatures transform."""
        self.in_keys = in_keys
        self.out_key = out_key
        self.dim = dim

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Apply the ConcatFeatures transform to the specified keys in the ChemGraph."""
        features = [getattr(sample, key) for key in self.in_keys]
        concat_features = torch.cat(features, dim=self.dim)
        return sample.replace(**{self.out_key: concat_features})


@functional_transform("cell_to_lattice")
class CellToLattice(Transform):
    """Replace `cell` with 6D lattice vector while preserving original matrix."""

    def __init__(
        self,
        cell_key: str = "cell",
        lattice_key: str = "l",
        matrix_key: str = "cell_matrix",
    ) -> None:
        """Initialize the CellToLattice transform."""
        self.cell_key = cell_key
        self.lattice_key = lattice_key
        self.matrix_key = matrix_key

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Convert the cell matrix to lattice parameters (lengths and angles) and store them in the ChemGraph."""
        return sample.replace(
            **{
                self.matrix_key: getattr(sample, self.cell_key),
                self.cell_key: getattr(sample, self.lattice_key),
            }
        )


@functional_transform("unsqueeze_lattice")
class UnsqueezeLattice(Transform):
    """Ensure lattice feature `l` has a batch dimension.

    Converts shape (6,) -> (1, 6) so PyG stacks to (B, 6) instead of concatenating to (B*6,).
    """

    def __init__(self, key: str = "l", dim: int = 0) -> None:
        """Initialize the UnsqueezeLattice transform."""
        self.key = key
        self.dim = dim

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Apply the UnsqueezeLattice transform to the specified key in the ChemGraph."""
        l = getattr(sample, self.key)  # noqa: E741

        # Only unsqueeze if it's flat (6,)
        if l.ndim == 1:
            l = l.unsqueeze(self.dim)  # (6,) -> (1, 6)  # noqa: E741

        return sample.replace(**{self.key: l})


@functional_transform("batch_lattice")
class BatchLattice(Transform):
    """Ensure lattice feature `l` has a batch dimension.

    Converts shape (6,) -> (1, 6) so PyG stacks to (B, 6) instead of concatenating to (B*6,).
    """

    def __init__(self, key: str = "l", dim: int = 0) -> None:
        """Initialize the BatchLattice transform."""
        self.key = key
        self.dim = dim
        self.out_key = f"{key}_batch"

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Apply the BatchLattice transform to the specified key in the ChemGraph."""
        l = getattr(sample, self.key)  # noqa: E741

        # Only unsqueeze if it's flat (6,)
        if l.ndim == 1:
            l = l.unsqueeze(self.dim)  # (6,) -> (1, 6)  # noqa: E741

        return sample.replace(**{self.out_key: l})


@functional_transform("copy_property")
class CopyProperty(Transform):
    """Copy an existing field to a new key in the ChemGraph.

    Useful for aliasing, e.g. copying ``atomic_numbers`` to ``h`` before
    encoding so that the original field is preserved.
    """

    def __init__(self, src: str, dst: str) -> None:
        """Initialize the CopyProperty transform."""
        self.src = src
        self.dst = dst

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Copy *src* field to *dst* in the ChemGraph."""
        return sample.replace(**{self.dst: getattr(sample, self.src).clone()})

    def __repr__(self) -> str:
        """Return a string representation of the CopyProperty transform."""
        return f"{self.__class__.__name__}({self.src!r} -> {self.dst!r})"


@functional_transform("task_metadata")
class TaskMetadata(Transform):
    """Attach task-level scalar flags to every sample.

    Stored fields (all ``torch.long`` scalars):
        * ``task_id`` - integer task identifier (e.g. 0 = CSP).
        * ``diffuse_h`` - whether the atom-type channel should be diffused
          (``0`` for CSP where species are given, ``1`` for unconditional
          generation).
    """

    def __init__(self, task_id: int = 0, *, diffuse_h: bool = False) -> None:
        """Initialize the TaskMetadata transform."""
        self.task_id = task_id
        self.diffuse_h = diffuse_h

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        """Attach task metadata tensors to the sample."""
        return sample.replace(
            task_id=torch.tensor(self.task_id, dtype=torch.long),
            diffuse_h=torch.tensor(int(self.diffuse_h), dtype=torch.long),
        )

    def __repr__(self) -> str:
        """Return a string representation of the TaskMetadata transform."""
        return f"{self.__class__.__name__}(task_id={self.task_id}, diffuse_h={self.diffuse_h})"
