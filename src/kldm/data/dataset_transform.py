# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterable  # noqa: TC003

import numpy as np
from mattergen.common.data.dataset import BaseDataset, CrystalDataset  # noqa: TC002
from pymatgen.core.periodic_table import Element

# Dataset transforms
# These transforms are used to modify the dataset in various ways, such as filtering out
# structures with missing properties.


def filter_elements(dataset: CrystalDataset, exclude_elements: Iterable[int | str]) -> CrystalDataset:
    """Exclude structures containing any atomic number in `exclude_elements`.

    Args:
        dataset: CrystalDataset to filter.
        exclude_elements: Iterable of atomic numbers or element symbols to exclude (e.g., [43, 61, "Tc", "Pm"]).

    Returns:
        Subsetted CrystalDataset with forbidden elements removed.

    """
    forbidden_Z = set()  # noqa: N806
    for e in exclude_elements:
        if isinstance(e, int):
            forbidden_Z.add(e)
        elif isinstance(e, str):
            forbidden_Z.add(Element(e).Z)
        else:
            msg = f"Unsupported type in exclude_elements: {type(e)}"
            raise TypeError(msg)

    # Create a boolean array of forbidden atoms
    forbidden_mask = np.isin(dataset.atomic_numbers, list(forbidden_Z))

    # Sum forbidden atoms per structure
    # len(num_atoms) == number of structures
    counts_per_structure = np.add.reduceat(forbidden_mask, dataset.index_offset)

    # Keep structures with 0 forbidden atoms
    mask = counts_per_structure == 0
    indices = np.nonzero(mask)[0]

    return dataset.subset(list(indices))


def filter_energy_above_hull(dataset: BaseDataset, threshold: float = 0.1) -> BaseDataset:
    """Filter out structures with energy above the hull greater than threshold."""
    if "energy_above_hull" not in dataset.properties:
        # Nothing to filter
        return dataset

    energies = dataset.properties["energy_above_hull"]
    indices = np.where(energies <= threshold)[0]
    return dataset.subset(list(indices))
