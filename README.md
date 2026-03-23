# KLDM

## Crystal Graph: MatterGen `ChemGraph` vs. original KLDM Torch Geometric `Data` object

### KLDM

KLDM does one PyG `Data` object per crystal structure:
```python
Data(
    pos=...,
    h=...,          # atomic numbers
    lengths=...,
    angles=...
)
```

So the dataset becomes a list of objects:
```python
[data_0, data_1, data_2, ...]
```

each contains its own tensor and own memory allocation. There's a lot of Python overhead per sample and it does not use caching efficiently. This is slow for large datasets.

The way that KLDM handle lattices is `length` + `angles`.


### MatterGen

MatterGen does one big tensorized dataset for all structures in a `ChemGraph` object:
```python
CrystalDataset(
    pos=...,              # all atoms (concatenated)
    atomic_numbers=...,
    num_atoms=...,        # structure boundaries
    cell=...,
    structure_id=...
)
```

The dataset is then of the form:
```python
pos = [atoms of struct 0 | atoms of struct 1 | ...]
num_atoms = [n0, n1, ...]
```

and accessed via `index_offset`. This is better for vectorized operations and enables efficient batching (see `ChemGraphBatch`).

For lattices, MatterGen actually stores a 3x3 matrix as expected.
