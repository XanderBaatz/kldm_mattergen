"""Synthetic in-memory dataset and data module for debug/CI runs.

No real data files are needed — crystal structures are generated randomly on the
fly using the same field layout as the mp_20 data module.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate
from torch.utils.data import DataLoader, Dataset


class SyntheticCrystalDataset(Dataset):
    """In-memory dataset of randomly generated crystal graphs.

    Each sample has the same field layout as a transformed mp_20 ChemGraph
    processed by the ``mp_20_kldm`` pipeline (6D cell, fully connected graph).
    """

    def __init__(
        self,
        size: int = 16,
        n_atoms: int = 6,
        cell_dim: int = 6,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.n_atoms = n_atoms
        self.cell_dim = cell_dim
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._samples = [self._make(rng) for _ in range(size)]

    def _make(self, rng: torch.Generator) -> ChemGraph:
        n = self.n_atoms
        # Fully-connected edge index (excluding self-loops)
        rows = torch.arange(n).repeat(n)
        cols = torch.arange(n).repeat_interleave(n)
        mask = rows != cols
        edge_node_index = torch.stack([rows[mask], cols[mask]])

        return ChemGraph(
            pos=torch.rand(n, 3, generator=rng),
            cell=torch.rand(1, self.cell_dim, generator=rng),
            atomic_numbers=torch.randint(1, 100, (n,), generator=rng),
            num_atoms=torch.tensor(n),
            edge_node_index=edge_node_index,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> ChemGraph:
        return self._samples[idx]


class DebugDataModule(pl.LightningDataModule):
    """Lightweight data module that serves ``SyntheticCrystalDataset`` samples.

    Suitable for smoke-testing the training loop without needing real data.
    """

    def __init__(
        self,
        size: int = 16,
        n_atoms: int = 6,
        cell_dim: int = 6,
        batch_size: int = 4,
    ) -> None:
        super().__init__()
        self._size = size
        self._n_atoms = n_atoms
        self._cell_dim = cell_dim
        self._batch_size = batch_size

    def _dataset(self, seed: int) -> SyntheticCrystalDataset:
        return SyntheticCrystalDataset(
            size=self._size,
            n_atoms=self._n_atoms,
            cell_dim=self._cell_dim,
            seed=seed,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset(0), batch_size=self._batch_size, shuffle=True, collate_fn=collate)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset(1), batch_size=self._batch_size, shuffle=False, collate_fn=collate)
