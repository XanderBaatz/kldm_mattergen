"""Lightning DataModule wrapping MatterGen's CrystalDataset for KLDM-New.

Hydra instantiates this via ``_target_: kldm_new.scripts._datamodule.KLDMNewDataModule``.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from mattergen.common.data.collate import collate
from mattergen.common.data.dataset import CrystalDataset
from mattergen.common.data.transform import Transform
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from kldm_new.data.prepare import ensure_preprocessed_dataset, infer_dataset_name_from_processed_path


class KLDMNewDataModule(LightningDataModule):
    """Thin wrapper around MatterGen's ``CrystalDataset`` with standard DataLoaders.

    Parameters
    ----------
    data_path : str | Path
        Root directory containing ``train/``, ``val/``, ``test/`` sub-folders
        with numpy cache files produced by MatterGen's data pipeline.
    train_batch_size, val_batch_size, test_batch_size : int
        Batch sizes for each split.
    num_workers : int
        DataLoader workers.
    pin_memory : bool
        Pin memory for GPU transfers.

    """

    def __init__(
        self,
        data_path: str | Path,
        dataset_name: str | None = None,
        auto_prepare: bool = True,
        transforms: list[Transform] | None = None,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["transforms"])
        # Resolve relative to original cwd (Hydra changes cwd before instantiation)
        p = Path(data_path)
        if not p.is_absolute():
            try:
                original_cwd = Path(hydra.utils.get_original_cwd())
            except ValueError:
                original_cwd = Path.cwd()
            p = original_cwd / p
        self.data_path = p.resolve()
        self.dataset_name = dataset_name or infer_dataset_name_from_processed_path(self.data_path)
        self.auto_prepare = auto_prepare
        self.transforms = transforms

    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Load datasets from cache. Called on every process in DDP."""
        data_path = self.data_path
        required_splits: tuple[str, ...]
        if stage in (None, "fit"):
            required_splits = ("train", "val")
        elif stage == "test" or stage == "predict":
            required_splits = ("test",)
        else:
            required_splits = tuple()

        if self.auto_prepare and required_splits:
            ensure_preprocessed_dataset(
                data_path=data_path,
                dataset_name=self.dataset_name,
                splits=required_splits,
            )

        if stage in (None, "fit"):
            self.train_dataset = CrystalDataset.from_cache_path(str(data_path / "train"), transforms=self.transforms)
            self.val_dataset = CrystalDataset.from_cache_path(str(data_path / "val"), transforms=self.transforms)
        if stage in (None, "test"):
            self.test_dataset = CrystalDataset.from_cache_path(str(data_path / "test"), transforms=self.transforms)
        if stage == "predict":
            self.predict_dataset = CrystalDataset.from_cache_path(str(data_path / "test"), transforms=self.transforms)

    # ------------------------------------------------------------------

    def _make_loader(self, dataset: CrystalDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, self.hparams.val_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, self.hparams.test_batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._make_loader(self.predict_dataset, self.hparams.test_batch_size, shuffle=False)
