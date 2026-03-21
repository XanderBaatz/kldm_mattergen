from pathlib import Path

import requests
from mattergen.common.data.dataset import CrystalDataset, CrystalDatasetBuilder
from mattergen.common.data.transform import Transform  # noqa: TC002


# Inspired by: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
class CrystalStructureDataset:
    """Dataset class for loading crystal structures for CIF-compatible dataset."""

    dataset_name = "crystal_structure_dataset"
    url = "https://example.com/"  # Placeholder URL

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: list[Transform] | None = None,
        download: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the CrystalStructureDataset."""
        if not isinstance(split, str) or split not in {"train", "val", "test"}:
            msg = "split must be one of 'train', 'val', or 'test'"
            raise ValueError(msg)

        if isinstance(root, str):
            root = Path(root).expanduser()
        self.root = Path(root)

        self.split = split
        self.transforms = transforms if transforms is not None else []

        if download:
            self.download()

        if not self._check_exists_raw():
            msg = "Dataset not found. You can use download=True to download it"
            raise RuntimeError(msg)

        self.data: CrystalDataset = self._build()

    def _build(self) -> CrystalDataset:
        """Build the dataset using CrystalDatasetBuilder."""
        processed_path = Path(self.processed_folder, f"{self.split}")

        if not self._check_exists_processed():
            self.processed_folder.mkdir(parents=True, exist_ok=True)

            builder = CrystalDatasetBuilder.from_csv(
                csv_path=(self.raw_folder / f"{self.split}.csv").__str__(),
                cache_path=processed_path.__str__(),
                transforms=self.transforms,
            )
        else:
            builder = CrystalDatasetBuilder.from_cache_path(
                cache_path=processed_path.__str__(),
                transforms=self.transforms,
            )

        return builder.build(dataset_class=CrystalDataset)

    @property
    def raw_folder(self) -> Path:
        """Returns the path to the raw data folder."""
        return Path(self.root, self.dataset_name, "raw")  # os.path.join(self.root, self.dataset_name, "raw")

    @property
    def processed_folder(self) -> Path:
        """Returns the path to the processed data folder."""
        return Path(self.root, self.dataset_name, "processed")

    def _check_exists_raw(self) -> bool:
        path = self.raw_folder / f"{self.split}.csv"
        return path.exists()

    def _check_exists_processed(self) -> bool:
        path = self.processed_folder / f"{self.split}"
        return path.exists()

    def download(self) -> None:
        """Download the dataset if it doesn't exist in the raw folder."""
        if self._check_exists_raw():
            return

        Path.mkdir(self.raw_folder, exist_ok=True, parents=True)
        response = requests.get(url=self.url + self.split + ".csv", timeout=40)
        response.raise_for_status()
        with Path.open(self.raw_folder / f"{self.split}.csv", "wb") as f:
            f.write(response.content)


class MP20(CrystalStructureDataset):
    """MP-20 dataset first published by Jain et al., 2013."""

    dataset_name = "mp_20"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/"


class Perov5(CrystalStructureDataset):
    """Perovskite dataset first published by Jha et al., 2018."""

    dataset_name = "perov_5"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/"


class MPTS52(CrystalStructureDataset):
    """MPTS-52 dataset first published by Jha et al., 2018."""

    dataset_name = "mpts_52"
    url = ...


if __name__ == "__main__":
    dataset = Perov5(
        root="data",
        split="val",
        download=True,  # Set to True to download the dataset if not present
    )
    # print(dataset.data)  # noqa: ERA001
    # print(f"Loaded {dataset.dataset_name} {dataset.split} dataset with {len(dataset.data)} samples.")  # noqa: ERA001
