from pathlib import Path

import pandas as pd
import requests
from mattergen.common.data.chemgraph import ChemGraph  # noqa: TC002
from mattergen.common.data.dataset import CrystalDataset, CrystalDatasetBuilder, DatasetTransform
from mattergen.common.data.transform import Transform  # noqa: TC002
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS
from pymatgen.symmetry.groups import SpaceGroup
from torch.utils.data import Dataset
from tqdm.auto import tqdm


# Inspired by: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
class CrystalDatasetWrapper(Dataset):
    """Dataset class for loading crystal structures for CIF-compatible dataset."""

    dataset_name: str
    url: str  # Placeholder URL
    properties_map: dict[str, str]  # Mapping from raw property names to standardized property names used in ChemGraph

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transforms: list[Transform] | None = None,
        dataset_transforms: list[DatasetTransform] | None = None,
        download: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the CrystalDatasetWrapper."""
        if not isinstance(split, str) or split not in ["train", "val", "test"]:
            msg = "split must be one of 'train', 'val', or 'test'"
            raise ValueError(msg)

        if isinstance(root, str):
            root = Path(root).expanduser()
        self.root = Path(root)

        self.split = split
        self.transforms = transforms if transforms is not None else []
        self.dataset_transforms = dataset_transforms if dataset_transforms is not None else []

        if download:
            self.download()

        if not self._check_exists_raw():
            msg = "Dataset not found. You can use download=True to download it"
            raise RuntimeError(msg)

        self._df_raw, self.properties = self._prepare_df()  # Load the raw CSV data into a DataFrame and extract property names
        self.data: CrystalDataset = self._build()

    def _prepare_df(self) -> tuple[pd.DataFrame, list[str]]:
        """Prepare the DataFrame by renaming columns and extracting property names."""
        df = pd.read_csv(self.raw_folder / f"{self.split}.csv")
        df = df.rename(columns=self.properties_map)  # Rename columns based on the properties_map

        space_group_map = {
            i: SpaceGroup.from_int_number(i).symbol for i in range(1, len(SpaceGroup.full_sg_mapping) + 1)
        }  # Precompute space group mappings

        # Convert numeric space_group to Hermann-Mauguin string symbols
        if "space_group" in df.columns:
            df["space_group"] = df["space_group"].map(space_group_map)  # Map numeric space group to string symbols

        properties = list(set(df.columns) & set(PROPERTY_SOURCE_IDS))  # Extract the standardized property names
        return df, properties

    def _build(self) -> CrystalDataset:
        """Build the dataset using CrystalDatasetBuilder."""
        processed_path = self.processed_folder / f"{self.split}"

        if not self._check_exists_processed():
            self.processed_folder.mkdir(parents=True, exist_ok=True)

            builder = CrystalDatasetBuilder.from_csv(
                csv_path=(self.raw_folder / f"{self.split}.csv").__str__(),
                cache_path=processed_path.__str__(),
                transforms=self.transforms,
            )

            for prop in self.properties:
                if prop not in builder.property_names:  # Check if the property is already in the cache
                    values = self.df[prop].to_numpy()
                    data_dict = dict(zip(builder.structure_id, values, strict=False))
                    builder.add_property_to_cache(prop, data_dict)
        else:
            builder = CrystalDatasetBuilder.from_cache_path(
                cache_path=processed_path.__str__(),
                transforms=self.transforms,
                properties=self.properties,
            )

        return builder.build(dataset_class=CrystalDataset, dataset_transforms=self.dataset_transforms)

    def __getitem__(self, index: int) -> ChemGraph:
        """Return the sample at the given index."""
        return self.data[index]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    @property
    def df(self) -> pd.DataFrame:
        """Return the DataFrame, filtered by the indices if available."""
        if len(self.data) != len(self._df_raw):
            mask = self._df_raw["material_id"].isin(self.data.structure_id)
            return self._df_raw[mask].reset_index(drop=True)
        return self._df_raw

    @property
    def raw_folder(self) -> Path:
        """Returns the path to the raw data folder."""
        return Path(self.root, self.dataset_name, "raw")

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

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        response = requests.get(url=self.url + f"{self.split}.csv", stream=True, timeout=40)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024

        output_file = self.raw_folder / f"{self.split}.csv"

        with (
            Path.open(output_file, "wb") as f,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=f"Downloading {self.dataset_name} {self.split} dataset") as pbar,
        ):
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


class Carbon24(CrystalDatasetWrapper):
    """Carbon-24 dataset first published by Jha et al., 2018."""

    dataset_name = "carbon_24"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/carbon_24/"

    properties_map = {  # noqa: RUF012
        "energy_per_atom": "formation_energy_per_atom",
        "spacegroup.number": "space_group",
    }


class MP20(CrystalDatasetWrapper):
    """MP-20 dataset first published by Jain et al., 2013."""

    dataset_name = "mp_20"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/"

    properties_map = {  # noqa: RUF012
        "formation_energy_per_atom": "formation_energy_per_atom",
        "band_gap": "dft_band_gap",
        "e_above_hull": "energy_above_hull",
        "spacegroup.number": "space_group",
    }


class MPTS52(CrystalDatasetWrapper):
    """MPTS-52 dataset first published by Jha et al., 2018."""

    dataset_name = "mpts_52"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mpts_52/"

    properties_map = {  # noqa: RUF012
        "energy_above_hull": "energy_above_hull",
        "formation_energy_per_atom": "formation_energy_per_atom",
    }


class Perov5(CrystalDatasetWrapper):
    """Perovskite dataset first published by Jha et al., 2018."""

    dataset_name = "perov_5"
    url = "https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/"

    properties_map = {  # noqa: RUF012
        "heat_all": "formation_energy_per_atom",
        "ind_gap": "dft_band_gap",
        "spacegroup.number": "space_group",
    }


if __name__ == "__main__":
    dataset = Perov5(
        root="data",
        split="train",
        download=True,  # Set to True to download the dataset if not present
    )
    # print(dataset.data)  # noqa: ERA001
    # print(f"Loaded {dataset.dataset_name} {dataset.split} dataset with {len(dataset.data)} samples.")  # noqa: ERA001
