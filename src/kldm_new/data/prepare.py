"""Dataset download and preprocessing utilities for KLDM-New.

This module provides a lightweight pipeline to:
1) download raw CSV splits when missing
2) normalize selected metadata columns
3) build MatterGen cache folders used by training

It is used automatically by ``KLDMNewDataModule`` when cache folders are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from mattergen.common.data.dataset import CrystalDatasetBuilder
from mattergen.common.utils.globals import PROPERTY_SOURCE_IDS
from pymatgen.symmetry.groups import SpaceGroup
from tqdm.auto import tqdm

VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    """Configuration needed to fetch and normalize a dataset."""

    name: str
    base_url: str
    properties_map: dict[str, str]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mp_20": DatasetSpec(
        name="mp_20",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/",
        properties_map={
            "formation_energy_per_atom": "formation_energy_per_atom",
            "band_gap": "dft_band_gap",
            "e_above_hull": "energy_above_hull",
            "spacegroup.number": "space_group",
        },
    ),
    "carbon_24": DatasetSpec(
        name="carbon_24",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/carbon_24/",
        properties_map={
            "energy_per_atom": "formation_energy_per_atom",
            "spacegroup.number": "space_group",
        },
    ),
    "mpts_52": DatasetSpec(
        name="mpts_52",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mpts_52/",
        properties_map={
            "energy_above_hull": "energy_above_hull",
            "formation_energy_per_atom": "formation_energy_per_atom",
        },
    ),
    "perov_5": DatasetSpec(
        name="perov_5",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/",
        properties_map={
            "heat_all": "formation_energy_per_atom",
            "ind_gap": "dft_band_gap",
            "spacegroup.number": "space_group",
        },
    ),
}


def _space_group_map() -> dict[int, str]:
    return {i: SpaceGroup.from_int_number(i).symbol for i in range(1, len(SpaceGroup.full_sg_mapping) + 1)}


def _download_split(spec: DatasetSpec, split: str, raw_dir: Path) -> Path:
    if split not in VALID_SPLITS:
        msg = f"Invalid split {split!r}, expected one of {VALID_SPLITS}"
        raise ValueError(msg)

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{split}.csv"
    if out_path.exists():
        return out_path

    response = requests.get(f"{spec.base_url}{split}.csv", stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with (
        Path.open(out_path, "wb") as handle,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {spec.name}:{split}",
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                handle.write(chunk)
                pbar.update(len(chunk))

    return out_path


def _normalized_dataframe(raw_csv: Path, spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df = df.rename(columns=spec.properties_map)

    if "space_group" in df.columns:
        df["space_group"] = df["space_group"].map(_space_group_map())

    return df


def _build_cache_from_csv(raw_csv: Path, out_dir: Path, spec: DatasetSpec) -> None:
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    builder = CrystalDatasetBuilder.from_csv(
        csv_path=str(raw_csv),
        cache_path=str(out_dir),
        transforms=[],
    )

    df = _normalized_dataframe(raw_csv, spec)
    available_props = sorted(set(df.columns) & set(PROPERTY_SOURCE_IDS))

    for prop in available_props:
        if prop in builder.property_names:
            continue
        values = df[prop].to_numpy()
        data_dict = dict(zip(builder.structure_id, values, strict=False))
        builder.add_property_to_cache(prop, data_dict)

    done_file = out_dir / "DONE"
    done_file.touch()


def ensure_preprocessed_dataset(
    *,
    data_path: str | Path,
    dataset_name: str,
    splits: tuple[str, ...] = VALID_SPLITS,
) -> Path:
    """Ensure cache folders exist for all requested splits.

    Parameters
    ----------
    data_path:
        Path to the processed cache root (typically ``.../<dataset>/processed``).
    dataset_name:
        Short dataset key, e.g. ``mp_20``.
    splits:
        Split names to check and build.

    Returns
    -------
    Path
        Resolved processed root path.

    """
    processed_root = Path(data_path).expanduser().resolve()
    spec = DATASET_SPECS.get(dataset_name)
    if spec is None:
        msg = f"Unknown dataset {dataset_name!r}. Available: {sorted(DATASET_SPECS)}"
        raise ValueError(msg)

    dataset_root = processed_root.parent
    raw_root = dataset_root / "raw"

    for split in splits:
        if split not in VALID_SPLITS:
            msg = f"Invalid split {split!r}, expected one of {VALID_SPLITS}"
            raise ValueError(msg)

        split_cache = processed_root / split
        if (split_cache / "DONE").exists():
            continue

        raw_csv = _download_split(spec, split, raw_root)
        _build_cache_from_csv(raw_csv=raw_csv, out_dir=split_cache, spec=spec)

    return processed_root


def infer_dataset_name_from_processed_path(data_path: str | Path) -> str:
    """Infer dataset key from a processed-root path.

    Example: ``data/mp_20/processed`` -> ``mp_20``.
    """
    p = Path(data_path)
    return p.parent.name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and preprocess KLDM-New datasets")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--data-path", default="data/mp_20/processed")
    parser.add_argument("--splits", nargs="+", default=list(VALID_SPLITS), choices=list(VALID_SPLITS))
    args = parser.parse_args()

    ensure_preprocessed_dataset(
        data_path=args.data_path,
        dataset_name=args.dataset,
        splits=tuple(args.splits),
    )
