"""MatterGen-native dataset downloader and preprocessor for kldm_plus.

This utility keeps the existing MatterGen cache layout used by the data module:

- <data_root>/<dataset>/raw/{train,val,test}.csv
- <data_root>/<dataset>/processed/{train,val,test}/(pos.npy, cell.npy, ...)

Optionally it also computes ``train_loc_scale.json`` (frnct-style log-length stats
per number of atoms) in ``<data_root>/<dataset>/train_loc_scale.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from mattergen.common.data.dataset import CrystalDatasetBuilder

VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    base_url: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "mp_20": DatasetSpec(
        name="mp_20",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mp_20/",
    ),
    "carbon_24": DatasetSpec(
        name="carbon_24",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/carbon_24/",
    ),
    "mpts_52": DatasetSpec(
        name="mpts_52",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/mpts_52/",
    ),
    "perov_5": DatasetSpec(
        name="perov_5",
        base_url="https://raw.githubusercontent.com/jiaor17/DiffCSP/refs/heads/main/data/perov_5/",
    ),
}


def _download_split(spec: DatasetSpec, split: str, raw_dir: Path, force: bool) -> Path:
    if split not in VALID_SPLITS:
        msg = f"Invalid split {split!r}. Expected one of {VALID_SPLITS}."
        raise ValueError(msg)

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_csv = raw_dir / f"{split}.csv"
    if out_csv.exists() and not force:
        return out_csv

    url = f"{spec.base_url}{split}.csv"
    print(f"Downloading {url} -> {out_csv}")
    urlretrieve(url, out_csv)  # noqa: S310
    return out_csv


def _build_processed_cache(raw_csv: Path, split_cache_dir: Path, force: bool) -> None:
    done_marker = split_cache_dir / "DONE"
    if done_marker.exists() and not force:
        return

    split_cache_dir.mkdir(parents=True, exist_ok=True)
    CrystalDatasetBuilder.from_csv(
        csv_path=str(raw_csv),
        cache_path=str(split_cache_dir),
        transforms=[],
    )
    done_marker.touch()


def _compute_train_loc_scale(processed_train_dir: Path, out_json: Path, quantile: float = 0.025) -> None:
    """Compute frnct-style log-length stats grouped by number of atoms.

    Uses MatterGen cached arrays ``cell.npy`` and ``num_atoms.npy``.
    """
    cell = np.load(processed_train_dir / "cell.npy")
    num_atoms = np.load(processed_train_dir / "num_atoms.npy")

    lengths = np.linalg.norm(cell, axis=2)
    log_lengths = np.log(lengths)

    grouped: dict[int, list[np.ndarray]] = {}
    for n, vec in zip(num_atoms.tolist(), log_lengths, strict=False):
        grouped.setdefault(int(n), []).append(vec)

    loc_scale: dict[int, tuple[list[float], list[float]]] = {}
    for n_atoms, rows in grouped.items():
        vals = np.array(rows)
        vals_sorted = np.sort(vals, axis=0)
        q = int(len(vals_sorted) * quantile)
        vals_trimmed = vals_sorted[q:-q, :] if q > 0 and len(vals_sorted) > 2 * q else vals_sorted
        loc = vals_trimmed.mean(axis=0)
        scale = vals_trimmed.std(axis=0)
        loc_scale[n_atoms] = (loc.tolist(), scale.tolist())

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in sorted(loc_scale.items())}, handle, indent=2)
    print(f"Wrote {out_json}")


def prepare_dataset(  # noqa: PLR0913
    *,
    dataset: str,
    data_root: str | Path = "data",
    splits: tuple[str, ...] = VALID_SPLITS,
    download: bool = True,
    process: bool = True,
    make_train_loc_scale: bool = True,
    force_download: bool = False,
    force_process: bool = False,
) -> Path:
    """Download and/or preprocess dataset using MatterGen cache conventions."""
    if dataset not in DATASET_SPECS:
        msg = f"Unknown dataset {dataset!r}. Available: {sorted(DATASET_SPECS)}"
        raise ValueError(msg)

    spec = DATASET_SPECS[dataset]
    data_root = Path(data_root).expanduser().resolve()
    dataset_root = data_root / dataset
    raw_root = dataset_root / "raw"
    processed_root = dataset_root / "processed"

    for split in splits:
        if split not in VALID_SPLITS:
            msg = f"Invalid split {split!r}. Expected one of {VALID_SPLITS}."
            raise ValueError(msg)

        raw_csv = raw_root / f"{split}.csv"
        if download:
            raw_csv = _download_split(spec, split, raw_root, force=force_download)
        elif not raw_csv.exists():
            msg = f"Raw CSV not found at {raw_csv}. Use --download or provide the file."
            raise FileNotFoundError(msg)

        if process:
            _build_processed_cache(raw_csv=raw_csv, split_cache_dir=processed_root / split, force=force_process)

    if make_train_loc_scale:
        train_cache = processed_root / "train"
        if not train_cache.exists():
            msg = f"Train cache missing at {train_cache}. Run with --process first."
            raise FileNotFoundError(msg)
        _compute_train_loc_scale(
            processed_train_dir=train_cache,
            out_json=processed_root / "train_loc_scale.json",
        )

    return processed_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare kldm_plus datasets with MatterGen cache format")
    parser.add_argument("--dataset", default="mp_20", choices=sorted(DATASET_SPECS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--splits", nargs="+", default=list(VALID_SPLITS), choices=list(VALID_SPLITS))

    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--process", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-train-loc-scale", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--force-download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-process", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    processed_root = prepare_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        splits=tuple(args.splits),
        download=args.download,
        process=args.process,
        make_train_loc_scale=args.make_train_loc_scale,
        force_download=args.force_download,
        force_process=args.force_process,
    )
    print(f"Prepared dataset cache at {processed_root}")


if __name__ == "__main__":
    main()
