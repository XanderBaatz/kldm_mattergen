"""CSP evaluation metrics for kldm_plus.

Mirrors kldm_frnct.metrics.csp but operates on mattergen ChemGraph / ChemGraphBatch
objects and uses the kldm_plus 6D cell encoding (log-lengths + tan(angle - π/2)).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from mattergen.common.data.chemgraph import ChemGraph
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element

# ---------------------------------------------------------------------------
# Structure validity (identical to kldm_frnct / CDVAE definition)
# ---------------------------------------------------------------------------


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    """Return True if no two atoms are closer than ``cutoff`` Å and volume > 0.1 Å³."""
    try:
        dist_mat = structure.distance_matrix
    except Exception:
        return False
    # Ignore self-distances by inflating the diagonal.
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or structure.volume < 0.1:
        return False
    return True


# ---------------------------------------------------------------------------
# 6D cell → pymatgen Structure
# ---------------------------------------------------------------------------


def _decode_cell_6d(
    cell_6d: torch.Tensor,
    angles_loc: float = 0.0,
    angles_scale: float = 0.35,
    lengths_loc: np.ndarray | None = None,
    lengths_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a (6,) tensor in kldm_plus encoding to (lengths_Å, angles_deg).

    Encoding layout: [log_len_a, log_len_b, log_len_c, enc_α, enc_β, enc_γ]
    where enc_angle = (tan(angle_rad − π/2) − angles_loc) / angles_scale
    and log_len = (log(l) − lengths_loc) / lengths_scale when per-atom
    normalization was applied (i.e. lengths_loc/scale are not None).
    """
    cell = cell_6d.detach().cpu().float()
    log_lengths = cell[:3]
    enc_angles = cell[3:]

    # Undo per-atom-count length normalization if it was applied during preprocessing.
    if lengths_loc is not None and lengths_scale is not None:
        loc_t = torch.as_tensor(lengths_loc, dtype=log_lengths.dtype)
        scale_t = torch.as_tensor(lengths_scale, dtype=log_lengths.dtype)
        log_lengths = log_lengths * scale_t + loc_t

    lengths = torch.exp(log_lengths).numpy()

    # Undo standardisation then undo tan transform.
    enc_angles = enc_angles * angles_scale + angles_loc
    angles_rad = torch.atan(enc_angles) + math.pi / 2
    angles_deg = torch.rad2deg(angles_rad).numpy()

    return lengths, angles_deg


def chemgraph_to_structures(
    batch: ChemGraph,
    angles_loc: float = 0.0,
    angles_scale: float = 0.35,
    lengths_loc_scale: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[Structure | None]:
    """Convert a (batched) ChemGraph with 6D cell encoding to pymatgen Structures.

    Returns a list with one entry per crystal; entry is ``None`` if conversion fails.

    Parameters
    ----------
    lengths_loc_scale:
        Dict mapping n_atoms → (loc, scale) arrays for the per-atom-count length
        normalization applied by ContinuousIntervalLattice.  When provided the
        encoded log-lengths are un-normalized before exp().  When ``None`` the
        raw log-lengths are used (i.e. no normalization was applied).
    """
    batch_size = batch.get_batch_size()
    batch_idx = batch.get_batch_idx("pos")  # (N,) crystal index per atom
    atomic_numbers = batch["atomic_numbers"].cpu()  # (N,) integer atom types
    pos = batch["pos"].cpu()  # (N, 3) fractional coords
    # cell is (B, 6) after the 6D transform pipeline
    cell = batch["cell"].cpu().reshape(batch_size, -1)  # (B, 6)

    structures: list[Structure | None] = []
    for i in range(batch_size):
        try:
            mask = batch_idx == i
            n_atoms = int(mask.sum().item())
            atom_z = atomic_numbers[mask].tolist()
            frac = pos[mask].numpy()

            lengths_loc: np.ndarray | None = None
            lengths_scale: np.ndarray | None = None
            if lengths_loc_scale is not None and n_atoms in lengths_loc_scale:
                loc_t, scale_t = lengths_loc_scale[n_atoms]
                lengths_loc = np.asarray(loc_t)
                lengths_scale = np.asarray(scale_t)

            lengths, angles = _decode_cell_6d(
                cell[i], angles_loc, angles_scale, lengths_loc, lengths_scale
            )
            species = [Element.from_Z(z) for z in atom_z]
            s = Structure(
                lattice=Lattice.from_parameters(
                    a=float(lengths[0]),
                    b=float(lengths[1]),
                    c=float(lengths[2]),
                    alpha=float(angles[0]),
                    beta=float(angles[1]),
                    gamma=float(angles[2]),
                ),
                species=species,
                coords=frac,
                coords_are_cartesian=False,
            )
            structures.append(s)
        except Exception:
            structures.append(None)
    return structures


# ---------------------------------------------------------------------------
# CSP Metrics
# ---------------------------------------------------------------------------


class CSPMetrics:
    """Accumulates CSP evaluation metrics over batches.

    Usage::

        metrics = CSPMetrics()
        for pred_batch, gt_batch in zip(predictions, ground_truth):
            pred_structs = chemgraph_to_structures(pred_batch)
            gt_structs   = chemgraph_to_structures(gt_batch)
            metrics.update(pred_structs, gt_structs)
        print(metrics.summarize())
        metrics.reset()

    Parameters
    ----------
    stol, angle_tol, ltol:
        Tolerances forwarded to :class:`pymatgen.analysis.structure_matcher.StructureMatcher`.
    angles_loc, angles_scale:
        Standardisation parameters for the angle encoding — must match the
        ``angles_loc_scale`` used by :class:`kldm_plus.data.transform.ContinuousIntervalLattice`.
    cache_file:
        Path to ``train_loc_scale.json`` produced by
        :func:`kldm_plus.data.prepare.prepare_dataset`.  When provided the
        per-atom-count length normalization is correctly inverted during
        structure decoding.  Set to ``null`` / ``None`` only if the dataset
        was prepared without length normalization.

    """

    def __init__(
        self,
        stol: float = 0.5,
        angle_tol: float = 10.0,
        ltol: float = 0.3,
        angles_loc: float = 0.0,
        angles_scale: float = 0.35,
        cache_file: str | Path | None = None,
    ) -> None:
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.angles_loc = angles_loc
        self.angles_scale = angles_scale
        self.lengths_loc_scale: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if cache_file is not None:
            p = Path(cache_file)
            if p.exists():
                with p.open() as f:
                    loaded = json.load(f)
                self.lengths_loc_scale = {
                    int(k): (torch.tensor(v[0]), torch.tensor(v[1])) for k, v in loaded.items()
                }
        self.reset()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        pred: Sequence[Structure | None],
        target: Sequence[Structure | None],
    ) -> None:
        """Accumulate metrics for one batch of (predicted, target) structure pairs."""
        assert len(pred) == len(target)
        for si, st in zip(pred, target):
            valid = 0
            match = 0
            if si is not None and st is not None:
                valid = int(structure_validity(si))
                if valid:
                    rms = self.matcher.get_rms_dist(si, st)
                    match = int(rms is not None)
                    if rms is not None:
                        self._rmse.append(float(rms[0]))
            self._valid.append(valid)
            self._match.append(match)

    def update_from_chemgraphs(
        self,
        pred: ChemGraph,
        target: ChemGraph,
    ) -> None:
        """Convenience wrapper: convert ChemGraphs then call ``update``."""
        loc_scale = self.lengths_loc_scale or None
        pred_structs = chemgraph_to_structures(pred, self.angles_loc, self.angles_scale, loc_scale)
        gt_structs = chemgraph_to_structures(target, self.angles_loc, self.angles_scale, loc_scale)
        self.update(pred_structs, gt_structs)

    # ------------------------------------------------------------------
    # Summarise
    # ------------------------------------------------------------------

    def summarize(self) -> dict[str, float]:
        n = len(self._valid)
        n_rmse = len(self._rmse)
        return {
            "valid": sum(self._valid) / n if n else 0.0,
            "match_rate": sum(self._match) / n if n else 0.0,
            "rmse": sum(self._rmse) / n_rmse if n_rmse else float("nan"),
        }

    # ------------------------------------------------------------------
    # Reset / details
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._valid: list[int] = []
        self._match: list[int] = []
        self._rmse: list[float] = []

    @property
    def details(self) -> dict[str, list]:
        return {"valid": self._valid, "match": self._match, "rmse": self._rmse}
