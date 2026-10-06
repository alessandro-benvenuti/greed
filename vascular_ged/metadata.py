from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GraphRecord:
    sample_id: str
    patient_id: str
    patch_index: str
    relative_path: str
    node_count: int
    edge_count: int
    relationformer_split: str = "train"

    @property
    def size(self) -> int:
        return self.node_count + self.edge_count


ALIASES = {
    "sample_id": ("sample_id", "patch_id", "id", "name", "filename"),
    "patient_id": ("patient_id", "patient", "subject_id", "subject", "case_id"),
    "patch_index": ("patch_index", "patch_idx", "spatial_index", "index"),
    "relative_path": ("graph_relative_path", "relative_path", "vtp_path", "graph_path", "path", "filename"),
    "node_count": ("node_count", "num_nodes", "n_nodes", "nodes"),
    "edge_count": ("edge_count", "num_edges", "n_edges", "edges"),
    "relationformer_split": ("relationformer_split", "split", "dataset_split"),
}


def _count(value: str, logical: str, row_num: int) -> int:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"invalid {logical} at patch-index row {row_num}: {value!r}") from error
    if not math.isfinite(parsed) or parsed < 0 or not parsed.is_integer():
        raise ValueError(f"invalid {logical} at patch-index row {row_num}: {value!r}")
    return int(parsed)


def _resolve(fieldnames: Iterable[str], requested: str | None, logical: str, required: bool = True) -> str | None:
    names = set(fieldnames)
    if requested:
        if requested not in names:
            raise ValueError(f"requested {logical} column {requested!r} is absent")
        return requested
    for candidate in ALIASES[logical]:
        if candidate in names:
            return candidate
    if required:
        raise ValueError(f"cannot infer {logical}; available columns: {sorted(names)}")
    return None


def load_patch_index(
    path: str | Path,
    *,
    columns: dict[str, str] | None = None,
    train_graph_prefix: str = "train/vtp",
) -> list[GraphRecord]:
    columns = columns or {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("patch index has no header")
        resolved = {
            key: _resolve(reader.fieldnames, columns.get(key), key, required=key not in {"sample_id", "relative_path", "relationformer_split"})
            for key in ALIASES
        }
        records = []
        for row_num, row in enumerate(reader, start=2):
            split = row.get(resolved["relationformer_split"] or "", "train").strip().lower() or "train"
            if split not in {"train", "training", "val", "validation", "test", "testing"}:
                raise ValueError(f"unknown RelationFormer split at row {row_num}: {split!r}")
            if split not in {"train", "training"}:
                continue
            patient = row[resolved["patient_id"]].strip()  # type: ignore[index]
            patch = row[resolved["patch_index"]].strip()  # type: ignore[index]
            sample = row.get(resolved["sample_id"] or "", "").strip() or f"{patient}_{patch}_{row_num}"
            if not patient or not patch or not sample:
                raise ValueError(f"empty graph identity field at patch-index row {row_num}")
            relative = row.get(resolved["relative_path"] or "", "").strip()
            if not relative:
                relative = f"{train_graph_prefix}/{sample}_graph.vtp"
            relative = relative.lstrip("/")
            if not relative.startswith(f"{train_graph_prefix}/"):
                # Metadata may store just a filename. It must still resolve under train/vtp.
                relative = f"{train_graph_prefix}/{Path(relative).name}"
            records.append(
                GraphRecord(
                    sample, patient, patch, relative,
                    _count(row[resolved["node_count"]], "node_count", row_num),  # type: ignore[index]
                    _count(row[resolved["edge_count"]], "edge_count", row_num),  # type: ignore[index]
                    "train",
                )
            )
    if not records:
        raise ValueError("patch index produced no RelationFormer training records")
    sample_ids = [record.sample_id for record in records]
    relative_paths = [record.relative_path for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("patch index contains duplicate training sample IDs")
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("patch index contains duplicate training graph paths")
    return records
