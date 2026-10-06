from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .atomic import write_csv
from .metadata import GraphRecord

SPLIT_FIELDS = ("patient_id", "relationformer_split", "greed_split", "split_seed")


def patient_split(records: list[GraphRecord], seed: int, validation_fraction: float = 0.2) -> dict[str, str]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    patients = sorted({record.patient_id for record in records})
    if len(patients) < 2:
        raise ValueError("at least two training patients are required")
    ranked = sorted(patients, key=lambda p: hashlib.sha256(f"{seed}\0{p}".encode()).digest())
    n_validation = max(1, min(len(patients) - 1, round(len(patients) * validation_fraction)))
    validation = set(ranked[:n_validation])
    return {patient: ("validation" if patient in validation else "train") for patient in patients}


def write_patient_split(path: str | Path, assignment: dict[str, str], seed: int) -> None:
    rows = ({"patient_id": p, "relationformer_split": "train", "greed_split": s, "split_seed": seed} for p, s in sorted(assignment.items()))
    write_csv(path, rows, SPLIT_FIELDS)


def read_patient_split(path: str | Path) -> tuple[dict[str, str], int]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row["relationformer_split"] != "train" for row in rows):
        raise ValueError("patient split must contain only RelationFormer train rows")
    seeds = {int(row["split_seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError("patient split contains inconsistent seeds")
    assignment = {row["patient_id"]: row["greed_split"] for row in rows}
    if set(assignment.values()) - {"train", "validation"}:
        raise ValueError("invalid greed_split")
    return assignment, seeds.pop()
