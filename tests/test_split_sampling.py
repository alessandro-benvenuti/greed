import csv
from pathlib import Path

from vascular_ged.metadata import GraphRecord
from vascular_ged.sampling import MANIFEST_FIELDS, sample_pairs, stable_pair_id, write_manifest
from vascular_ged.split import patient_split


def records():
    result = []
    for p in range(20):
        for patch in range(6):
            result.append(GraphRecord(f"p{p}_x{patch}", f"p{p}", str(patch), f"train/vtp/p{p}_x{patch}.vtp", 2 + patch, 1 + patch))
    return result


def test_patient_split_is_stable_and_disjoint():
    data = records()
    first = patient_split(data, 123)
    second = patient_split(list(reversed(data)), 123)
    assert first == second
    assert list(first.values()).count("validation") == 4
    assert set(p for p, s in first.items() if s == "train").isdisjoint(p for p, s in first.items() if s == "validation")


def test_sampling_is_deterministic_unique_and_schema_complete(tmp_path: Path):
    data = records(); assignment = patient_split(data, 123)
    one, _ = sample_pairs(data, assignment, "train", 45, 456, "dataset", "cost")
    two, _ = sample_pairs(list(reversed(data)), assignment, "train", 45, 456, "dataset", "cost")
    assert one == two
    assert len({row["pair_id"] for row in one}) == 45
    assert len({tuple(sorted((row["sample_a"], row["sample_b"]))) for row in one}) == 45
    assert all(row["patient_a"] != row["patient_b"] for row in one)
    path = tmp_path / "manifest.csv"; write_manifest(path, one)
    with path.open(newline="") as handle:
        assert tuple(csv.DictReader(handle).fieldnames) == MANIFEST_FIELDS


def test_pair_id_is_unordered():
    assert stable_pair_id("a", "b", "d", "c") == stable_pair_id("b", "a", "d", "c")
