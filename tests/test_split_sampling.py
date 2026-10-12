import csv
from pathlib import Path

import pytest

from vascular_ged.metadata import GraphRecord, load_patch_index
from vascular_ged.sampling import MANIFEST_FIELDS, assign_strata, sample_pairs, stable_pair_id, write_manifest
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


def test_expansion_retains_original_pairs_and_fills_balanced_quotas():
    data = records(); assignment = patient_split(data, 123)
    original, _ = sample_pairs(data, assignment, 'train', 45, 456, 'dataset', 'cost')
    extended, _ = sample_pairs(data, assignment, 'train', 180, 456, 'dataset', 'cost', initial_rows=original)
    assert extended[:45] == original
    assert len({row['pair_id'] for row in extended}) == 180
    assert extended == sample_pairs(list(reversed(data)), assignment, 'train', 180, 456, 'dataset', 'cost', initial_rows=original)[0]
    from collections import Counter
    counts = Counter((r['sampling_strategy'], r['size_stratum']) for r in extended)
    assert set(counts.values()) == {20}


def test_expansion_rejects_changed_metadata_and_duplicate_retained_pairs():
    data = records(); assignment = patient_split(data, 123)
    original, _ = sample_pairs(data, assignment, 'train', 45, 456, 'dataset', 'cost')
    with pytest.raises(ValueError, match='duplicate retained'):
        sample_pairs(data, assignment, 'train', 180, 456, 'dataset', 'cost', initial_rows=original + original[:1])
    changed = [dict(row) for row in original]
    changed[0]['node_count_a'] = 999
    with pytest.raises(ValueError, match='metadata mismatch'):
        sample_pairs(data, assignment, 'train', 180, 456, 'dataset', 'cost', initial_rows=changed)


def test_equal_size_records_still_fill_all_strata():
    tied = [GraphRecord(f"g{i}", f"p{i}", "0", f"train/vtp/g{i}.vtp", 2, 1) for i in range(9)]
    strata = assign_strata(tied)
    assert [list(strata.values()).count(name) for name in ("small", "medium", "large")] == [3, 3, 3]


def test_patch_index_rejects_duplicate_samples_and_invalid_counts(tmp_path: Path):
    header = "sample_id,patient_id,split,patch_index,node_count,edge_count\n"
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(header + "a,p1,train,0,1,0\na,p2,train,0,1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate training sample"):
        load_patch_index(duplicate)
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(header + "a,p1,train,0,1.5,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="node_count"):
        load_patch_index(invalid)
