import csv
from pathlib import Path

import pytest

from vascular_ged.atomic import write_csv
from vascular_ged.results import merge_results


def test_atomic_csv_leaves_complete_file(tmp_path: Path):
    path = tmp_path / "out.csv"
    write_csv(path, [{"a": 1}], ["a"])
    assert path.read_text() == "a\n1\n"
    assert not list(tmp_path.glob(".out.csv.*"))


def test_merge_detects_missing_and_validates_bounds(tmp_path: Path):
    dataset = tmp_path / "data"; (dataset / "train/vtp").mkdir(parents=True)
    for name in ("a", "b"):
        (dataset / f"train/vtp/{name}.vtp").write_text("fixture")
    fields = ["pair_id", "sample_a", "sample_b", "patient_a", "patient_b", "greed_split", "graph_a_relative_path", "graph_b_relative_path"]
    row = dict(pair_id="p", sample_a="a", sample_b="b", patient_a="pa", patient_b="pb", greed_split="train",
               graph_a_relative_path="train/vtp/a.vtp", graph_b_relative_path="train/vtp/b.vtp")
    manifest = tmp_path / "manifest.csv"; write_csv(manifest, [row], fields)
    shards = tmp_path / "shards"; shards.mkdir()
    with pytest.raises(ValueError, match="missing"):
        merge_results(manifest, shards, tmp_path / "merged.csv", dataset, 1)
    result = dict(row, lower_bound="2", upper_bound="1", is_exact="false", solver_status="ok", failure_message="")
    write_csv(shards / "one.csv", [result], list(result))
    with pytest.raises(ValueError, match="invalid bounds"):
        merge_results(manifest, shards, tmp_path / "merged.csv", dataset, 1)
