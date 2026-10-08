import csv
from pathlib import Path

import pytest

from vascular_ged.atomic import write_csv
from vascular_ged.solver import RESULT_FIELDS, run_shard


MANIFEST_ROW = {
    "pair_id": "pair-1",
    "sample_a": "a",
    "sample_b": "b",
    "patient_a": "1",
    "patient_b": "2",
    "greed_split": "train",
    "graph_a_relative_path": "train/vtp/a.vtp",
    "graph_b_relative_path": "train/vtp/b.vtp",
    "cost_config_id": "vascular_3d_raw_v1",
}


def _result_row(**overrides: str) -> dict[str, str]:
    row = {
        **MANIFEST_ROW,
        "solver": "f2",
        "solver_version": "gedlib-test",
        "gurobi_version": "13.0.1",
        "gedlib_version": "v1.0",
        "time_limit_seconds": "60",
        "runtime_seconds": "1.25",
        "peak_memory_bytes": "1234",
        "lower_bound": "2.0",
        "upper_bound": "3.0",
        "absolute_bound_gap": "1.0",
        "relative_bound_gap": str(1 / 3),
        "is_exact": "False",
        "solver_status": "ok",
        "failure_message": "",
        "git_commit": "pilot-commit",
        "slurm_job_id": "pilot-job",
        "slurm_array_task_id": "7",
        "computed_at": "2026-10-06T00:00:00+00:00",
    }
    row.update(overrides)
    return row


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, [MANIFEST_ROW], list(MANIFEST_ROW))
    config = tmp_path / "cost.yaml"
    config.write_text("id: vascular_3d_raw_v1\n", encoding="utf-8")
    reuse_dir = tmp_path / "pilot"
    reuse_dir.mkdir()
    return manifest, config, reuse_dir


def test_run_shard_reuses_compatible_successful_pilot_result(tmp_path: Path) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    cached = _result_row()
    write_csv(reuse_dir / "pilot.csv", [cached], list(cached))
    output = tmp_path / "output.csv"

    # The dataset path is deliberately absent: successful reuse must not load graphs.
    run_shard(manifest, output, tmp_path / "missing-dataset", config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["lower_bound"] == "2.0"
    assert rows[0]["upper_bound"] == "3.0"
    assert rows[0]["git_commit"] == "pilot-commit"
    assert rows[0]["slurm_job_id"] == "pilot-job"


@pytest.mark.parametrize(
    "override",
    [
        {"cost_config_id": "vascular_3d_sqrt3_v1"},
        {"solver": "branch"},
        {"time_limit_seconds": "300"},
        {"solver_status": "failed"},
        {"lower_bound": "nan"},
        {"lower_bound": "4", "upper_bound": "3"},
    ],
)
def test_run_shard_rejects_incompatible_or_invalid_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: dict[str, str],
) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    cached = _result_row(**override)
    write_csv(reuse_dir / "pilot.csv", [cached], list(cached))
    expected = _result_row(lower_bound="9", upper_bound="9", git_commit="new-solve")

    monkeypatch.setattr("vascular_ged.solver.solve_row", lambda *args, **kwargs: expected)
    output = tmp_path / "output.csv"
    run_shard(manifest, output, tmp_path / "dataset", config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["lower_bound"] == "9"
    assert row["git_commit"] == "new-solve"


def test_run_shard_rejects_duplicate_compatible_reuse(tmp_path: Path) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    cached = _result_row()
    for name in ("one.csv", "two.csv"):
        write_csv(reuse_dir / name, [cached], list(cached))

    with pytest.raises(ValueError, match="duplicate reusable result"):
        run_shard(manifest, tmp_path / "output.csv", tmp_path, config, 0, 1, "f2", 60, reuse_dir)


def test_run_shard_ignores_incomplete_reusable_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    cached = _result_row()
    cached.pop("computed_at")
    write_csv(reuse_dir / "pilot.csv", [cached], list(cached))
    expected = _result_row(lower_bound="9", upper_bound="9", git_commit="new-solve")
    monkeypatch.setattr("vascular_ged.solver.solve_row", lambda *args, **kwargs: expected)

    output = tmp_path / "output.csv"
    run_shard(manifest, output, tmp_path, config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["git_commit"] == "new-solve"


def test_reused_row_schema_contains_all_result_fields(tmp_path: Path) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    cached = _result_row()
    write_csv(reuse_dir / "pilot.csv", [cached], list(cached))
    output = tmp_path / "output.csv"
    run_shard(manifest, output, tmp_path, config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert set(RESULT_FIELDS) <= set(row)


def test_run_shard_reuses_successful_row_from_existing_partial_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    output = tmp_path / "output.csv"
    cached = _result_row(git_commit="previous-attempt")
    write_csv(output, [cached], list(cached))

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("a successful existing row must not be recomputed")

    monkeypatch.setattr("vascular_ged.solver.solve_row", unexpected_solve)
    run_shard(manifest, output, tmp_path, config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["git_commit"] == "previous-attempt"


def test_run_shard_saves_failures_and_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, config, reuse_dir = _inputs(tmp_path)
    failed = _result_row(
        solver_status="failed", lower_bound="", upper_bound="", absolute_bound_gap="",
        relative_bound_gap="", failure_message="GRBException: token server unavailable",
    )
    monkeypatch.setattr("vascular_ged.solver.solve_row", lambda *args, **kwargs: failed)
    output = tmp_path / "output.csv"

    with pytest.raises(RuntimeError, match="1 failed solver row"):
        run_shard(manifest, output, tmp_path, config, 0, 1, "f2", 60, reuse_dir)

    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["solver_status"] == "failed"
    assert "token server" in row["failure_message"]
