from pathlib import Path

import pytest

from vascular_ged.atomic import write_csv
from vascular_ged.reporting import analyze_pilot
from vascular_ged.sampling import MANIFEST_FIELDS


def _pilot(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "output"
    shard_dir = root / "results/pilot/shards"
    manifest = {field: "value" for field in MANIFEST_FIELDS}
    manifest.update(pair_id="pair", node_count_a="2", node_count_b="3", edge_count_a="1", edge_count_b="2")
    write_csv(root / "manifests/pilot_pairs.csv", [manifest], MANIFEST_FIELDS)
    rows = []
    for cost in ("vascular_3d_raw_v1", "vascular_3d_sqrt3_v1"):
        for solver, timeout in (("f2", 10), ("f2", 60), ("f2", 300), ("branch", 0)):
            rows.append({
                "pair_id": "pair", "cost_config_id": cost, "solver": solver,
                "time_limit_seconds": timeout, "runtime_seconds": .1,
                "lower_bound": 1.0, "upper_bound": 1.0, "absolute_bound_gap": 0.0,
                "relative_bound_gap": 0.0, "is_exact": True, "solver_status": "ok",
                "failure_message": "", "node_count_a": 2, "node_count_b": 3,
                "edge_count_a": 1, "edge_count_b": 2,
            })
    write_csv(shard_dir / "complete.csv", rows, list(rows[0]))
    return root, shard_dir


def test_analyze_pilot_validates_complete_configuration_matrix(tmp_path: Path):
    root, shard_dir = _pilot(tmp_path)
    metrics = analyze_pilot(shard_dir, root)
    assert metrics["successful_rows"] == 8
    assert metrics["failed_rows"] == 0


def test_analyze_pilot_rejects_missing_configuration(tmp_path: Path):
    root, shard_dir = _pilot(tmp_path)
    path = shard_dir / "complete.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        analyze_pilot(shard_dir, root)
