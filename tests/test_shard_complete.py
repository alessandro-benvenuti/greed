import importlib.util
from pathlib import Path

from vascular_ged.atomic import write_csv

SCRIPT = Path(__file__).parents[1] / "scripts/jean_zay/shard_complete.py"
SPEC = importlib.util.spec_from_file_location("shard_complete", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
shard_is_complete = MODULE.shard_is_complete


FIELDS = (
    "pair_id", "solver_status", "cost_config_id", "solver", "time_limit_seconds",
    "lower_bound", "upper_bound",
)


def _write(path: Path, status: str = "ok") -> None:
    write_csv(path, [{
        "pair_id": "pair", "solver_status": status,
        "cost_config_id": "cost", "solver": "f2", "time_limit_seconds": 10,
        "lower_bound": 1, "upper_bound": 1,
    }], FIELDS)


def test_complete_shard_requires_successful_matching_rows(tmp_path: Path):
    path = tmp_path / "shard.csv"
    assert not shard_is_complete(path, 1, "cost", "f2", 10)
    _write(path)
    assert shard_is_complete(path, 1, "cost", "f2", 10)
    assert not shard_is_complete(path, 1, "other", "f2", 10)
    assert not shard_is_complete(path, 1, "cost", "branch", 10)
    assert not shard_is_complete(path, 1, "cost", "f2", 60)
    _write(path, "failed")
    assert not shard_is_complete(path, 1, "cost", "f2", 10)
