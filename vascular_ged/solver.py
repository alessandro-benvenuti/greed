from __future__ import annotations

import csv
import math
import os
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .atomic import write_csv
from .graph import load_vtp

RESULT_FIELDS = (
    "solver", "solver_version", "gurobi_version", "gedlib_version", "time_limit_seconds", "runtime_seconds", "peak_memory_bytes", "lower_bound", "upper_bound",
    "absolute_bound_gap", "relative_bound_gap", "is_exact", "solver_status", "failure_message",
    "git_commit", "slurm_job_id", "slurm_array_task_id", "computed_at",
)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def solve_row(row: dict[str, str], dataset_root: Path, cfg: dict, method: str, timeout: int) -> dict[str, object]:
    started = time.perf_counter()
    output: dict[str, object] = dict(row)
    output["cost_config_id"] = cfg["id"]
    try:
        g = load_vtp(dataset_root / row["graph_a_relative_path"])
        h = load_vtp(dataset_root / row["graph_b_relative_path"])
        if not g.num_nodes or not h.num_nodes:
            nonempty = h if not g.num_nodes else g
            value = nonempty.num_nodes * cfg["node_insertion_deletion"] + nonempty.num_edges * cfg["edge_insertion_deletion"]
            lb = ub = float(value)
            status, solver_version, gurobi_version, gedlib_version = "analytic_empty_graph", "analytic", "n/a", "n/a"
        else:
            import pyged  # type: ignore
            method_args = f"--threads 1 --time-limit {timeout}" if method == "f2" else "--threads 1"
            result = pyged.geometric_ged(
                (g.coordinates.tolist(), g.edges.tolist()), (h.coordinates.tolist(), h.edges.tolist()),
                method, method_args, float(cfg["coordinate_scale"]),
                float(cfg["node_insertion_deletion"]), float(cfg["edge_insertion_deletion"]),
            )
            lb, ub = map(float, result)
            status = "ok"
            gurobi_version = getattr(pyged, "solver_version", lambda: "unknown")()
            gedlib_version = getattr(pyged, "gedlib_version", lambda: "v1.0")()
            solver_version = gedlib_version
        if not (math.isfinite(lb) and math.isfinite(ub) and lb <= ub):
            raise ValueError(f"invalid bounds [{lb}, {ub}]")
        gap = ub - lb
        output.update(lower_bound=lb, upper_bound=ub, absolute_bound_gap=gap,
                      relative_bound_gap=(gap / ub if ub else 0.0), is_exact=(gap <= 1e-9),
                      solver_status=status, failure_message="", solver_version=solver_version,
                      gurobi_version=gurobi_version, gedlib_version=gedlib_version)
    except Exception as exc:
        output.update(lower_bound="", upper_bound="", absolute_bound_gap="", relative_bound_gap="", is_exact=False,
                      solver_status="failed", failure_message=f"{type(exc).__name__}: {exc}", solver_version="unknown",
                      gurobi_version="unknown", gedlib_version="v1.0+gurobi-bound-patch")
    output.update(solver=method, time_limit_seconds=timeout, runtime_seconds=time.perf_counter() - started,
                  git_commit=_git_commit(), slurm_job_id=os.getenv("SLURM_JOB_ID", ""),
                  slurm_array_task_id=os.getenv("SLURM_ARRAY_TASK_ID", ""),
                  computed_at=datetime.now(timezone.utc).isoformat())
    output["peak_memory_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return output


def _valid_reusable_row(
    row: dict[str, str], cost_config_id: str, method: str, timeout: int,
) -> bool:
    if (
        row.get("cost_config_id") != cost_config_id
        or row.get("solver") != method
        or row.get("time_limit_seconds") != str(timeout)
        or row.get("solver_status") not in {"ok", "analytic_empty_graph"}
        or not all(field in row for field in RESULT_FIELDS)
    ):
        return False
    try:
        lower, upper = float(row["lower_bound"]), float(row["upper_bound"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(row.get("pair_id")) and math.isfinite(lower) and math.isfinite(upper) and lower <= upper


def _reusable_results_from_paths(
    paths: list[Path], cost_config_id: str, method: str, timeout: int,
) -> dict[str, dict[str, str]]:
    reusable: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not _valid_reusable_row(row, cost_config_id, method, timeout):
                    continue
                pair_id = row["pair_id"]
                if pair_id in reusable:
                    raise ValueError(f"duplicate reusable result for {pair_id}")
                reusable[pair_id] = row
    return reusable


def _reusable_results(
    directory: str | Path | None, cost_config_id: str, method: str, timeout: int,
) -> dict[str, dict[str, str]]:
    paths = [] if directory is None else sorted(Path(directory).glob("*.csv"))
    return _reusable_results_from_paths(paths, cost_config_id, method, timeout)


def run_shard(manifest: str | Path, output: str | Path, dataset_root: str | Path, config: str | Path,
              start: int, stop: int, method: str, timeout: int,
              reuse_dir: str | Path | None = None) -> None:
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[start:stop]
    cfg = yaml.safe_load(Path(config).read_text())
    reusable = _reusable_results(reuse_dir, cfg["id"], method, timeout)
    output_path = Path(output)
    if output_path.is_file():
        # A failed shard may still contain valid expensive results. Preserve
        # those rows and recompute only its failed or malformed entries.
        previous = _reusable_results_from_paths([output_path], cfg["id"], method, timeout)
        for pair_id, result in previous.items():
            reusable.setdefault(pair_id, result)
    solved = []
    for row in rows:
        cached = reusable.get(row["pair_id"])
        if cached is None:
            solved.append(solve_row(row, Path(dataset_root), cfg, method, timeout))
            continue
        reused: dict[str, object] = dict(row)
        reused["cost_config_id"] = cfg["id"]
        reused.update({field: cached[field] for field in RESULT_FIELDS})
        solved.append(reused)
    fields = list(rows[0]) + [field for field in RESULT_FIELDS if field not in rows[0]] if rows else list(RESULT_FIELDS)
    write_csv(output_path, solved, fields)
    failures = [row for row in solved if row.get("solver_status") == "failed"]
    if failures:
        raise RuntimeError(
            f"shard contains {len(failures)} failed solver row(s); results were saved for a resumable retry"
        )
