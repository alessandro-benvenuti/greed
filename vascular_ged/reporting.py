from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .atomic import write_csv, write_json


def _rows(paths) -> list[dict[str, str]]:
    result = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            result.extend(csv.DictReader(handle))
    return result


def analyze_pilot(shard_dir: str | Path, report_root: str | Path) -> dict[str, object]:
    report_root = Path(report_root)
    all_rows = _rows(sorted(Path(shard_dir).glob("*.csv")))
    manifest = _rows([report_root / "manifests/pilot_pairs.csv"])
    pair_ids = [row["pair_id"] for row in manifest]
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pilot manifest is empty or contains duplicate pair IDs")
    expected = {
        (pair_id, cost, solver, timeout)
        for pair_id in pair_ids
        for cost in ("vascular_3d_raw_v1", "vascular_3d_sqrt3_v1")
        for solver, timeout in (("f2", 10), ("f2", 60), ("f2", 300), ("branch", 0))
    }
    actual_keys = [
        (row["pair_id"], row["cost_config_id"], row["solver"], int(row["time_limit_seconds"]))
        for row in all_rows
    ]
    if len(set(actual_keys)) != len(actual_keys):
        raise ValueError("pilot results contain duplicate pair/configuration rows")
    actual = set(actual_keys)
    if actual != expected:
        raise ValueError(f"pilot result mismatch: {len(expected - actual)} missing, {len(actual - expected)} extra")
    for row in all_rows:
        if row["solver_status"] == "failed":
            if not row["failure_message"]:
                raise ValueError(f"failed pilot row lacks a failure message: {row['pair_id']}")
            continue
        lower, upper = float(row["lower_bound"]), float(row["upper_bound"])
        runtime = float(row["runtime_seconds"])
        absolute_gap = float(row["absolute_bound_gap"])
        relative_gap = float(row["relative_bound_gap"])
        if not all(math.isfinite(value) for value in (lower, upper, runtime, absolute_gap, relative_gap)):
            raise ValueError(f"pilot row contains non-finite values: {row['pair_id']}")
        if lower > upper or runtime < 0 or not math.isclose(absolute_gap, upper - lower, abs_tol=1e-9):
            raise ValueError(f"pilot row contains inconsistent bounds or runtime: {row['pair_id']}")
        expected_relative = absolute_gap / upper if upper else 0.0
        if not math.isclose(relative_gap, expected_relative, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"pilot row contains an inconsistent relative gap: {row['pair_id']}")
        declared_exact = row["is_exact"].lower() in {"true", "1"}
        if declared_exact != (absolute_gap <= 1e-9):
            raise ValueError(f"pilot row contains an inconsistent exact flag: {row['pair_id']}")
    rows = [row for row in all_rows if row["solver_status"] != "failed"]
    if not rows:
        raise ValueError("pilot has no successful rows")
    groups: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["cost_config_id"], row["solver"], int(row["time_limit_seconds"]))].append(row)
    configurations = {}
    for key, group in sorted(groups.items()):
        runtimes = np.asarray([float(r["runtime_seconds"]) for r in group])
        gaps = np.asarray([float(r["relative_bound_gap"]) for r in group])
        uppers = np.asarray([float(r["upper_bound"]) for r in group])
        sizes = np.asarray([int(r["node_count_a"]) + int(r["edge_count_a"]) + int(r["node_count_b"]) + int(r["edge_count_b"]) for r in group])
        similarities = {str(lam): {"min": float(np.exp(-lam * uppers).min()), "median": float(np.median(np.exp(-lam * uppers))), "max": float(np.exp(-lam * uppers).max())} for lam in (.01, .05, .1, .2, 1.0)}
        configurations["|".join(map(str, key))] = {
            "rows": len(group), "exact_fraction": float(np.mean(gaps <= 1e-9)),
            "relative_gap_median": float(np.median(gaps)), "relative_gap_max": float(gaps.max()),
            "runtime_median_seconds": float(np.median(runtimes)), "runtime_max_seconds": float(runtimes.max()),
            "estimated_cpu_hours_10000_from_mean": float(runtimes.mean() * 10000 / 3600),
            "ged_size_correlation": float(np.corrcoef(uppers, sizes)[0, 1]) if len(group) > 1 else None,
            "similarity_exp_minus_lambda_ged": similarities,
        }
    selection = {
        "cost_config_id": "vascular_3d_raw_v1", "coordinate_scale": 1.0,
        "solver": "f2", "timeout_seconds": 60, "threads_per_pair": 1,
        "chunk_size": 25, "array_concurrency": 8, "bounded_labels": True,
        "include_empty_graphs": False, "similarity_lambda": 0.05,
        "pilot_exact_fraction": configurations["vascular_3d_raw_v1|f2|60"]["exact_fraction"],
        "pilot_estimated_cpu_hours_10000": configurations["vascular_3d_raw_v1|f2|60"]["estimated_cpu_hours_10000_from_mean"],
        "worst_case_cpu_hours": 10000 * 60 / 3600,
        "rationale": "Raw/60 matched raw/300 exactness while using substantially less CPU; interval labels preserve non-exact bounds.",
    }
    metrics = {
        "successful_rows": len(rows), "failed_rows": len(all_rows) - len(rows),
        "configurations": configurations, "selected_production_configuration": selection,
        "production_worst_case_cpu_hours_at_60_seconds": 10000 * 60 / 3600,
    }
    (report_root / "reports").mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "results/pilot/pilot_ged_results.csv", all_rows, list(all_rows[0]))
    write_json(report_root / "reports/solver_scaling_metrics.json", metrics)
    lines = ["# Solver scaling report", "", "This report compares F2/BRANCH, coordinate scaling, and 10/60/300-second limits.", "",
             f"Successful runs: {metrics['successful_rows']}; failed runs: {metrics['failed_rows']}.", "",
             "The selected 60-second hard ceiling implies 166.67 CPU-hours for 10,000 pairs.", "", "## Configurations", ""]
    for key, value in configurations.items():
        lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True)}")
    lines += ["", "Production must not be launched until environment, licence, C++ build, stock F2, handcrafted tests, and finite-bound guardrails all pass."]
    lines += ["", "## Selected production configuration", "", json.dumps(selection, indent=2, sort_keys=True)]
    (report_root / "reports/solver_scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def label_summary(label_paths: list[str | Path], output: str | Path) -> None:
    rows = _rows(label_paths)
    if not rows:
        raise ValueError("no labels")
    lower = np.asarray([float(row["lower_bound"]) for row in rows]); upper = np.asarray([float(row["upper_bound"]) for row in rows])
    runtime = np.asarray([float(row["runtime_seconds"]) for row in rows]); exact = np.isclose(lower, upper)
    by_strategy = {key: sum(row["sampling_strategy"] == key for row in rows) for key in sorted({row["sampling_strategy"] for row in rows})}
    by_stratum = {key: sum(row["size_stratum"] == key for row in rows) for key in sorted({row["size_stratum"] for row in rows})}
    hashes = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in label_paths}
    lines = ["# Geometric GED label dataset", "", f"Rows: {len(rows)}", f"Exact: {int(exact.sum())}; bounded: {int((~exact).sum())}",
             f"Failures/timeouts retained: {sum(row['solver_status'] == 'failed' for row in rows)}",
             f"Lower-bound quantiles: {np.quantile(lower, [0,.25,.5,.75,1]).tolist()}",
             f"Upper-bound quantiles: {np.quantile(upper, [0,.25,.5,.75,1]).tolist()}",
             f"Runtime quantiles (seconds): {np.quantile(runtime, [0,.25,.5,.75,1]).tolist()}",
             f"Total CPU hours: {runtime.sum()/3600:.6f}", f"Distribution by strategy: `{by_strategy}`",
             f"Distribution by size stratum: `{by_stratum}`", f"File SHA-256: `{hashes}`", ""]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(lines), encoding="utf-8")
