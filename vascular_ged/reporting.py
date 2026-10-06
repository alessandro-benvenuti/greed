from __future__ import annotations

import csv
import hashlib
import json
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
    all_rows = _rows(sorted(Path(shard_dir).glob("*.csv")))
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
    metrics = {"successful_rows": len(rows), "failed_rows": len(_rows(sorted(Path(shard_dir).glob("*.csv")))) - len(rows), "configurations": configurations,
               "production_worst_case_cpu_hours_at_300_seconds": 10000 * 300 / 3600}
    report_root = Path(report_root); (report_root / "reports").mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "results/pilot/pilot_ged_results.csv", all_rows, list(all_rows[0]))
    write_json(report_root / "reports/solver_scaling_metrics.json", metrics)
    lines = ["# Solver scaling report", "", "This report compares F2/BRANCH, coordinate scaling, and 10/60/300-second limits.", "",
             f"Successful runs: {metrics['successful_rows']}; failed runs: {metrics['failed_rows']}.", "",
             "The 300-second hard ceiling implies 833.33 CPU-hours for 10,000 pairs.", "", "## Configurations", ""]
    for key, value in configurations.items():
        lines.append(f"- `{key}`: {json.dumps(value, sort_keys=True)}")
    lines += ["", "Production must not be launched until environment, licence, C++ build, stock F2, handcrafted tests, and finite-bound guardrails all pass."]
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
