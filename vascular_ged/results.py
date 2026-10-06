from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

from .atomic import write_csv


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_results(manifest_path: str | Path, shard_dir: str | Path, output: str | Path,
                  dataset_root: str | Path, expected_count: int) -> dict[str, object]:
    manifest = _read(Path(manifest_path))
    if len(manifest) != expected_count:
        raise ValueError(f"manifest has {len(manifest)} rows, expected {expected_count}")
    ids = {row["pair_id"] for row in manifest}
    if len(ids) != expected_count:
        raise ValueError("manifest pair IDs are not unique")
    pair_keys = {tuple(sorted((row["sample_a"], row["sample_b"]))) for row in manifest}
    if len(pair_keys) != expected_count:
        raise ValueError("manifest contains duplicate unordered graph pairs")
    split_by_patient: dict[str, str] = {}
    root = Path(dataset_root)
    for row in manifest:
        for suffix in ("a", "b"):
            patient, split = row[f"patient_{suffix}"], row["greed_split"]
            if patient in split_by_patient and split_by_patient[patient] != split:
                raise ValueError(f"patient {patient} crosses GREED partitions")
            split_by_patient[patient] = split
            if not (root / row[f"graph_{suffix}_relative_path"]).is_file():
                raise FileNotFoundError(row[f"graph_{suffix}_relative_path"])
    results: dict[str, dict[str, str]] = {}
    for shard in sorted(Path(shard_dir).glob("*.csv")):
        for row in _read(shard):
            pair_id = row["pair_id"]
            if pair_id in results:
                raise ValueError(f"duplicate result for {pair_id}")
            results[pair_id] = row
    if set(results) != ids:
        missing, extra = ids - set(results), set(results) - ids
        raise ValueError(f"result/manifest mismatch: {len(missing)} missing, {len(extra)} extra")
    merged = []
    exact = failures = 0
    cost_configs: set[str] = set()
    solver_configs: set[tuple[str, str]] = set()
    for item in manifest:
        row = results[item["pair_id"]]
        if row["solver_status"] == "failed":
            failures += 1
            raise ValueError(f"failed solver row {row['pair_id']}: {row['failure_message']}")
        lb, ub = float(row["lower_bound"]), float(row["upper_bound"])
        if not (math.isfinite(lb) and math.isfinite(ub) and lb <= ub):
            raise ValueError(f"invalid bounds for {row['pair_id']}")
        is_exact = row["is_exact"].lower() in {"true", "1"}
        if is_exact and not math.isclose(lb, ub, abs_tol=1e-9):
            raise ValueError(f"exact row has unequal bounds: {row['pair_id']}")
        exact += is_exact
        cost_configs.add(row["cost_config_id"])
        solver_configs.add((row["solver"], row["time_limit_seconds"]))
        merged.append(row)
    if len(cost_configs) != 1 or len(solver_configs) != 1:
        raise ValueError(f"inconsistent production configurations: costs={cost_configs}, solvers={solver_configs}")
    fields = list(merged[0])
    write_csv(output, merged, fields)
    digest = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    return {"rows": len(merged), "exact": exact, "bounded": len(merged) - exact, "failures": failures, "sha256": digest}


def combine_partitions(train_path: str | Path, validation_path: str | Path, output: str | Path,
                       expected_train: int = 8000, expected_validation: int = 2000) -> dict[str, object]:
    train, validation = _read(Path(train_path)), _read(Path(validation_path))
    if len(train) != expected_train or len(validation) != expected_validation:
        raise ValueError(f"expected {expected_train}/{expected_validation} rows, got {len(train)}/{len(validation)}")
    if any(row["greed_split"] != "train" for row in train) or any(row["greed_split"] != "validation" for row in validation):
        raise ValueError("label file contains rows from the wrong GREED partition")
    combined = train + validation
    if len({row["pair_id"] for row in combined}) != len(combined):
        raise ValueError("pair IDs overlap across label partitions")
    train_patients = {row[key] for row in train for key in ("patient_a", "patient_b")}
    validation_patients = {row[key] for row in validation for key in ("patient_a", "patient_b")}
    if train_patients & validation_patients:
        raise ValueError("patients overlap across GREED label partitions")
    write_csv(output, combined, list(combined[0]))
    return {"rows": len(combined), "sha256": hashlib.sha256(Path(output).read_bytes()).hexdigest()}
