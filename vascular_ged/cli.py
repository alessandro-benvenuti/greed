from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .atomic import write_json
from .audit import audit, write_audit
from .metadata import load_patch_index
from .results import combine_partitions, merge_results
from .reporting import analyze_pilot, label_summary
from .sampling import MANIFEST_FIELDS, sample_pairs, write_manifest
from .solver import run_shard
from .split import patient_split, read_patient_split, write_patient_split
from .training import train


def _dataset(value: str | None) -> Path:
    raw = value or os.getenv("SYNTHETIC_MRI_DATASET")
    if not raw:
        raise ValueError("pass --dataset-root or set SYNTHETIC_MRI_DATASET")
    return Path(raw)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vascular-ged")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "split", "sample"):
        p = sub.add_parser(name); p.add_argument("--dataset-root"); p.add_argument("--output-root", required=True)
        p.add_argument("--patch-index")
    sub.choices["audit"].add_argument("--workers", type=int, default=1)
    p = sub.choices["split"]; p.add_argument("--seed", type=int, default=314159)
    p = sub.choices["sample"]; p.add_argument("--split-file"); p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--train-count", type=int, default=8000); p.add_argument("--validation-count", type=int, default=2000)
    p.add_argument("--dataset-id", default="syntheticMRI_new_patches_boundary"); p.add_argument("--cost-config-id", default="vascular_3d_raw_v1")
    p = sub.add_parser("run-shard")
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True); p.add_argument("--dataset-root")
    p.add_argument("--config", required=True); p.add_argument("--start", type=int, required=True); p.add_argument("--stop", type=int, required=True)
    p.add_argument("--method", choices=("f2", "branch"), default="f2"); p.add_argument("--timeout", type=int, default=300)
    p = sub.add_parser("merge")
    p.add_argument("--manifest", required=True); p.add_argument("--shard-dir", required=True); p.add_argument("--output", required=True)
    p.add_argument("--dataset-root"); p.add_argument("--expected-count", type=int, required=True)
    p = sub.add_parser("combine")
    p.add_argument("--train", required=True); p.add_argument("--validation", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("train"); p.add_argument("--config", required=True)
    p = sub.add_parser("analyze-pilot"); p.add_argument("--shard-dir", required=True); p.add_argument("--output-root", required=True)
    p = sub.add_parser("label-summary"); p.add_argument("--labels", nargs="+", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "train": train(args.config); return
    if args.command == "analyze-pilot": analyze_pilot(args.shard_dir, args.output_root); return
    if args.command == "label-summary": label_summary(args.labels, args.output); return
    if args.command == "combine": print(json.dumps(combine_partitions(args.train, args.validation, args.output), indent=2)); return
    dataset = _dataset(getattr(args, "dataset_root", None))
    if args.command == "run-shard":
        run_shard(args.manifest, args.output, dataset, args.config, args.start, args.stop, args.method, args.timeout); return
    if args.command == "merge":
        print(json.dumps(merge_results(args.manifest, args.shard_dir, args.output, dataset, args.expected_count), indent=2)); return
    output = Path(args.output_root); patch_index = Path(args.patch_index) if args.patch_index else dataset / "patch_index.csv"
    records = load_patch_index(patch_index)
    if args.command == "audit": write_audit(output, audit(records, dataset, patch_index, args.workers)); return
    if args.command == "split": write_patient_split(output / "metadata/patient_split.csv", patient_split(records, args.seed), args.seed); return
    split_file = Path(args.split_file) if args.split_file else output / "metadata/patient_split.csv"
    assignment, _ = read_patient_split(split_file)
    record_patients = {record.patient_id for record in records}
    if set(assignment) != record_patients:
        missing = record_patients - set(assignment)
        extra = set(assignment) - record_patients
        raise ValueError(f"patient split mismatch: {len(missing)} missing, {len(extra)} extra")
    train_rows, train_avail = sample_pairs(records, assignment, "train", args.train_count, args.seed, args.dataset_id, args.cost_config_id)
    val_rows, val_avail = sample_pairs(records, assignment, "validation", args.validation_count, args.seed, args.dataset_id, args.cost_config_id)
    write_manifest(output / "manifests/greed_train_pairs.csv", train_rows)
    write_manifest(output / "manifests/greed_validation_pairs.csv", val_rows)
    write_manifest(output / "manifests/all_pairs.csv", train_rows + val_rows)
    # Two examples from every strategy/stratum combination.
    pilot = []
    for strategy in ("corresponding_region", "size_matched", "unrestricted_random"):
        for stratum in ("small", "medium", "large"):
            pilot.extend([r for r in train_rows + val_rows if r["sampling_strategy"] == strategy and r["size_stratum"] == stratum][:2])
    write_manifest(output / "manifests/pilot_pairs.csv", pilot)
    availability = {
        "sampling_seed": args.seed, "train_requested": args.train_count,
        "validation_requested": args.validation_count, "train": train_avail,
        "validation": val_avail,
    }
    write_json(output / "metadata/pair_sampling_availability.json", availability)
    print(json.dumps(availability, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
