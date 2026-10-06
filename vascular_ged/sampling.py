from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .atomic import write_csv
from .metadata import GraphRecord

MANIFEST_FIELDS = (
    "pair_id", "greed_split", "sampling_strategy", "size_stratum", "sample_a", "sample_b",
    "patient_a", "patient_b", "patch_index_a", "patch_index_b", "node_count_a", "node_count_b",
    "edge_count_a", "edge_count_b", "graph_a_relative_path", "graph_b_relative_path", "dataset_id",
    "cost_config_id", "sampling_seed",
)
STRATEGIES = ("corresponding_region", "size_matched", "unrestricted_random")
STRATA = ("small", "medium", "large")


def stable_pair_id(a: str, b: str, dataset_id: str, cost_config_id: str) -> str:
    first, second = sorted((a, b))
    return hashlib.sha256(f"{dataset_id}\0{cost_config_id}\0{first}\0{second}".encode()).hexdigest()[:24]


def assign_strata(records: list[GraphRecord]) -> dict[str, str]:
    values = np.asarray([r.size for r in records], dtype=float)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    return {r.sample_id: ("small" if r.size <= q1 else "medium" if r.size <= q2 else "large") for r in records}


def _quotas(total: int) -> dict[tuple[str, str], int]:
    cells = [(strategy, stratum) for strategy in STRATEGIES for stratum in STRATA]
    return {cell: total // len(cells) + (i < total % len(cells)) for i, cell in enumerate(cells)}


def sample_pairs(
    records: list[GraphRecord], assignment: dict[str, str], split: str, count: int, seed: int,
    dataset_id: str, cost_config_id: str, max_attempt_factor: int = 1000,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    selected = sorted((r for r in records if assignment.get(r.patient_id) == split and r.node_count > 0), key=lambda r: r.sample_id)
    if len(selected) < 2:
        raise ValueError(f"too few non-empty graphs in {split}")
    strata = assign_strata(selected)
    by_stratum: dict[str, list[GraphRecord]] = defaultdict(list)
    by_patch_stratum: dict[tuple[str, str], list[GraphRecord]] = defaultdict(list)
    by_size_stratum: dict[tuple[str, int], list[GraphRecord]] = defaultdict(list)
    for record in selected:
        stratum = strata[record.sample_id]
        by_stratum[stratum].append(record)
        by_patch_stratum[(stratum, record.patch_index)].append(record)
        # Log-size buckets make size matching scale across the long graph-size tail.
        bucket = int(np.floor(np.log2(record.size + 1)))
        by_size_stratum[(stratum, bucket)].append(record)
    rng = random.Random(f"{seed}:{split}")
    quotas = _quotas(count)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    availability: dict[str, int] = {}

    def candidates(strategy: str, stratum: str) -> list[list[GraphRecord]]:
        if strategy == "corresponding_region":
            return [group for (s, _), group in by_patch_stratum.items() if s == stratum and len({r.patient_id for r in group}) >= 2]
        if strategy == "size_matched":
            return [group for (s, _), group in by_size_stratum.items() if s == stratum and len({r.patient_id for r in group}) >= 2]
        return [by_stratum[stratum]]

    def different_patient_pairs(group: list[GraphRecord]) -> int:
        per_patient: dict[str, int] = defaultdict(int)
        for record in group:
            per_patient[record.patient_id] += 1
        return len(group) * (len(group) - 1) // 2 - sum(n * (n - 1) // 2 for n in per_patient.values())

    for (strategy, stratum), quota in quotas.items():
        groups = candidates(strategy, stratum)
        availability[f"{strategy}:{stratum}"] = sum(different_patient_pairs(group) for group in groups)
        produced = 0
        for _ in range(max(1, quota) * max_attempt_factor):
            if produced == quota:
                break
            group = rng.choice(groups) if groups else []
            if len(group) < 2:
                continue
            a, b = rng.sample(group, 2)
            if a.patient_id == b.patient_id:
                continue
            key = tuple(sorted((a.sample_id, b.sample_id)))
            if key in seen:
                continue
            seen.add(key)
            if key != (a.sample_id, b.sample_id):
                a, b = b, a
            row = {
                "pair_id": stable_pair_id(a.sample_id, b.sample_id, dataset_id, cost_config_id),
                "greed_split": split, "sampling_strategy": strategy, "size_stratum": stratum,
                "sample_a": a.sample_id, "sample_b": b.sample_id, "patient_a": a.patient_id,
                "patient_b": b.patient_id, "patch_index_a": a.patch_index, "patch_index_b": b.patch_index,
                "node_count_a": a.node_count, "node_count_b": b.node_count, "edge_count_a": a.edge_count,
                "edge_count_b": b.edge_count, "graph_a_relative_path": a.relative_path,
                "graph_b_relative_path": b.relative_path, "dataset_id": dataset_id,
                "cost_config_id": cost_config_id, "sampling_seed": seed,
            }
            rows.append(row)
            produced += 1
        if produced != quota:
            raise RuntimeError(f"could only produce {produced}/{quota} pairs for {strategy}:{stratum}")
    if len(rows) != count or len({row["pair_id"] for row in rows}) != count:
        raise AssertionError("pair sampler did not produce the requested unique count")
    return rows, availability


def write_manifest(path: str | Path, rows: list[dict[str, object]]) -> None:
    write_csv(path, rows, MANIFEST_FIELDS)
