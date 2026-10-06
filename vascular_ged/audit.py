from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .atomic import write_json
from .graph import Graph3D, read_vtp_raw
from .metadata import GraphRecord


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values), "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "q25": float(np.quantile(array, .25)),
        "median": float(np.quantile(array, .5)), "q75": float(np.quantile(array, .75)),
        "q95": float(np.quantile(array, .95)),
    }


def _components(n: int, edges: np.ndarray) -> int:
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)})


def audit(records: list[GraphRecord], dataset_root: str | Path, patch_index: str | Path) -> dict[str, object]:
    root = Path(dataset_root).resolve()
    metadata_path = Path(patch_index)
    topology = {"non_finite_coordinates": 0, "invalid_endpoints": 0, "self_loops": 0, "duplicate_edges": 0}
    components, beta1, coordinate_rows = [], [], []
    per_patient: dict[str, list[int]] = defaultdict(list)
    per_patch: dict[str, list[int]] = defaultdict(list)
    for record in records:
        path = (root / record.relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"graph path does not resolve below dataset root: {record.relative_path}")
        coords, raw_edges = read_vtp_raw(path)
        topology["non_finite_coordinates"] += int((~np.isfinite(coords)).sum())
        valid_endpoint = np.ones(len(raw_edges), dtype=bool)
        if len(raw_edges):
            valid_endpoint = ((raw_edges >= 0) & (raw_edges < len(coords))).all(axis=1)
            topology["invalid_endpoints"] += int((~valid_endpoint).sum())
            valid_edges = raw_edges[valid_endpoint]
            topology["self_loops"] += int((valid_edges[:, 0] == valid_edges[:, 1]).sum())
            normalized = [tuple(sorted(map(int, edge))) for edge in valid_edges if edge[0] != edge[1]]
            topology["duplicate_edges"] += len(normalized) - len(set(normalized))
        else:
            valid_edges = raw_edges
        finite_rows = coords[np.isfinite(coords).all(axis=1)]
        coordinate_rows.append(finite_rows)
        # Continue the audit even when malformed entries are found; invalid rows
        # are reported and omitted from topology summaries.
        graph = Graph3D(np.nan_to_num(coords), valid_edges)
        c = _components(graph.num_nodes, graph.edges) if graph.num_nodes else 0
        components.append(c)
        beta1.append(graph.num_edges - graph.num_nodes + c)
        per_patient[record.patient_id].append(record.size)
        per_patch[record.patch_index].append(record.size)
    coords = np.concatenate(coordinate_rows) if coordinate_rows and sum(map(len, coordinate_rows)) else np.empty((0, 3))
    generation_path = root / "generation_summary.json"
    generation_metadata = None
    if generation_path.is_file():
        generation_metadata = {
            "sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
            "summary": json.loads(generation_path.read_text(encoding="utf-8")),
        }
    return {
        "dataset_root": str(root),
        "patch_index_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "training_patch_count": len(records), "training_patient_count": len({r.patient_id for r in records}),
        "node_counts": _summary([r.node_count for r in records]), "edge_counts": _summary([r.edge_count for r in records]),
        "empty_graph_count": sum(r.node_count == 0 for r in records),
        "empty_graph_percentage": 100 * sum(r.node_count == 0 for r in records) / len(records),
        "connected_components": _summary(components), "cycle_rank_beta1": _summary(beta1),
        "coordinate_min_dhw": coords.min(axis=0).tolist() if len(coords) else [None] * 3,
        "coordinate_max_dhw": coords.max(axis=0).tolist() if len(coords) else [None] * 3,
        "generation_metadata": generation_metadata,
        **topology,
        "graph_size_by_patient": {key: _summary(value) for key, value in sorted(per_patient.items())},
        "graph_size_by_patch_index": {key: _summary(value) for key, value in sorted(per_patch.items())},
    }


def write_audit(root: str | Path, report: dict[str, object]) -> None:
    root = Path(root)
    write_json(root / "metadata/dataset_audit.json", report)
    write_json(root / "metadata/dataset_metadata.json", {
        key: report[key] for key in ("dataset_root", "patch_index_sha256", "training_patch_count", "training_patient_count")
    })
    lines = ["# SyntheticMRI training audit", "", f"Dataset root: `{report['dataset_root']}`", ""]
    for key in ("training_patch_count", "training_patient_count", "empty_graph_count", "empty_graph_percentage", "node_counts", "edge_counts", "connected_components", "cycle_rank_beta1", "coordinate_min_dhw", "coordinate_max_dhw"):
        lines.append(f"- {key}: `{report[key]}`")
    lines += ["", "## Initial empty-graph policy", "",
              "Empty graphs are excluded from the first 10,000-pair experiment. Their prevalence is recorded above; "
              "the solver supports the analytic identity `GED(empty,G)=|V_G|+|E_G|`. The initial GIN therefore never receives an undefined empty graph."]
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    tmp = root / "metadata/.dataset_audit.md.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(root / "metadata/dataset_audit.md")
