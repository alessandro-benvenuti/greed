from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections import defaultdict
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


def _audit_graph(
    task: tuple[str, GraphRecord],
) -> tuple[str, dict[str, int], int, int, int, int, list[float] | None, list[float] | None]:
    root_text, record = task
    root = Path(root_text)
    train_root = (root / "train/vtp").resolve()
    path = (root / record.relative_path).resolve()
    if train_root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"graph path does not resolve below train/vtp: {record.relative_path}")
    coords, raw_edges = read_vtp_raw(path)
    counts = {
        "non_finite_coordinates": int((~np.isfinite(coords)).sum()),
        "invalid_endpoints": 0,
        "self_loops": 0,
        "duplicate_edges": 0,
        "node_count_mismatches": 0,
        "edge_count_mismatches": 0,
        "actual_empty_graphs": 0,
    }
    if len(raw_edges):
        valid_endpoint = ((raw_edges >= 0) & (raw_edges < len(coords))).all(axis=1)
        counts["invalid_endpoints"] = int((~valid_endpoint).sum())
        valid_edges = raw_edges[valid_endpoint]
        counts["self_loops"] = int((valid_edges[:, 0] == valid_edges[:, 1]).sum())
        normalized = [tuple(sorted(map(int, edge))) for edge in valid_edges if edge[0] != edge[1]]
        counts["duplicate_edges"] = len(normalized) - len(set(normalized))
    else:
        valid_edges = raw_edges
    finite_rows = coords[np.isfinite(coords).all(axis=1)]
    coordinate_min = finite_rows.min(axis=0).tolist() if len(finite_rows) else None
    coordinate_max = finite_rows.max(axis=0).tolist() if len(finite_rows) else None
    # Coordinates do not affect topology; replace non-finite values only so the
    # audit can report all malformed files instead of stopping at the first one.
    graph = Graph3D(np.nan_to_num(coords), valid_edges)
    counts["node_count_mismatches"] = int(graph.num_nodes != record.node_count)
    counts["edge_count_mismatches"] = int(graph.num_edges != record.edge_count)
    counts["actual_empty_graphs"] = int(graph.num_nodes == 0)
    components = _components(graph.num_nodes, graph.edges) if graph.num_nodes else 0
    beta1 = graph.num_edges - graph.num_nodes + components
    return (
        record.sample_id, counts, graph.num_nodes, graph.num_edges,
        components, beta1, coordinate_min, coordinate_max,
    )


def audit(
    records: list[GraphRecord], dataset_root: str | Path, patch_index: str | Path, workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(dataset_root).resolve()
    metadata_path = Path(patch_index)
    topology = {
        "non_finite_coordinates": 0, "invalid_endpoints": 0, "self_loops": 0,
        "duplicate_edges": 0, "node_count_mismatches": 0,
        "edge_count_mismatches": 0, "actual_empty_graphs": 0,
    }
    components, beta1 = [], []
    coordinate_min: np.ndarray | None = None
    coordinate_max: np.ndarray | None = None
    node_count_corrections: dict[str, int] = {}
    edge_count_corrections: dict[str, int] = {}
    records_by_sample = {record.sample_id: record for record in records}
    per_patient: dict[str, list[int]] = defaultdict(list)
    per_patch: dict[str, list[int]] = defaultdict(list)
    for record in records:
        per_patient[record.patient_id].append(record.size)
        per_patch[record.patch_index].append(record.size)
    tasks = ((str(root), record) for record in records)
    if workers == 1:
        results = map(_audit_graph, tasks)
        pool = None
    else:
        pool = multiprocessing.Pool(processes=min(workers, len(records)))
        results = pool.imap_unordered(_audit_graph, tasks, chunksize=64)
    try:
        for index, result in enumerate(results, start=1):
            sample_id, counts, actual_nodes, actual_edges, component_count, cycle_rank, graph_min, graph_max = result
            for key, value in counts.items():
                topology[key] += value
            record = records_by_sample[sample_id]
            if actual_nodes != record.node_count:
                node_count_corrections[sample_id] = actual_nodes
            if actual_edges != record.edge_count:
                edge_count_corrections[sample_id] = actual_edges
            components.append(component_count)
            beta1.append(cycle_rank)
            if graph_min is not None:
                values = np.asarray(graph_min)
                coordinate_min = values if coordinate_min is None else np.minimum(coordinate_min, values)
            if graph_max is not None:
                values = np.asarray(graph_max)
                coordinate_max = values if coordinate_max is None else np.maximum(coordinate_max, values)
            if index % 5000 == 0 or index == len(records):
                print(f"Audited {index}/{len(records)} training graphs", flush=True)
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    else:
        if pool is not None:
            pool.close()
    finally:
        if pool is not None:
            pool.join()
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
        "empty_graph_count": topology["actual_empty_graphs"],
        "empty_graph_percentage": 100 * topology["actual_empty_graphs"] / len(records),
        "connected_components": _summary(components), "cycle_rank_beta1": _summary(beta1),
        "coordinate_min_dhw": coordinate_min.tolist() if coordinate_min is not None else [None] * 3,
        "coordinate_max_dhw": coordinate_max.tolist() if coordinate_max is not None else [None] * 3,
        "generation_metadata": generation_metadata,
        "normalized_node_count_corrections": node_count_corrections,
        "normalized_edge_count_corrections": edge_count_corrections,
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
    for key in (
        "training_patch_count", "training_patient_count", "empty_graph_count", "empty_graph_percentage",
        "node_counts", "edge_counts", "connected_components", "cycle_rank_beta1",
        "coordinate_min_dhw", "coordinate_max_dhw", "non_finite_coordinates",
        "invalid_endpoints", "self_loops", "duplicate_edges", "node_count_mismatches",
        "edge_count_mismatches",
    ):
        lines.append(f"- {key}: `{report[key]}`")
    if report["generation_metadata"] is not None:
        lines.append(f"- generation_metadata_sha256: `{report['generation_metadata']['sha256']}`")
    lines += ["", "## Initial empty-graph policy", "",
              "Empty graphs are excluded from the first 10,000-pair experiment. Their prevalence is recorded above; "
              "the solver supports the analytic identity `GED(empty,G)=|V_G|+|E_G|`. The initial GIN therefore never receives an undefined empty graph."]
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    tmp = root / "metadata/.dataset_audit.md.tmp"
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(root / "metadata/dataset_audit.md")
