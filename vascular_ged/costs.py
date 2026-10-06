from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np

from .graph import Graph3D


@dataclass(frozen=True)
class CostConfig:
    id: str = "vascular_3d_raw_v1"
    coordinate_scale: float = 1.0
    node_insertion: float = 1.0
    node_deletion: float = 1.0
    edge_insertion: float = 1.0
    edge_deletion: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.coordinate_scale) or self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        for name in ("node_insertion", "node_deletion", "edge_insertion", "edge_deletion"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def mapping_cost(g: Graph3D, h: Graph3D, mapping: dict[int, int], cfg: CostConfig) -> float:
    cost = (g.num_nodes - len(mapping)) * cfg.node_deletion
    cost += (h.num_nodes - len(mapping)) * cfg.node_insertion
    cost += sum(np.linalg.norm(g.coordinates[i] - h.coordinates[j]) / cfg.coordinate_scale for i, j in mapping.items())
    ge = {tuple(edge) for edge in g.edges.tolist()}
    he = {tuple(edge) for edge in h.edges.tolist()}
    mapped_edges: set[tuple[int, int]] = set()
    for u, v in ge:
        if u in mapping and v in mapping:
            mapped_edges.add(tuple(sorted((mapping[u], mapping[v]))))
        else:
            cost += cfg.edge_deletion
    cost += len(mapped_edges - he) * cfg.edge_deletion
    cost += len(he - mapped_edges) * cfg.edge_insertion
    return float(cost)


def exact_geometric_ged(g: Graph3D, h: Graph3D, cfg: CostConfig = CostConfig(), max_nodes: int = 8) -> float:
    """Brute-force exact GED oracle for handcrafted tests, not production graphs."""
    if max(g.num_nodes, h.num_nodes) > max_nodes:
        raise ValueError(f"brute-force oracle is limited to {max_nodes} nodes")
    best = math.inf
    gn, hn = range(g.num_nodes), range(h.num_nodes)
    for size in range(min(g.num_nodes, h.num_nodes) + 1):
        for g_nodes in itertools.combinations(gn, size):
            for h_nodes in itertools.combinations(hn, size):
                for perm in itertools.permutations(h_nodes):
                    best = min(best, mapping_cost(g, h, dict(zip(g_nodes, perm)), cfg))
    return best
