from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Graph3D:
    """Undirected simple graph with node coordinates in canonical [D,H,W] order."""

    coordinates: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.coordinates, dtype=np.float64)
        edges = np.asarray(self.edges, dtype=np.int64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"coordinates must have shape [N,3], got {xyz.shape}")
        if not np.isfinite(xyz).all():
            raise ValueError("coordinates contain non-finite values")
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must have shape [E,2], got {edges.shape}")
        if edges.size and ((edges < 0).any() or (edges >= len(xyz)).any()):
            raise ValueError("edge endpoint is outside [0, number of nodes)")
        # Explicit policy: self-loops are dropped; duplicate and reverse duplicate
        # edges are collapsed. Vascular graphs are treated as undirected and simple.
        canonical = {tuple(sorted(map(int, edge))) for edge in edges if edge[0] != edge[1]}
        edges = np.asarray(sorted(canonical), dtype=np.int64).reshape((-1, 2))
        object.__setattr__(self, "coordinates", xyz)
        object.__setattr__(self, "edges", edges)

    @property
    def num_nodes(self) -> int:
        return len(self.coordinates)

    @property
    def num_edges(self) -> int:
        return len(self.edges)


def _numbers(data_array: ET.Element, dtype) -> np.ndarray:
    fmt = data_array.attrib.get("format", "ascii").lower()
    if fmt != "ascii":
        raise ValueError("built-in VTP reader supports ASCII arrays only; install pyvista for binary/appended VTP")
    return np.fromstring(data_array.text or "", sep=" ", dtype=dtype)


def _load_ascii_vtp_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"no VTK PolyData Piece in {path}")
    point_array = piece.find("./Points/DataArray")
    if point_array is None:
        raise ValueError(f"no point coordinates in {path}")
    coords = _numbers(point_array, np.float64).reshape((-1, 3))
    lines = piece.find("./Lines")
    raw_edges: list[tuple[int, int]] = []
    if lines is not None:
        arrays = lines.findall("./DataArray")
        named = {a.attrib.get("Name", "").lower(): a for a in arrays}
        conn_el = named.get("connectivity", arrays[0] if arrays else None)
        off_el = named.get("offsets", arrays[1] if len(arrays) > 1 else None)
        if conn_el is not None and off_el is not None:
            connectivity = _numbers(conn_el, np.int64)
            offsets = _numbers(off_el, np.int64)
            start = 0
            for end in offsets:
                line = connectivity[start:int(end)]
                raw_edges.extend((int(a), int(b)) for a, b in zip(line[:-1], line[1:]))
                start = int(end)
    return coords, np.asarray(raw_edges, dtype=np.int64).reshape((-1, 2))


def read_vtp_raw(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return stored D/H/W coordinates and raw polyline segments before normalization."""
    path = Path(path)
    try:
        import pyvista as pv  # type: ignore
    except ImportError:
        return _load_ascii_vtp_arrays(path)
    mesh = pv.read(path)
    edges: list[tuple[int, int]] = []
    lines = np.asarray(mesh.lines, dtype=np.int64)
    cursor = 0
    while cursor < len(lines):
        count = int(lines[cursor])
        ids = lines[cursor + 1 : cursor + 1 + count]
        edges.extend((int(a), int(b)) for a, b in zip(ids[:-1], ids[1:]))
        cursor += count + 1
    return np.asarray(mesh.points), np.asarray(edges, dtype=np.int64).reshape((-1, 2))


def load_vtp(path: str | Path) -> Graph3D:
    """Load a VTP graph without reordering its stored D/H/W point columns."""
    return Graph3D(*read_vtp_raw(path))
