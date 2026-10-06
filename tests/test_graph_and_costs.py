from pathlib import Path

import numpy as np
import pytest

from vascular_ged.costs import CostConfig, exact_geometric_ged
from vascular_ged.graph import Graph3D, load_vtp


def graph(points, edges=()):
    return Graph3D(np.asarray(points, dtype=float).reshape((-1, 3)), np.asarray(edges, dtype=int).reshape((-1, 2)))


def test_identical_permuted_and_symmetric():
    a = graph([[0, 0, 0], [.2, 0, 0]], [[0, 1]])
    b = graph([[.2, 0, 0], [0, 0, 0]], [[1, 0]])
    assert exact_geometric_ged(a, a) == 0
    assert exact_geometric_ged(a, b) == 0
    assert exact_geometric_ged(a, b) == exact_geometric_ged(b, a)


def test_known_node_and_edge_costs():
    origin = graph([[0, 0, 0]])
    moved = graph([[.1, .2, .2]])
    assert exact_geometric_ged(origin, moved) == pytest.approx(.3)
    assert exact_geometric_ged(origin, graph([])) == 1
    edge = graph([[0, 0, 0], [.2, 0, 0]], [[0, 1]])
    no_edge = graph([[0, 0, 0], [.2, 0, 0]])
    assert exact_geometric_ged(edge, no_edge) == 1


def test_depth_is_not_ignored_and_scaling_is_configurable():
    a = graph([[0, 0, 0]])
    b = graph([[.5, 0, 0]])
    assert exact_geometric_ged(a, b) == pytest.approx(.5)
    assert exact_geometric_ged(a, b, CostConfig(coordinate_scale=np.sqrt(3))) == pytest.approx(.5 / np.sqrt(3))


def test_triangle_inequality_tiny_triplet():
    graphs = [graph([[x, 0, 0]]) for x in (0, .2, .5)]
    ab, bc, ac = exact_geometric_ged(graphs[0], graphs[1]), exact_geometric_ged(graphs[1], graphs[2]), exact_geometric_ged(graphs[0], graphs[2])
    assert ac <= ab + bc + 1e-12


def test_validation_and_undirected_policy():
    with pytest.raises(ValueError, match="non-finite"):
        graph([[np.nan, 0, 0]])
    with pytest.raises(ValueError, match="endpoint"):
        graph([[0, 0, 0]], [[0, 1]])
    g = graph([[0, 0, 0], [1, 0, 0]], [[0, 0], [0, 1], [1, 0], [0, 1]])
    assert g.edges.tolist() == [[0, 1]]


def test_vtp_preserves_dhw_column_order():
    loaded = load_vtp(Path(__file__).parent / "fixtures/line_dhw.vtp")
    np.testing.assert_allclose(loaded.coordinates[0], [.1, .2, .3])
    assert loaded.edges.tolist() == [[0, 1], [1, 2]]
