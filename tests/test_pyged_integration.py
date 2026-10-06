import math

import pytest

pyged = pytest.importorskip("pyged")


def solve(points_a, edges_a, points_b, edges_b):
    lower, upper = pyged.geometric_ged(
        (points_a, edges_a), (points_b, edges_b), "f2",
        "--time-limit 10 --threads 1 --map-root-to-root FALSE", 1.0, 1.0, 1.0,
    )
    assert lower == pytest.approx(upper)
    return upper


def test_compiled_geometric_ged_correctness_contract():
    line = ([[0, 0, 0], [.2, 0, 0]], [[0, 1]])
    permuted = ([[.2, 0, 0], [0, 0, 0]], [[1, 0]])
    assert solve(*line, *line) == pytest.approx(0)
    assert solve(*line, *permuted) == pytest.approx(0)
    assert solve([[0, 0, 0]], [], [[.1, .2, .2]], []) == pytest.approx(.3)
    assert solve(*line, [[0, 0, 0], [.2, 0, 0]], []) == pytest.approx(1)
    assert solve(*line, *permuted) == solve(*permuted, *line)
    assert solve([[0, 0, 0]], [], [[.5, 0, 0]], []) == pytest.approx(.5)
    a, b, c = ([[0, 0, 0]], []), ([[.2, 0, 0]], []), ([[.5, 0, 0]], [])
    assert solve(*a, *c) <= solve(*a, *b) + solve(*b, *c) + 1e-9


def test_compiled_wrapper_rejects_invalid_data_and_normalizes_edges():
    with pytest.raises(Exception, match="non-finite"):
        solve([[math.nan, 0, 0]], [], [[0, 0, 0]], [])
    with pytest.raises(Exception, match="endpoint"):
        solve([[0, 0, 0]], [[0, 1]], [[0, 0, 0]], [])
    duplicate = ([[0, 0, 0], [.2, 0, 0]], [[0, 0], [0, 1], [1, 0], [0, 1]])
    simple = ([[0, 0, 0], [.2, 0, 0]], [[0, 1]])
    assert solve(*duplicate, *simple) == pytest.approx(0)
