"""
Test minmax solver function
"""
from numpy.testing import assert_equal

from preferendus.weighted_minmax.algorithm import aggregate_max


def test_solver_minmax():
    w = [0.8, 0.2]
    p1 = [50, 80, 80]
    p2 = [50, 80, 80]

    p = [p1, p2]

    res = aggregate_max(w, p, 100)
    expected_res = [
        max(w[0] * (100 - p[0][0]), w[1] * (100 - p[1][0])),
        max(w[0] * (100 - p[0][1]), w[1] * (100 - p[1][1])),
        max(w[0] * (100 - p[0][2]), w[1] * (100 - p[1][2])),
    ]

    assert_equal(res, expected_res)
    return


if __name__ == "__main__":
    test_solver_minmax()
