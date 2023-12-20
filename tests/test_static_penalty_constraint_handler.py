"""
Test to check if the static penalty constraint handler of the algorithm is functioning
correct.
"""
import numpy as np
from numpy.testing import assert_equal

from preferendus._constraints import _const_handler, K


def constraint_1(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    return x1 + x2  # < 0


def constraint_2(variables):
    x1 = variables[:, 0]
    x2 = variables[:, 1]

    return (x1 + x2) - 3  # == 0


def test_static_penalty_constraint_handler():
    handler = "simple"
    constraints = (("ineq", constraint_1), ("eq", constraint_2))
    decoded = np.array([[-2, 1], [2, 1], [3, 1]])

    scores = [100, 10, 20]
    score_result, non_feasible_counter = _const_handler(
        handler=handler, constraints=constraints, decoded=decoded, scores=scores
    )

    expected_results = [
        round(K - (2 - 1) * K / 2),
        round(K - (2 - 1) * K / 2),
        round(K - (2 - 2) * K / 2),
    ]

    assert non_feasible_counter == 3
    assert_equal(score_result, expected_results)


if __name__ == "__main__":
    test_static_penalty_constraint_handler()
