"""
Test to see if the crossover function is functioning correctly
"""
from collections import Counter

from preferendus._nextgen import _crossover


def test_crossover():
    """
    The crossover for integer and bools is a stochastic process, making testing
    non-deterministic. Instead, 100,000 runs are made and a threshold value is
    implemented.
    """
    r_cross = 1.0  # unrealistic high to cancel its influence
    type_of_variables = ["real", "int", "bool"]

    p1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 320, 0]
    p2 = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 452, 1]

    counter_bitstring = 0
    counter_int = 0
    counter_bool = 0

    n_iter = 100_000
    for _ in range(100_000):
        ret = _crossover(p1=p1, p2=p2, r_cross=r_cross, approach=type_of_variables)
        c = Counter(ret[0][0])
        if c["1"] == len(p1[0]):  # the bit strings should have changed as R_cross = 1
            counter_bitstring += 1

        if ret[0][1] == p2[1]:
            counter_int += 1
        if ret[0][2] == p2[2]:
            counter_bool += 1

    assert counter_bitstring == 0
    assert 0.45 * n_iter < counter_int < 0.55 * n_iter  # chance of mutation = 0.5
    assert 0.45 * n_iter < counter_bool < 0.55 * n_iter  # chance of mutation = 0.5
    return


if __name__ == "__main__":
    test_crossover()
