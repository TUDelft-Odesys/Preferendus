"""
Test to see if the mutation function is functioning correctly
"""
from preferendus._nextgen import _mutation


def test_mutation():
    """
    With the r_mut = 1 and the bounds this strict, only the int mutation is stochastic.
    There, a threshold value is introduced
    """
    r_mut = 1.0  # unrealistic high to cancel its influence
    type_of_variables = ["real", "int", "bool"]
    bounds = ((0, 7000), (1, 2), (0, 1))

    p = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0]

    counter_int = 0
    n_iter = 1000
    check_value = 1

    for _ in range(n_iter):
        _mutation(member=p, r_mut=r_mut, approach=type_of_variables, bounds=bounds)
        assert p[0].count(check_value) == len(p[0])  # all bits are flipped
        assert p[2] == check_value  # the bit is flipped
        if p[1] == 2:
            counter_int += 1
        check_value = 1 - check_value

    assert 0.45 * n_iter < counter_int < 0.55 * n_iter
    return


if __name__ == "__main__":
    test_mutation()
