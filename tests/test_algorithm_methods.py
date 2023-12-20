"""
Tests to check the functioning of the GeneticAlgorithm class
"""
from preferendus import Preferendus


def objective(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return [0.5, 0.5], [x1 + x2 + x3, x1 - x2 - x3]


def test_algorithm_methods_ga():
    """
    Not all elements are directly tested, but rather indirectly by running the
    algorithm with extreme values.
    """
    options = {
        "n_pop": 4,
        "n_bits": 64,
        "var_type_mixed": ["real", "int", "bool"],
        "n_iter": 1,
    }
    ga = Preferendus(
        objective=objective,
        constraints=(),
        bounds=((1, 10), (2, 20), (0, 1)),
        options=options,
    )
    init_pop, count = ga._initiate_population()
    assert count == 264
    assert len(init_pop[0][0]) == 64
    assert 2 <= init_pop[0][1] <= 20
    return


def test_algorithm_methods_preferendus():
    """
    Not all elements are directly tested, but rather indirectly by running the
    algorithm with extreme values.
    """
    options = {
        "n_pop": 4,
        "n_bits": 64,
        "var_type_mixed": ["real", "int", "bool"],
        "n_iter": 1,
        "aggregation": "IMAP",
    }
    ga = Preferendus(
        objective=objective,
        constraints=(),
        bounds=((1, 10), (2, 20), (0, 1)),
        options=options,
        start_points_population=[[4, 4, 1]],
    )
    init_pop, count = ga._initiate_population()
    assert count == 264
    assert len(init_pop[0][0]) == 64
    assert 2 <= init_pop[0][1] <= 20
    assert init_pop[0][2] in [0, 1]

    ip = [[0, 0, 1, 1], [1, 1, 0, 0]]
    set_pop = ga._starting_points_population(ip)
    assert set_pop[0] == set_pop[2]
    assert set_pop[1] == set_pop[3]
    assert set_pop[0] == ip[0]
    assert set_pop[1] == ip[1]

    res = ga.run()
    assert res[0] == -50.0
    assert res[1] == [4, 4, 1]
    return


if __name__ == "__main__":
    test_algorithm_methods_ga()
    test_algorithm_methods_preferendus()
