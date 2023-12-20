"""
Test function to check if the selection function is functioning correctly
"""
from preferendus._nextgen import _selection


def test_selection():
    """
    The selection is a stochastic process, making testing non-deterministic.
    Instead, 100 runs are made and a threshold value is implemented.
    """
    pop = [1, 2, 3, 1, 2, 3]
    scores = [10, 20, 30, 10, 20, 30]

    counter = 0
    for _ in range(1000):
        ret = _selection(pop=pop, scores=scores)
        counter += 1 if ret != 1 else 0

    assert isinstance(ret, int)
    assert counter < 250
    return


if __name__ == "__main__":
    test_selection()
