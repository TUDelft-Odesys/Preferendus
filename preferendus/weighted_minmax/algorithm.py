"""
Copyright (c) 2022. Harold Van Heukelum
"""
import numpy as np


def aggregate_max(w, p, goal):
    """
    Function to find the element with the highest distance towards the utopian point
    (goal). Function to be used for the min-max aggregation method. Minimizing the
    result of this function will give you the "compromise" solution.

    :param w: weights of the different objectives
    :param p: 2d-array with the scores of the objectives. n-by-m, where n is the number
        of objectives and m the population size
    :param goal: utopian point (single value of separated per objective)
    :return: List with (weighted) maximum distances
    """
    assert len(w) == len(p), (
        f"The number of weights ({len(w)}) is not equal to the number of "
        f"objectives ({len(p)})."
    )
    assert (
        round(sum(w), 4) == 1
    ), f"The sum of the weights ({round(sum(w), 4)}) is not equal to 1."

    p = np.array(p)

    if isinstance(goal, int) or isinstance(goal, str):
        goal_array = [int(goal)] * len(w)
    elif isinstance(goal, list) or isinstance(goal, np.ndarray):
        assert len(goal) == len(w), (
            f"List with goal values should be of equal size as the weights "
            f"({len(goal)}, {len(w)})"
        )
        goal_array = goal
    else:
        raise TypeError(
            "Goal value(s) should either be an integer, string, list or ndarray"
        )

    distance_array = list()
    for i in range(len(w)):
        weight = w[i]
        p_i = p[i, :]
        goal_i = goal_array[i]
        distance_array.append(weight * (goal_i - p_i))

    return np.amax(distance_array, axis=0)
