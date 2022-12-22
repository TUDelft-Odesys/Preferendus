"""
functions for multiprocessing
"""
import numpy as np


def h_ult_clay_piled_anchors(x, var):
    """
    Function to calculate the ultimate horizontal capacity of anchor piles in clay

    References:
        -Randolph, M., & Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
    """
    l, p_max = x

    angle = var[1]
    length = var[2]
    diameter = var[3]
    n_p = var[4]
    su = var[5]
    za = var[6]

    f1 = n_p * su * diameter * l
    f2 = n_p * su * diameter * (length - l)

    b1 = l / 2
    b2 = l + (length - l) / 2

    return p_max * np.cos(angle) + f2 - f1, b1 * f1 - f2 * b2 - p_max * za * np.cos(
        angle) - p_max * diameter / 2 * np.sin(angle)


def h_ult_sand_piled_anchors(x, var):
    """
    Function to calculate the ultimate horizontal capacity of anchor piles in sand

    References:
        -Randolph, M., & Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
    """
    l, p_max = x

    angle = var[1]
    length = var[2]
    diameter = var[3]
    k_p = var[4]
    sat_weight = var[5]
    za = var[6]

    f1 = 0.5 * diameter * k_p ** 2 * sat_weight * l ** 2
    f2 = diameter * k_p ** 2 * sat_weight * l * (length - l)
    f3 = 0.5 * diameter * k_p ** 2 * sat_weight * (length - l) ** 2

    b1 = 2 / 3 * l
    b2 = l + 0.5 * (length - l)
    b3 = l + 2 / 3 * (length - l)

    return p_max * np.cos(angle) + f2 + f3 - f1, b1 * f1 - f2 * b2 - f3 * b3 - p_max * za * np.cos(
        angle) - p_max * diameter / 2 * np.sin(angle)
