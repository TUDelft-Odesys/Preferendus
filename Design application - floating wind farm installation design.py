"""
Python code for the floating wind farm installation design problem.

Copyright (c) 2022. Harold Van Heukelum
"""

import pathlib
from math import ceil
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import pi
from scipy.interpolate import pchip_interpolate
from scipy.optimize import fsolve

from preferendus import Preferendus

HERE = pathlib.Path(__file__).parent

MAX_T = 3800
N_ANCHORS = 108

CONSTANTS = {
    "NC": 9,
    "NQ": 1,
    "W_steel": 78.5,  # kN/m3
    "W_water": 10.25,  # kN/m3
    "W_concrete": 25,  # kN/m3
}

# https://www.oilandgasiq.com/drilling-and-development/articles/offshore-support-vessels-leading-emissions-reducti
SHIP_OPTIONS = {
    "OCV small": {
        "day_rate": 47000,
        "deck_space": 8,
        "max_available": 3,
        "CO2_emission": 30,  # tonnes per day
        "chance": 0.7,
    },
    "OCV big": {
        "day_rate": 55000,
        "deck_space": 12,
        "max_available": 2,
        "CO2_emission": 40,  # tonnes per day
        "chance": 0.8,
    },
    "Barge": {
        "day_rate": 35000,
        "deck_space": 16,
        "max_available": 2,
        "CO2_emission": 35,  # tonnes per day
        "chance": 0.5,
    },
}

SOIL_DATA = {
    "type": "clay",
    "su": 60,  # kPa
    "a_i": 0.64,
    "a_o": 0.64,
    "sat_weight": 9,  # kN/m3
}

MOORING_DATA = {
    "type": "catenary",
    "line type": "chain",
    "d": 0.24,  # m
    "mu": 0.25,  # -
    "AWB": 2.5,  # -
}

TIME_INSTALLATION = 1
TIME_BUNKERING = [1.5, 2, 2.5]

# The Preference scores (p_points) and corresponding Objective results (x_points)
X_POINTS_1, P_POINTS_1 = [[45, 80, 113], [100, 60, 0]]
X_POINTS_2, P_POINTS_2 = [[9_500_000, 11_000_000, 17_000_000], [100, 20, 0]]
X_POINTS_3, P_POINTS_3 = [[0, 0.6, 1], [100, 50, 0]]
X_POINTS_4, P_POINTS_4 = [[3_200, 5_000, 10_200], [100, 40, 0]]

# weights for each objective
W_1 = 0.30
W_2 = 0.35
W_3 = 0.15
W_4 = 0.20

# arrays for plotting continuous preference curves
c1 = np.linspace(X_POINTS_1[0], X_POINTS_1[-1])
c2 = np.linspace(X_POINTS_2[0], X_POINTS_2[-1])
c3 = np.linspace(X_POINTS_3[0], X_POINTS_3[-1])
c4 = np.linspace(X_POINTS_4[0], X_POINTS_4[-1])

# calculate the preference functions
p1 = pchip_interpolate(X_POINTS_1, P_POINTS_1, c1)
p2 = pchip_interpolate(X_POINTS_2, P_POINTS_2, c2)
p3 = pchip_interpolate(X_POINTS_3, P_POINTS_3, c3)
p4 = pchip_interpolate(X_POINTS_4, P_POINTS_4, c4)


def objective_time(
    ocv_s: np.ndarray[int], ocv_l: np.ndarray[int], barge: np.ndarray[int]
) -> tuple[list, list, list, list]:
    """Function to calculate the project duration"""
    t_array = list()
    t_ocv_s = list()
    t_ocv_l = list()
    t_barge = list()

    for ip in range(len(ocv_s)):
        inf_loop_prevent = 0
        time_ocv_s = 0
        time_ocv_l = 0
        time_barge = 0
        anchor_counter = 0

        ds_ocv_s = SHIP_OPTIONS["OCV small"]["deck_space"]
        ds_ocv_l = SHIP_OPTIONS["OCV big"]["deck_space"]
        ds_barge = SHIP_OPTIONS["Barge"]["deck_space"]

        while N_ANCHORS - anchor_counter > 0:
            if (
                N_ANCHORS - anchor_counter
                < ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[ip] * ds_barge
            ):
                n = ocv_s[ip] + ocv_l[ip] + barge[ip]
                anchors_left_per_vessel = ceil((N_ANCHORS - anchor_counter) / n)
                diff_1 = 0
                diff_2 = 0
                ds_ocv_s = anchors_left_per_vessel
                ds_ocv_l = anchors_left_per_vessel
                ds_barge = anchors_left_per_vessel

                if ds_ocv_s > SHIP_OPTIONS["OCV small"]["deck_space"]:
                    diff_1 = ocv_s[ip] * (
                        anchors_left_per_vessel
                        - SHIP_OPTIONS["OCV small"]["deck_space"]
                    )
                    ds_ocv_s = SHIP_OPTIONS["OCV small"]["deck_space"]

                    if ocv_l[ip] != 0:
                        if (
                            ds_ocv_l + diff_1 / ocv_l[ip]
                            > SHIP_OPTIONS["OCV big"]["deck_space"]
                        ):
                            diff_2 = ocv_l[ip] * (
                                anchors_left_per_vessel
                                + round(diff_1 / ocv_l[ip])
                                - SHIP_OPTIONS["OCV big"]["deck_space"]
                            )
                            ds_ocv_l = SHIP_OPTIONS["OCV big"]["deck_space"]
                            ds_barge += diff_2 / barge[ip]
                        else:
                            ds_ocv_l = anchors_left_per_vessel + ceil(
                                diff_1 / (ocv_l[ip] + barge[ip])
                            )
                            ds_barge = anchors_left_per_vessel + ceil(
                                diff_1 / (ocv_l[ip] + barge[ip])
                            )
                    else:
                        ds_barge = anchors_left_per_vessel + ceil(diff_1 / barge[ip])

                assert ocv_s[ip] * ds_ocv_s + ocv_l[ip] * ds_ocv_l + barge[
                    ip
                ] * ds_barge >= (N_ANCHORS - anchor_counter)

            time_ocv_s += ocv_s[ip] * ds_ocv_s * TIME_INSTALLATION
            time_ocv_l += ocv_l[ip] * ds_ocv_l * TIME_INSTALLATION
            time_barge += barge[ip] * ds_barge * TIME_INSTALLATION

            anchor_counter += (
                ocv_s[ip] * SHIP_OPTIONS["OCV small"]["deck_space"]
                + ocv_l[ip] * SHIP_OPTIONS["OCV big"]["deck_space"]
                + barge[ip] * SHIP_OPTIONS["Barge"]["deck_space"]
            )

            if (
                N_ANCHORS - anchor_counter <= 0
            ):  # check if it is still the case after installation of last anchors
                time_ocv_s += ocv_s[ip] * TIME_BUNKERING[0]
                time_ocv_l += ocv_l[ip] * TIME_BUNKERING[1]
                time_barge += barge[ip] * TIME_BUNKERING[2]
            inf_loop_prevent += 1
            if inf_loop_prevent > 20:
                time_ocv_s += 1e4
                time_ocv_l += 1e4
                time_barge += 1e4
                break

        t_ocv_s.append(time_ocv_s)
        t_ocv_l.append(time_ocv_l)
        t_barge.append(time_barge)
        t_array.append(max(time_ocv_s, time_ocv_l, time_barge))

    return t_array, t_ocv_s, t_ocv_l, t_barge


def objective_costs(
    diameter: np.ndarray[float],
    length: np.ndarray[float],
    t_ocv_s: list[float],
    t_ocv_l: list[float],
    t_barge: list[float],
) -> np.ndarray[float]:
    """Function to calculate the installation costs"""

    t = 0.02 * diameter
    mass_steel = (
        pi * length * diameter * t + pi / 4 * diameter**2 * t
    ) * 7.85  # tonn
    production_costs_anchor = (mass_steel * 815 + 40000) * N_ANCHORS

    costs_ocv_s = np.array(t_ocv_s) * SHIP_OPTIONS["OCV small"]["day_rate"]
    costs_ocv_l = np.array(t_ocv_l) * SHIP_OPTIONS["OCV big"]["day_rate"]
    costs_barge = np.array(t_barge) * SHIP_OPTIONS["Barge"]["day_rate"]
    return production_costs_anchor + costs_ocv_s + costs_ocv_l + costs_barge


def objective_fleet_utilization(
    ocv_s: np.ndarray[int], ocv_l: np.ndarray[int], barge: np.ndarray[int]
) -> np.ndarray[float]:
    """Function to calculate the fleet utilization"""
    chance_ocv_s = SHIP_OPTIONS["OCV small"]["chance"] ** ocv_s
    chance_ocv_l = SHIP_OPTIONS["OCV big"]["chance"] ** ocv_l
    chance_barge = SHIP_OPTIONS["Barge"]["chance"] ** barge
    return np.prod(
        [
            np.power(chance_ocv_s, ocv_s),
            np.power(chance_ocv_l, ocv_l),
            np.power(chance_barge, barge),
        ],
        axis=0,
    )


def objective_co2(
    ocv_s: np.ndarray[int],
    ocv_l: np.ndarray[int],
    barge: np.ndarray[int],
    t_ocv_s: list[float],
    t_ocv_l: list[float],
    t_barge: list[float],
) -> np.ndarray[float]:
    """Function to calculate the CO2 emissions"""
    co2_emission_ocv_s = (
        np.array(t_ocv_s) * SHIP_OPTIONS["OCV small"]["CO2_emission"] * ocv_s
    )
    co2_emission_ocv_l = (
        np.array(t_ocv_l) * SHIP_OPTIONS["OCV big"]["CO2_emission"] * ocv_l
    )
    co2_emission_barge = (
        np.array(t_barge) * SHIP_OPTIONS["Barge"]["CO2_emission"] * barge
    )
    return co2_emission_ocv_s + co2_emission_ocv_l + co2_emission_barge


def single_objective_time(variables: np.ndarray) -> list:
    """Function for single objective optimization of the project duration"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    t_array, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return t_array


def single_objective_costs(variables: np.ndarray) -> np.ndarray:
    """Function for single objective optimization of the installation costs"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    diameter = variables[:, 3]
    length = variables[:, 4]
    _, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return objective_costs(diameter, length, t_ocv_s, t_ocv_l, t_barge)


def single_objective_fleet(variables: np.ndarray) -> np.ndarray:
    """Function for single objective optimization of the fleet utilization"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    return objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)


def single_objective_co2(variables: np.ndarray) -> np.ndarray:
    """Function for single objective optimization of the CO2 emissions"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    _, t_ocv_s, t_ocv_l, t_barge = objective_time(n_ocv_s, n_ocv_l, n_barge)
    return objective_co2(n_ocv_s, n_ocv_l, n_barge, t_ocv_s, t_ocv_l, t_barge)


def check_p_score(p_array: np.ndarray):
    """Function to mak sure all preference scores are in [0,100]"""
    mask1 = p_array < 0
    mask2 = p_array > 100
    p_array[mask1] = 0
    p_array[mask2] = 100
    return p_array


def objective(variables: np.ndarray) -> tuple[list, list]:
    """
    Objective function for the GA. Calculates all sub-objectives and their corresponding
    preference scores. The aggregation is done in the GA.
    """
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]
    diameter = variables[:, 3]
    length = variables[:, 4]

    project_time, time_ocv_s, time_ocv_l, time_barge = objective_time(
        n_ocv_s, n_ocv_l, n_barge
    )
    costs = objective_costs(diameter, length, time_ocv_s, time_ocv_l, time_barge)
    fleet_util = objective_fleet_utilization(n_ocv_s, n_ocv_l, n_barge)
    co2_emission = objective_co2(
        n_ocv_s, n_ocv_l, n_barge, time_ocv_s, time_ocv_l, time_barge
    )

    p_1 = check_p_score(pchip_interpolate(X_POINTS_1, P_POINTS_1, project_time))
    p_2 = check_p_score(pchip_interpolate(X_POINTS_2, P_POINTS_2, costs))
    p_3 = check_p_score(pchip_interpolate(X_POINTS_3, P_POINTS_3, fleet_util))
    p_4 = check_p_score(pchip_interpolate(X_POINTS_4, P_POINTS_4, co2_emission))

    return [W_1, W_2, W_3, W_4], [p_1, p_2, p_3, p_4]


def constraint_1(variables):
    """Constraint that ensures there is at least one vessel on the project"""
    n_ocv_s = variables[:, 0]
    n_ocv_l = variables[:, 1]
    n_barge = variables[:, 2]

    return -1 * (n_ocv_s + n_ocv_l + n_barge) + 1  # < 0


def _solve_ta_ta(p, tension_mudline, theta_m, za, d, mu, su, nc=7.6):
    """
    Solve the force o the anchor and its angle, based on the tension and angle of the
    mooring line at the seabed
    """
    tension_a, theta = p
    za_q = MOORING_DATA["AWB"] * d * nc * su * za

    return (2 * za_q / tension_a) - (theta**2 - theta_m**2), np.exp(
        mu * (theta - theta_m)
    ) - tension_mudline / tension_a


def constraint_2(variables):
    """
    Constraint that checks if the pull force on the anchors is lower than the
    resistance of the anchors to this force.

    The calculations are based on:
        - Houlsby, G. T. and Byrne, B. W. (2005). “Design procedures for installation
        of suction caissons in clay and other materials.” Proceedings of the
        Institution of Civil Engineers-Geotechnical Engineering, 158(2), 75–82.
        - Randolph, M. and Gourvenec, S. (2017). Offshore geotechnical engineering. CRC
        press.
        - Arany, L. and Bhattacharya, S. (2018). “Simplified load estimation and sizing
        of suction anchors for spar buoy type floating offshore wind turbines.” Ocean
        Engineering, 159, 348–357.
    """
    diameter = variables[:, 3]
    length = variables[:, 4]

    t = 0.02 * diameter
    d_i = diameter - t
    d_e = diameter + t
    mean_diameter = diameter
    weight_anchor = (
        np.pi * length * mean_diameter * t + np.pi * mean_diameter**2 * t / 4
    ) * (CONSTANTS["W_steel"] - CONSTANTS["W_water"])

    weight_plug = np.pi / 4 * d_i**2 * length * SOIL_DATA["sat_weight"]

    external_shaft_fric = np.pi * d_e * length * SOIL_DATA["a_o"] * SOIL_DATA["su"]
    internal_shaft_fric = np.pi * d_i * length * SOIL_DATA["a_i"] * SOIL_DATA["su"]
    reverse_end_bearing = 6.7 * SOIL_DATA["su"] * d_e**2 * np.pi / 4

    v_mode_1 = weight_anchor + external_shaft_fric + reverse_end_bearing
    v_mode_2 = weight_anchor + external_shaft_fric + internal_shaft_fric
    v_mode_3 = weight_anchor + external_shaft_fric + weight_plug

    v_max = np.amin([v_mode_1, v_mode_2, v_mode_3], axis=0)
    h_max = length * d_e * 10 * SOIL_DATA["su"]

    rel_pos_pad_eye = 0.5

    tension_pad_eye = np.zeros(len(length))
    angle_pad_eye = np.zeros(len(length))
    for lng in np.unique(length):
        x = fsolve(
            _solve_ta_ta,
            np.array([10000, 1]),
            (
                MAX_T,
                0,
                rel_pos_pad_eye * lng,
                MOORING_DATA["d"],
                MOORING_DATA["mu"],
                SOIL_DATA["su"],
                12,
            ),
        )
        mask = length == lng
        tension_pad_eye[mask] = x[0]
        angle_pad_eye[mask] = x[1]

    h = np.cos(angle_pad_eye) * tension_pad_eye
    v = np.sin(angle_pad_eye) * tension_pad_eye

    a = length / mean_diameter + 0.5
    b = length / (3 * mean_diameter) + 4.5

    hor_util = h / h_max
    ver_util = v / v_max

    return (hor_util**a + ver_util**b) - 1


def print_results(res: Union[list, np.ndarray]):
    """Function that prints the results of the optimizations"""
    print(f"Optimal result for:\n")
    print(f"\t {res[0]} small Offshore Construction Vessels\n")
    print(f"\t {res[1]} large Offshore Construction Vessels\n")
    print(f"\t {res[2]} Barges\n")
    print(f"\tAn anchor diameter of {round(res[3], 2)}m\n")
    print(f"\tAn anchor length of {round(res[4], 2)}m\n")


def make_figures(
    objective_results: list, preference_results: list, method: list
) -> None:
    """Function to plot all the figures of this problem"""
    # create figure that plots all preference curves and the preference scores of the
    # returned results of the GA
    markers = ["o", "s", "+"]
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 5))

    ax1.plot(c1, p1, label="Preference Function")
    for i in range(len(objective_results[0])):
        ax1.scatter(
            objective_results[0][i],
            preference_results[0][i],
            label=method[i],
            marker=markers[i],
        )
    ax1.set_ylim((0, 100))
    ax1.set_title("Project Duration")
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Preference function outcome")
    ax1.grid()
    fig2.legend()

    ax2.plot(c2, p2)
    for i in range(len(objective_results[1])):
        ax2.scatter(
            objective_results[1][i], preference_results[1][i], marker=markers[i]
        )
    ax2.set_ylim((0, 100))
    ax2.set_title("Installation Costs")
    ax2.set_xlabel("Costs [€]")
    ax2.set_ylabel("Preference function outcome")
    ax2.grid()

    ax3.plot(c3, p3)
    for i in range(len(objective_results[2])):
        ax3.scatter(
            objective_results[2][i], preference_results[2][i], marker=markers[i]
        )
    ax3.set_ylim((0, 100))
    ax3.set_title("Fleet Utilization")
    ax3.set_xlabel("Number of vessels [-]")
    ax3.set_ylabel("Preference function outcome")
    ax3.grid()

    ax4.plot(c4 * 1e-3, p4)
    for i in range(len(objective_results[3])):
        ax4.scatter(
            objective_results[3][i] * 1e-3, preference_results[3][i], marker=markers[i]
        )
    ax4.set_ylim((0, 100))
    ax4.set_title(r"$CO_2$ emissions")
    ax4.set_xlabel(r"$CO_2$ emission [$10^3$ tonnes]")
    ax4.set_ylabel("Preference function outcome")
    ax4.grid()


if __name__ == "__main__":
    # define bounds and set constraints list
    bounds = (
        (0, SHIP_OPTIONS["OCV small"]["max_available"]),
        (0, SHIP_OPTIONS["OCV big"]["max_available"]),
        (0, SHIP_OPTIONS["Barge"]["max_available"]),
        (1.5, 4),
        (2, 8),
    )
    cons = (("ineq", constraint_1), ("ineq", constraint_2))

    ####################################################################################
    # run single objectives and save to save_array
    save_array = list()
    methods = list()

    # make dictionary with parameter settings for the GA
    options = {
        "n_bits": 16,
        "n_iter": 400,
        "n_pop": 500,
        "r_cross": 0.8,
        "max_stall": 20,
        "var_type_mixed": ["int", "int", "int", "real", "real"],
        "aggregation": None,
    }

    # time
    ga = Preferendus(
        objective=single_objective_time,
        constraints=cons,
        cons_handler="CND",
        bounds=bounds,
        options=options,
    )
    res_time, design_variables_SO_time, _ = ga.run()
    print_results(design_variables_SO_time)
    print(f"SODO project duration: {round(res_time, 2)} days")

    # fleet utilization
    ga = Preferendus(
        objective=single_objective_fleet,
        constraints=cons,
        cons_handler="CND",
        bounds=bounds,
        options=options,
    )
    res_fleet, design_variables_SO_fleet, _ = ga.run()
    print_results(design_variables_SO_fleet)
    print(f"SODO fleet utilization: {round(res_fleet, 2)}")

    # CO2
    ga = Preferendus(
        objective=single_objective_co2,
        constraints=cons,
        cons_handler="CND",
        bounds=bounds,
        options=options,
    )
    res_co2, design_variables_SO_co2, _ = ga.run()
    print_results(design_variables_SO_co2)
    print(f"SODO CO2 emissions: {round(res_co2, 2)} tonnes")

    # costs
    options["n_bits"] = 20
    options["n_pop"] = 1500
    options["r_cross"] = 0.85
    options["mutation_rate_order"] = 4
    options["elitism percentage"] = 10

    ga = Preferendus(
        objective=single_objective_costs,
        constraints=cons,
        cons_handler="CND",
        bounds=bounds,
        options=options,
    )
    res_costs, design_variables_SO_costs, _ = ga.run()
    print_results(design_variables_SO_costs)
    print(f"SODO installation costs: €{round(res_costs, 2)}")
    save_array.append(design_variables_SO_costs)
    methods.append("SODO Costs")

    ####################################################################################
    # run multi-objective with minmax solver

    # change some entries in the options dictionary
    options["n_bits"] = 24
    options["r_cross"] = 0.8
    options["aggregation"] = "minmax"

    ga = Preferendus(
        objective=objective,
        constraints=cons,
        cons_handler="CND",
        bounds=bounds,
        options=options,
    )
    _, design_variables_minmax, best_mm = ga.run()
    print_results(design_variables_minmax)
    save_array.append(design_variables_minmax)
    methods.append("Min-max")

    ####################################################################################
    # run multi-objective with IMAP solver

    # change some entries in the options dictionary
    options["n_bits"] = 20
    options["n_pop"] = 500
    options["r_cross"] = 0.9
    options["aggregation"] = "IMAP"
    options["mutation_rate_order"] = 2
    options["max_stall"] = 10

    ga = Preferendus(
        objective=objective,
        constraints=cons,
        bounds=bounds,
        options=options,
        start_points_population=[design_variables_minmax],
    )
    _, design_variables_IMAP, best_t = ga.run()
    print_results(design_variables_IMAP)
    save_array.append(design_variables_IMAP)
    methods.append("IMAP")

    ###################################################################################
    # evaluate all runs

    variable = np.array(save_array)  # make ndarray

    w, p = objective(variable)  # evaluate objective
    r = ga.solver.request(w, p)  # get aggregated scores to rank them

    # create pandas DataFrame and print it to console
    d = {
        "Method": methods,
        "Results": np.round(r),
        "Variable 1": np.round(variable[:, 0]),
        "Variable 2": np.round(variable[:, 1]),
        "Variable 3": np.round(variable[:, 2]),
        "Variable 4": np.round(variable[:, 3], 2),
        "Variable 5": np.round(variable[:, 4], 2),
    }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    c1_res, t_res_1, t_res_2, t_res_3 = objective_time(
        variable[:, 0], variable[:, 1], variable[:, 2]
    )
    c2_res = objective_costs(variable[:, 3], variable[:, 4], t_res_1, t_res_2, t_res_3)
    c3_res = objective_fleet_utilization(variable[:, 0], variable[:, 1], variable[:, 2])
    c4_res = objective_co2(
        variable[:, 0], variable[:, 1], variable[:, 2], t_res_1, t_res_2, t_res_3
    )

    p1_res = pchip_interpolate(X_POINTS_1, P_POINTS_1, c1_res)
    p2_res = pchip_interpolate(X_POINTS_2, P_POINTS_2, c2_res)
    p3_res = pchip_interpolate(X_POINTS_3, P_POINTS_3, c3_res)
    p4_res = pchip_interpolate(X_POINTS_4, P_POINTS_4, c4_res)

    d = {
        "Method": methods,
        "Project duration": np.round(c1_res, 2),
        "Costs [1e6]": np.round(c2_res * 1e-6, 2),
        "Fleet util": np.round(c3_res, 2),
        "Emissions": np.round(c4_res),
    }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    d = {
        "Method": methods,
        "Project duration": np.round(p1_res),
        "Costs": np.round(p2_res),
        "Fleet util": np.round(p3_res),
        "Emissions": np.round(p4_res),
    }
    print()
    print(pd.DataFrame(data=d).to_string())
    print()

    make_figures(
        objective_results=[c1_res, c2_res, c3_res, c4_res],
        preference_results=[p1_res, p2_res, p3_res, p4_res],
        method=methods,
    )
    plt.show()
