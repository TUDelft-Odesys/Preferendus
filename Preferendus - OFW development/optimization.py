"""Copyright (c) 2022. Harold Van Heukelum"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import pchip_interpolate

from _get_data_from_excel import GetData
from _scheduler import CreateSchedule
from engineering.anchor_modelling import EngineeringCalculations
from genetic_algorithm_pfm import GeneticAlgorithm

HERE = pathlib.Path(__file__).parent

scheduler = CreateSchedule()
get_data = GetData()
anchor_calc = EngineeringCalculations(get_data.soil_data, get_data.mooring_data)

bounds_vessels, var_type_vessels, masks, vessel_names = get_data.build_bounds_vessels()
force_data = get_data.get_governing_forces(get_data.project_data['water depth'])

s_data_basis = {
    'idle': 0,
    'port': 0,
    'transit': 0,
    'DP': 0,
    'AH': 0,
    'towing': 0
}

s_data = dict()
for keys_basis in s_data_basis.keys():
    temp_array = list()
    for name in vessel_names:
        temp_array.append(get_data.vessel_fuel[name][keys_basis])
    s_data[keys_basis] = temp_array

DENSITY_MGO = 890 * 1e-3  # mT/m3, https://maritimepage.com/what-are-mgo-and-mdo-fuels-marine-fuels-explained/

# arrays for plotting continuous preference curves
x_points_1, p_points_1 = [[600, 1000, 1600], [100, 40, 0]]
x_points_2, p_points_2 = [[50E6, 65e6, 120e6, 600e6], [100, 95, 10, 0]]
x_points_3, p_points_3 = [[0.5, 0.8, 1], [0, 70, 100]]
x_points_4, p_points_4 = [[14_000, 75_000, 180_000], [100, 60, 0]]

weights = [0.25, 0.5, 0.20, 0.05]

c1 = np.linspace(x_points_1[0], x_points_1[-1])
c2 = np.linspace(x_points_2[0], x_points_2[-1])
c3 = np.linspace(x_points_3[0], x_points_3[-1])
c4 = np.linspace(x_points_4[0], x_points_4[-1])

p1 = pchip_interpolate(x_points_1, p_points_1, c1)
p2 = pchip_interpolate(x_points_2, p_points_2, c2)
p3 = pchip_interpolate(x_points_3, p_points_3, c3)
p4 = pchip_interpolate(x_points_4, p_points_4, c4)

s_sp_vessels = 0
e_sp_vessels = s_sp_vessels + np.count_nonzero(masks['sp'])
s_pile_vessels = e_sp_vessels
e_pile_vessels = s_pile_vessels + np.count_nonzero(masks['pile'])
s_dea_vessels = e_pile_vessels
e_dea_vessels = s_dea_vessels + np.count_nonzero(masks['dea'])
s_ml_taut_vessels = e_dea_vessels
e_ml_taut_vessels = s_ml_taut_vessels + np.count_nonzero(masks['taut'])
s_ml_catenary_vessels = e_ml_taut_vessels
e_ml_catenary_vessels = s_ml_catenary_vessels + np.count_nonzero(masks['chain'])
s_ten_vessels = e_ml_catenary_vessels
e_ten_vessels = s_ten_vessels + np.count_nonzero(masks['ten'])
s_hu_vessels = e_ten_vessels
e_hu_vessels = s_hu_vessels + 1


def rounder(values):
    """
    Set value to the closest one in a list
    """

    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]

    return np.frompyfunc(f, 1, 1)


def set_length_diameter_dea(lengths, diameter, anchor_id):
    """
    Set length and width of DEA anchors to the values that are available
    """
    lengths_list = [2.429, 2.774, 3.493, 4.120, 4.602, 5.012, 5.516, 5.942, 6.372, 7.289]  # MK3
    lengths_list += [2.954, 3.721, 4.412, 5.161, 5.559, 5.908, 6.364, 6.763, 7.004, 7.230, 7.545, 8.018]  # MK5
    lengths_list += [2.797, 3.523, 4.178, 4.886, 5.263, 5.593, 6.025, 6.402, 6.631, 6.845, 7.148, 7.591]  # MK6

    widths_list = [2.654, 3.038, 3.828, 4.538, 5.077, 5.521, 6.076, 6.545, 6.986, 7.997]  # MK3
    widths_list += [3.184, 4.011, 4756, 5.563, 5.992, 6.368, 6.860, 7.290, 7.550, 7.794, 8.133, 8.643]  # MK5
    widths_list += [3.059, 3.870, 4602, 5.390, 5.807, 6.171, 6.679, 7.101, 7.368, 7.625, 7.962, 8.451]  # MK6

    ln = list()
    dm = list()
    loc_id = list()
    for i in range(len(anchor_id)):
        if anchor_id[i] == 0:
            dm.append(rounder(np.array(widths_list))(diameter[i]))
            loc = widths_list.index(dm[i])
            ln.append(lengths_list[loc])
            loc_id.append(loc)
        else:
            dm.append(diameter[i])
            ln.append(lengths[i])
            loc_id.append(-2)

    return np.array(ln), np.array(dm), np.array(loc_id)


def objective_time(variables, loads, ml_type_str, export=False, method_name=None):
    """Function to calculate the installation time"""
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = ml_type_str
    shared = variables[:, -3]
    lengths = variables[:, -2]
    diameters = variables[:, -1]

    lengths, diameters, _ = set_length_diameter_dea(lengths, diameters, anchor_ids)

    ret = scheduler.construct_schedule_optimization(anchor_id=anchor_ids, ml_type=ml_type, vessels=vessels,
                                                    lengths=lengths, diameters=diameters,
                                                    work_windows=get_data.work_windows,
                                                    workability_limits=get_data.workability_limits, proof_load=loads,
                                                    shared=shared, export=export, name=method_name)

    return ret


def objective_costs(ship_times, anchor_ids, ml_type, extra_ships_times, hu_vessels):
    """Function to calculate the installation costs"""

    costs_vessels = np.zeros_like(ship_times)

    mask_sp = np.logical_or(anchor_ids == 1, anchor_ids == 2)
    costs_vessels[mask_sp, s_sp_vessels:e_sp_vessels] += ship_times[mask_sp, s_sp_vessels:e_sp_vessels] * \
                                                         (get_data.vessel_costs_list[masks['sp']] +
                                                          get_data.additional_costs['Suction anchor'])

    mask_pile = anchor_ids == 3
    costs_vessels[mask_pile, s_pile_vessels:e_pile_vessels] += ship_times[mask_pile, s_pile_vessels:e_pile_vessels] * \
                                                               (get_data.vessel_costs_list[masks['pile']] +
                                                                get_data.additional_costs['Anchor pile'])

    mask_dea = anchor_ids == 0
    costs_vessels[mask_dea, s_dea_vessels:e_dea_vessels] += ship_times[mask_dea, s_dea_vessels:e_dea_vessels] * \
                                                            (get_data.vessel_costs_list[masks['dea']] +
                                                             get_data.additional_costs['DEA'])

    mask_taut = np.logical_and(ml_type == 1, anchor_ids != 0)
    costs_vessels[mask_taut, s_ml_taut_vessels:e_ml_taut_vessels] += ship_times[mask_taut,
                                                                     s_ml_taut_vessels:e_ml_taut_vessels] * \
                                                                     get_data.vessel_costs_list[masks['taut']]

    mask_catenary = np.logical_and(ml_type == 0, anchor_ids != 0)
    costs_vessels[mask_catenary, s_ml_catenary_vessels:e_ml_catenary_vessels] += ship_times[mask_catenary,
                                                                                 s_ml_catenary_vessels:e_ml_catenary_vessels] * \
                                                                                 get_data.vessel_costs_list[
                                                                                     masks['chain']]

    costs_vessels[mask_dea, s_ten_vessels:e_ten_vessels] += ship_times[mask_dea, s_ten_vessels:e_ten_vessels] * \
                                                            get_data.vessel_costs_list[masks['ten']]

    cost_hu = np.zeros_like(hu_vessels)
    for i in range(len(hu_vessels)):
        cost_hu[i] += get_data.vessel_costs_list[masks['hu']][int(hu_vessels[i])]
    costs_vessels[:, s_hu_vessels:e_hu_vessels] += ship_times[:, s_hu_vessels:e_hu_vessels] * cost_hu

    cost_vessels_total = np.sum(costs_vessels, axis=1)

    for i in range(len(cost_vessels_total)):
        try:
            additional_costs_ten = sum(extra_ships_times[i]['HLV'][0]) * 150000
        except KeyError:
            additional_costs_ten = sum(extra_ships_times[i]['AHT'][0]) * 50000

        additional_costs_tugs = 0.5 * sum(extra_ships_times[i]['tugs'][0]) * 24000 + \
                                0.5 * sum(extra_ships_times[i]['tugs'][0]) * 54000
        cost_vessels_total[i] += additional_costs_ten + additional_costs_tugs

    return cost_vessels_total


def objective_fleet_utilization(vessels, anchor_ids, ml_type, hu_vessels):
    """Function to calculate the expected fleet utilization"""
    util_vessels = np.ones_like(vessels)
    util_vessels = np.full(np.shape(util_vessels), np.nan)

    mask_sp = np.logical_or(anchor_ids == 1, anchor_ids == 2)
    util_vessels[mask_sp, s_sp_vessels:e_sp_vessels] = vessels[mask_sp, s_sp_vessels:e_sp_vessels] * \
                                                       get_data.vessel_util[:, 0][masks['sp']]

    mask_pile = anchor_ids == 3
    util_vessels[mask_pile, s_pile_vessels:e_pile_vessels] = vessels[mask_pile, s_pile_vessels:e_pile_vessels] * \
                                                             get_data.vessel_util[:, 1][masks['pile']]

    mask_dea = anchor_ids == 0
    util_vessels[mask_dea, s_dea_vessels:e_dea_vessels] = vessels[mask_dea, s_dea_vessels:e_dea_vessels] * \
                                                          get_data.vessel_util[:, 2][masks['dea']]

    mask_taut = np.logical_and(ml_type == 1, anchor_ids != 0)
    util_vessels[mask_taut, s_ml_taut_vessels:e_ml_taut_vessels] = vessels[mask_taut,
                                                                   s_ml_taut_vessels:e_ml_taut_vessels] * \
                                                                   get_data.vessel_util[:, 3][masks['taut']]

    mask_catenary = np.logical_and(ml_type == 0, anchor_ids != 0)
    util_vessels[mask_catenary, s_ml_catenary_vessels:e_ml_catenary_vessels] = vessels[mask_catenary,
                                                                               s_ml_catenary_vessels:e_ml_catenary_vessels] * \
                                                                               get_data.vessel_util[:, 4][
                                                                                   masks['chain']]

    util_vessels[mask_dea, s_ten_vessels:e_ten_vessels] = vessels[mask_dea, s_ten_vessels:e_ten_vessels] * \
                                                          get_data.vessel_util[:, 5][masks['ten']]

    util_hu = np.zeros_like(hu_vessels)
    for i in range(len(hu_vessels)):
        util_hu[i] += get_data.vessel_util[:, 6][masks['hu']][int(hu_vessels[i])]
    util_vessels[:, s_hu_vessels:e_hu_vessels] = util_hu

    return np.nanmean(1 - util_vessels, axis=1)


def objective_sustainability(sustainability_data, hu_vessels):
    """Function to calculate the expected CO2 emissions of the project"""
    emissions = np.zeros_like(sustainability_data, dtype=float)

    for i, data in enumerate(sustainability_data):
        emissions_temp = 0
        for key in data.keys():
            data_task = data[key]

            data_hu = data_task[-1]
            hu_time = np.zeros(len(s_data[key][s_hu_vessels:]))
            id_hu_vessel = int(hu_vessels[i])
            hu_time[id_hu_vessel] = data_hu
            data_task_extended = np.append(data_task[:-1], hu_time)

            fuel_usage_task = data_task_extended * s_data[key]  # m3
            fuel_usage_sum = np.multiply(sum(fuel_usage_task), DENSITY_MGO)  # mT
            emissions_temp += np.multiply(fuel_usage_sum, 3.206)  # mT CO2
        emissions[i] = emissions_temp

    return emissions


def mask_p(p):
    """
    Make sure no preference score is not in [0, 100]
    """
    mask1 = p > 100
    mask2 = p < 0
    p[mask1] = 100
    p[mask2] = 0
    return p


def objective(variables):
    """
    Objective function for the GA. Calculates all sub-objectives and transforms it from their scale to the preference
    scale.
    """
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    hu_vessels = variables[:, s_hu_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]
    shared = variables[:, -3]

    loads = list()
    ml_type_str = list()
    for it, item in enumerate(shared):
        if item:  # shared
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, shared, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, shared, governing'])
        else:
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, single, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, single, governing'])

    project_duration, vessel_times, extra_ships_times, sustainability_data = objective_time(variables,
                                                                                            loads,
                                                                                            ml_type_str)
    costs = objective_costs(vessel_times, anchor_ids, ml_type, extra_ships_times, hu_vessels)
    fleet_utilization = objective_fleet_utilization(vessels, anchor_ids, ml_type, hu_vessels)
    sustainability = objective_sustainability(sustainability_data, hu_vessels)

    p_1 = mask_p(pchip_interpolate(x_points_1, p_points_1, project_duration))
    p_2 = mask_p(pchip_interpolate(x_points_2, p_points_2, costs))
    p_3 = mask_p(pchip_interpolate(x_points_3, p_points_3, fleet_utilization))
    p_4 = mask_p(pchip_interpolate(x_points_4, p_points_4, sustainability))

    return weights, [p_1, p_2, p_3, p_4]


def single_objective_costs(variables):
    """Function for single objective optimization of the costs objective"""
    hu_vessels = variables[:, s_hu_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]
    shared = variables[:, -3]

    loads = list()
    ml_type_str = list()
    for it, item in enumerate(shared):
        if item:  # shared
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, shared, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, shared, governing'])
        else:
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, single, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, single, governing'])

    project_duration, vessel_times, extra_ships_times, sustainability_data = objective_time(variables,
                                                                                            loads,
                                                                                            ml_type_str)
    return objective_costs(vessel_times, anchor_ids, ml_type, extra_ships_times, hu_vessels)


def constraint_1(variables):
    """Constraint that checks if the number of anchor installation vessels and mooring line installation vessels > 1"""
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]

    counter = np.zeros(len(variables))
    for i in range(len(variables)):
        if anchor_ids[i] == 0:
            a_vessel = sum(vessels[i, s_dea_vessels:e_dea_vessels])
        elif anchor_ids[i] == 3:
            a_vessel = sum(vessels[i, s_pile_vessels:e_pile_vessels])
        else:
            a_vessel = sum(vessels[i, s_sp_vessels:e_sp_vessels])

        if anchor_ids[i] == 0:
            ml_vessel = sum(vessels[i, s_ten_vessels:e_ten_vessels])
        elif ml_type[i]:
            ml_vessel = sum(vessels[i, s_ml_taut_vessels:e_ml_taut_vessels])
        else:
            ml_vessel = sum(vessels[i, s_ml_catenary_vessels:e_ml_catenary_vessels])

        counter[i] += max(1 - a_vessel, 1 - ml_vessel)

    return counter  # should be ≤ 0


def constraint_2(variables):
    """
    Constraint that checks if the number of vessels for anchor installation is equal to the number of mooring line
    installation vessels in the case of simultaneous installation
    """
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]

    # simultaneous task should have same number of vessels
    counter = np.zeros(len(variables))
    for i in range(len(variables)):
        if anchor_ids[i] == 0:
            a_vessel = sum(vessels[i, s_dea_vessels:e_dea_vessels])
        elif anchor_ids[i] == 3:
            a_vessel = sum(vessels[i, s_pile_vessels:e_pile_vessels])
        else:
            a_vessel = sum(vessels[i, s_sp_vessels:e_sp_vessels])

        if anchor_ids[i] == 0:
            ml_vessel = sum(vessels[i, s_ten_vessels:e_ten_vessels])
        elif ml_type[i]:
            ml_vessel = sum(vessels[i, s_ml_taut_vessels:e_ml_taut_vessels])
        else:
            ml_vessel = sum(vessels[i, s_ml_catenary_vessels:e_ml_catenary_vessels])

        if anchor_ids[i] in [2., 3.]:
            counter[i] += abs(a_vessel - ml_vessel)

    return counter  # should be ≤ 0


def constraint_3(variables):  # V_ult * n_anchor > F_pull
    """Constraint that checks if a vessel is not performing two tasks simultaneously"""
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]

    counter = np.zeros(len(variables))
    for i in range(len(variables)):
        if anchor_ids[i] == 0:
            a_vessel = vessels[i, s_dea_vessels:e_dea_vessels]
            names_a = np.array(vessel_names[s_dea_vessels:e_dea_vessels])
        elif anchor_ids[i] == 3:
            a_vessel = vessels[i, s_pile_vessels:e_pile_vessels]
            names_a = np.array(vessel_names[s_pile_vessels:e_pile_vessels])
        else:
            a_vessel = vessels[i, s_sp_vessels:e_sp_vessels]
            names_a = np.array(vessel_names[s_sp_vessels:e_sp_vessels])

        if anchor_ids[i] == 0:
            ml_vessel = vessels[i, s_ten_vessels:e_ten_vessels]
            names_ml = np.array(vessel_names[s_ten_vessels:e_ten_vessels])
        elif ml_type[i]:
            ml_vessel = vessels[i, s_ml_taut_vessels:e_ml_taut_vessels]
            names_ml = np.array(vessel_names[s_ml_taut_vessels:e_ml_taut_vessels])
        else:
            ml_vessel = vessels[i, s_ml_catenary_vessels:e_ml_catenary_vessels]
            names_ml = np.array(vessel_names[s_ml_catenary_vessels:e_ml_catenary_vessels])

        hu_vessel = vessels[i, s_hu_vessels:e_hu_vessels]
        name_hu_vessel = vessel_names[s_hu_vessels:][int(hu_vessel)]

        mask_ml = ml_vessel == 1
        mask_a = a_vessel == 1
        if anchor_ids[i] in [2., 3.]:  # check if vessel is not install anchors and ML at the same time
            for nm in names_a[mask_a]:
                if nm in names_ml[mask_ml]:
                    counter[i] += 1

        if name_hu_vessel in names_ml[mask_ml]:  # check if vessel is not installing MLs and FWT at the same time
            counter[i] += 1

        if name_hu_vessel in names_a[mask_a]:  # check if vessel is not installing anchors and FWT at the same time
            counter[i] += 1

    return counter  # should be ≤ 0


def constraint_4(variables):
    """Constraint that checks that DEa is not used for taut or shared moorings"""
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]
    shared = variables[:, -3]

    counter = np.zeros(len(variables))
    for i in range(len(variables)):
        if anchor_ids[i] == 0 and ml_type[i]:
            counter[i] += 1
        elif shared[i] and anchor_ids[i] == 0:
            counter[i] += 1

    return counter  # should be ≤ 0


def constraint_5(variables):
    """Constraint that checks if the resistance of the anchors is sufficient"""
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]
    shared = variables[:, -3]
    lengths = variables[:, -2]
    diameters = variables[:, -1]

    lengths, diameters, loc = set_length_diameter_dea(lengths, diameters, anchor_ids)

    forces = np.zeros_like(shared)
    mask_shared = shared == 1
    mask_taut = ml_type == 1
    forces[np.logical_and(mask_shared, mask_taut)] = force_data['taut, shared, governing']
    forces[np.logical_and(np.invert(mask_shared), mask_taut)] = force_data['taut, single, governing']
    forces[np.logical_and(mask_shared, np.invert(mask_taut))] = force_data['catenary, shared, governing']
    forces[np.logical_and(np.invert(mask_shared), np.invert(mask_taut))] = force_data['catenary, single, governing']
    forces *= 1e-3  # N to kN

    angles_mudline = np.zeros_like(shared)
    mask_taut = ml_type == 1
    angles_mudline[mask_taut] = force_data['taut, angle']

    utility = np.zeros_like(shared)

    mask_sp = np.logical_or(anchor_ids == 1, anchor_ids == 2)
    utility[mask_sp] = anchor_calc.suction_anchor(forces[mask_sp], angles_mudline[mask_sp], diameters[mask_sp],
                                                  lengths[mask_sp])[0]

    mask_pile = anchor_ids == 3
    utility[mask_pile] = anchor_calc.piled_anchors(forces[mask_pile], angles_mudline[mask_pile], diameters[mask_pile],
                                                   lengths[mask_pile])

    mask_dea = anchor_ids == 0
    utility[mask_dea] = anchor_calc.fluke_anchors(forces[mask_dea], loc[mask_dea])

    return -1 + utility  # should be ≤ 0


def constraint_6(variables):
    """Constraint that checks if the L/D ratios of the anchors are OK"""
    anchor_ids = variables[:, -5]
    lengths = variables[:, -2]
    diameters = variables[:, -1]

    violation = [0] * len(anchor_ids)
    for it, a_id in enumerate(anchor_ids):
        if a_id == 0:
            if lengths[it] > 8.1:
                violation[it] += lengths[it] - 8.1
            if diameters[it] > 8.7:
                violation[it] += diameters[it] - 8.4
        elif a_id == 3:
            l_d = lengths[it] / diameters[it]
            ratio = 30 if get_data.soil_data['type'] == 'clay' else 40
            if l_d < 0.5 * ratio:
                violation[it] = ratio - l_d
            elif l_d > ratio:
                violation[it] = l_d - ratio
        else:
            l_d = lengths[it] / diameters[it]
            ratio = 6
            if l_d < 3:
                violation[it] = ratio - l_d
            elif l_d > ratio:
                violation[it] = l_d - ratio

    return np.array(violation)


def get_results(variables, methods, scenario):
    """

    :param scenario:
    :param methods:
    :param variables:
    :return:
    """
    vessels = variables[:, s_sp_vessels:e_hu_vessels]
    hu_vessels = variables[:, s_hu_vessels:e_hu_vessels]
    anchor_ids = variables[:, -5]
    ml_type = variables[:, -4]
    shared = variables[:, -3]

    loads = list()
    ml_type_str = list()
    for it, item in enumerate(shared):
        if item:  # shared
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, shared, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, shared, governing'])
        else:
            if ml_type[it]:  # taut
                ml_type_str.append('taut')
                loads.append(force_data['taut, single, governing'])
            else:
                ml_type_str.append('catenary')
                loads.append(force_data['catenary, single, governing'])

    names = list()
    for met in methods:
        names.append(f'{met}_{scenario}')

    project_duration, vessel_times, extra_ships_times, sustainability_data = objective_time(variables,
                                                                                            loads,
                                                                                            ml_type_str,
                                                                                            export=True,
                                                                                            method_name=names)

    costs = objective_costs(vessel_times, anchor_ids, ml_type, extra_ships_times, hu_vessels)
    fleet_utilization = objective_fleet_utilization(vessels, anchor_ids, ml_type, hu_vessels)
    sustainability = objective_sustainability(sustainability_data, hu_vessels)
    return project_duration, costs, fleet_utilization, sustainability


def main(scenario):
    """Function to run optimizations and save all figures etc."""

    # set bounds
    bounds = bounds_vessels
    var_type_mixed = var_type_vessels

    bounds += [0, 3],  # anchor id
    var_type_mixed += ['int']

    bounds += [0, 1],  # ML type
    var_type_mixed += ['bool']

    bounds += [0, 1],  # shared mooring
    var_type_mixed += ['bool']

    bounds += [
        [2, 64],  # length anchors
        [1, 6],  # diameter anchors
    ]
    var_type_mixed += ['real', 'real']

    # set constraints
    cons = [['ineq', constraint_1], ['ineq', constraint_2], ['ineq', constraint_3], ['ineq', constraint_4],
            ['ineq', constraint_5], ['ineq', constraint_6]]

    ####################################################################################
    # run single objectives and save to save_array
    save_array = list()
    methods = list()

    # make dictionary with parameter settings for the GA
    options = {
        'n_bits': 24,
        'n_iter': 400,
        'n_pop': 1000,
        'r_cross': 0.8,
        'max_stall': 10,
        'var_type_mixed': var_type_mixed,
        'mutation_rate_order': 3,
    }

    import_variables = True

    # costs
    if not import_variables:
        ga = GeneticAlgorithm(objective=single_objective_costs, constraints=cons, cons_handler='CND', bounds=bounds,
                              options=options)
        res_costs, design_variables_SO_costs, _ = ga.run()
        np.savetxt(f'design_var_results/{scenario}_SODO_costs.txt', design_variables_SO_costs)
        print(f'res costs: {res_costs}')
    else:
        design_variables_SO_costs = np.loadtxt(f'design_var_results/{scenario}_SODO_costs.txt')
    save_array.append(design_variables_SO_costs)
    methods.append('Costs')

    ####################################################################################
    # run multi-objective with minmax solver

    # change some entries in the options dictionary
    options['n_bits'] = 24
    options['n_pop'] = 350
    options['max_stall'] = 7
    options['r_cross'] = 0.80
    options['aggregation'] = 'minmax'
    options['mutation_rate_order'] = 2

    if not import_variables:
        ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds,
                              options=options)
        _, design_variables_minmax, _ = ga.run()
        np.savetxt(f'design_var_results/{scenario}_min-max.txt', design_variables_minmax)
    else:
        design_variables_minmax = np.loadtxt(f'design_var_results/{scenario}_min-max.txt')
    save_array.append(design_variables_minmax)
    methods.append('minmax')

    ####################################################################################
    # run multi-objective with tetra solver
    options['n_pop'] = 350
    options['tetra'] = True
    options['aggregation'] = 'tetra'
    options['mutation_rate_order'] = 2.5

    if not import_variables:
        ga = GeneticAlgorithm(objective=objective, constraints=cons, cons_handler='CND', bounds=bounds,
                              options=options,
                              start_points_population=[design_variables_SO_costs])
        _, design_variables_tetra, _ = ga.run()
        np.savetxt(f'design_var_results/{scenario}_IMAP.txt', design_variables_tetra)
    else:
        ga = GeneticAlgorithm(objective=objective, constraints=cons, bounds=bounds)  # for solving the evaluation
        design_variables_tetra = np.loadtxt(f'design_var_results/{scenario}_IMAP.txt')
    save_array.append(design_variables_tetra)
    methods.append('IMAP')

    variable = np.array(save_array)
    w, p = objective(variable)  # evaluate objective
    r = ga.solver.request(w, p)  # get aggregated scores to rank them

    # create pandas DataFrame and print it to console
    d = {'Method': methods,
         'Results': r
         }
    print(scenario)
    print(pd.DataFrame(data=d).to_string())
    print()

    t, c, f, s = get_results(variable, methods, scenario)
    d = {'Method': methods,
         'Project duration': t,
         'Costs': c,
         'Fleet util': f,
         'Emissions': s,
         }
    print(scenario)
    print(pd.DataFrame(data=d).to_string())
    print()

    p1_res = pchip_interpolate(x_points_1, p_points_1, t)
    p2_res = pchip_interpolate(x_points_2, p_points_2, c)
    p3_res = pchip_interpolate(x_points_3, p_points_3, f)
    p4_res = pchip_interpolate(x_points_4, p_points_4, s)

    d = {'Method': methods,
         'Project duration': p1_res,
         'Costs': p2_res,
         'Fleet util': p3_res,
         'Emissions': p4_res,
         }
    print(scenario)
    print(pd.DataFrame(data=d).to_string())
    print()

    # create figure that plots all preference curves and the preference scores of the returned results of the GA
    markers = ['x', 'v', '1', 's', '+', 'o']
    fig2, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 10))

    ax1.plot(c1, p1, label='Preference Function')
    for i in range(len(variable)):
        ax1.scatter(t[i], p1_res[i], label=methods[i], marker=markers[i], s=[100])
    ax1.set_ylim((0, 100))
    ax1.set_title('Project Duration')
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Preference score')
    ax1.grid()
    fig2.legend()

    ax2.plot(c2 * 1e-6, p2)
    for i in range(len(variable)):
        ax2.scatter(c[i] * 1e-6, p2_res[i], label=methods[i], marker=markers[i], s=[100])
    ax2.set_ylim((0, 100))
    ax2.set_xlim((0, 110))
    ax2.set_title('Installation cost')
    ax2.set_xlabel('Euros [1e6€]')
    ax2.set_ylabel('Preference score')
    ax2.grid()

    ax3.plot(c3, p3)
    for i in range(len(variable)):
        ax3.scatter(f[i], p3_res[i], label=methods[i], marker=markers[i], s=[100])
    ax3.set_ylim((0, 100))
    ax3.set_title('Fleet utilisation')
    ax3.set_xlabel('Normalised utilisation')
    ax3.set_ylabel('Preference score')
    ax3.grid()

    ax4.plot(c4, p4)
    for i in range(len(variable)):
        ax4.scatter(s[i], p4_res[i], label=methods[i], marker=markers[i], s=[100])
    ax4.set_ylim((0, 100))
    ax4.set_title(r'$CO_2$ emission')
    ax4.set_xlabel(r'$CO_2$ emissions [mt]')
    ax4.set_ylabel('Preference score')
    ax4.grid()

    plt.savefig(f'figures/pref_curves_{scenario}')
    plt.close()
    return


if __name__ == '__main__':
    main('demo')
