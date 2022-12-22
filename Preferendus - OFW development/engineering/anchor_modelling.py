"""
(c) Harold van Heukelum, 2022
"""
from multiprocessing import Pool

import numpy as np
from data_dicts import CONSTANTS, STEEL
from engineering._mp_scrips import h_ult_clay_piled_anchors, h_ult_sand_piled_anchors
from scipy.optimize import fsolve


class EngineeringCalculations:
    """
    Class with anchor utilization calculations
    """

    def __init__(self, soil_database: dict, mooring_database):
        self.soil_data = soil_database
        self.soil_type = soil_database['type']
        self.mooring_data = mooring_database

    def _max_h_clay(self, h, d):
        """Calculate maximum suction-assisted penetration depth SPs in clay"""
        return CONSTANTS['NC'] / (4 * self.soil_data['a_o']) * 1 * (1 - 1 / (1 + 2 * h / d) ** 2) - h / d

    def _max_h_sand(self, h, d):
        """Calculate maximum suction-assisted penetration depth SPs in sand"""
        return d / (2 * self.soil_data['K_tan_delta_o']) - h

    def max_h_suction_anchor(self, mean_diameter: list or np.ndarray, length: list or np.ndarray):
        """
        Calculate maximum suction-assisted penetration length of the SPs
        """
        max_h = np.zeros(len(mean_diameter))
        for d in np.unique(mean_diameter):
            if self.soil_type == 'clay':
                max_h[np.where(mean_diameter == d)] = fsolve(self._max_h_clay, np.array([10]), (d,))
            else:
                max_h[np.where(mean_diameter == d)] = fsolve(self._max_h_sand, np.array([10]), (d,))

        return np.minimum(max_h, length)

    def _solve_ta_ta(self, p, tension_mudline, theta_m, za, d, mu, su, nc=7.6):
        """
        Function that contains the two equations needed for calculating the tension in and the angle of the ML at the
        padeye.

        Reference:
            - Neubecker, S., & Randolph, M. (1995). Performance of embedded anchor chains and consequences for anchor
                design. In Offshore technology conference.
        """
        tension_a, theta = p

        if self.mooring_data['line type'] == 'chain' and self.mooring_data['AWB'] == 0:
            awb = 2.5
        elif self.mooring_data['line type'] == 'chain':
            awb = self.mooring_data['AWB']
        else:
            awb = 1

        if self.soil_type == 'clay':
            za_q = awb * d * nc * su * za
        else:
            za_q = awb * d * nc * self.soil_data['specific_weight'] * za ** 2 / 2

        return (2 * za_q / tension_a) - (theta ** 2 - theta_m ** 2), np.exp(
            mu * (theta - theta_m)) - tension_mudline / tension_a

    def _solve_ta(self, p, tension_mudline, theta_m, theta_a, mu):
        """
        Function that contains the equations needed for calculating the tension in the ML at the padeye.

        Reference:
            - Neubecker, S., & Randolph, M. (1995). Performance of embedded anchor chains and consequences for anchor
                design. In Offshore technology conference.
        """
        tension_a = p
        return np.exp(mu * (theta_a - theta_m)) - tension_mudline / tension_a

    def _solve_theta_a(self, p, tension_a, tension_mudline, theta_m, mu):
        """
        Function that contains the equations needed for calculating the angle of the ML at the padeye.

        Reference:
            - Neubecker, S., & Randolph, M. (1995). Performance of embedded anchor chains and consequences for anchor
                design. In Offshore technology conference.
        """
        theta = p
        return np.exp(mu * (theta - theta_m)) - tension_mudline / tension_a

    def suction_anchor(self, tension_mudline, angle_mudline, d_e: list or np.ndarray, length: list or np.ndarray,
                       t: list or np.ndarray = None):
        """
        Function to calculate the utilization factor of suction anchors.

        References:
            - Houlsby, G. T., & Byrne, B. W. (2005). Design procedures for installation of suction caissons in clay
                and other materials. Proceedings of the Institution of Civil Engineers-Geotechnical Engineering,
                158(2), 75–82.
            - Houlsby, G. T., & Byrne, B. W. (2005). Design procedures for installation of suction caissons in sand.
                Proceedings of the Institution of Civil Engineers-Geotechnical Engineering, 158(3), 135–144
            - ABS. (2013). Offshore anchor data for preliminary design of anchors of floating offshore wind turbines.
                American Bureau of Shipping. Retrieved from https://www.osti.gov/servlets/purl/1178273
            - Randolph, M., & Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
            - Arany, L., & Bhattacharya, S. (2018). Simplified load estimation and sizing of suction anchors for spar
                buoy type floating offshore wind turbines. Ocean Engineering, 159, 348–357.
        """

        if t is None:
            t = d_e / 150

        if self.soil_type == 'clay':
            length = self.max_h_suction_anchor(d_e, length)

        d_i = d_e - 2 * t
        mean_diameter = d_e - t
        weight_anchor = (np.pi * length * mean_diameter * t + np.pi * mean_diameter ** 2 * t / 4) * (
                STEEL['Specific_weight'] - CONSTANTS['W_water'])

        if self.soil_type == 'clay':
            weight_plug = np.pi / 4 * d_i ** 2 * length * self.soil_data['specific_weight']

            external_shaft_fric = np.pi * d_e * length * self.soil_data['a_o'] * self.soil_data['su']
            internal_shaft_fric = np.pi * d_i * length * self.soil_data['a_i'] * self.soil_data['su']
            reverse_end_bearing = 6.7 * self.soil_data['su'] * d_e ** 2 * np.pi / 4

            v_mode_1 = weight_anchor + external_shaft_fric + reverse_end_bearing
            v_mode_2 = weight_anchor + external_shaft_fric + internal_shaft_fric
            v_mode_3 = weight_anchor + external_shaft_fric + weight_plug

            v_max = np.amin([v_mode_1, v_mode_2, v_mode_3], axis=0)
            h_max = length * d_e * 10 * self.soil_data['su']

        else:
            z_e = d_e / (4 * self.soil_data['K_tan_delta_o'])
            z_i = d_i / (4 * self.soil_data['K_tan_delta_i'])

            y_e = np.exp(-length / z_e) - 1 + length / z_e
            y_i = np.exp(-length / z_i) - 1 + length / z_i

            external_shaft_fric = self.soil_data['specific_weight'] * z_e ** 2 * y_e * self.soil_data[
                'K_tan_delta_o'] * np.pi * d_e
            internal_shaft_fric = self.soil_data['specific_weight'] * z_i ** 2 * y_i * self.soil_data[
                'K_tan_delta_i'] * np.pi * d_i

            v_max = weight_anchor + external_shaft_fric + internal_shaft_fric

            nq = np.exp(np.pi * np.tan(np.radians(self.soil_data['angle internal fric']))) * np.tan(
                np.radians(45) + np.radians(self.soil_data['angle internal fric']) / 2) ** 2

            h_max = 0.5 * d_e * nq * self.soil_data['specific weight'] * length ** 2

        if self.soil_type == 'clay':
            rel_pos_pad_eye = 0.5
        else:
            rel_pos_pad_eye = 2 / 3

        angle_mudline = np.radians(angle_mudline)

        tension_pad_eye = np.zeros(len(length))
        angle_pad_eye = np.zeros(len(length))
        for it, lng in enumerate(np.unique(length)):
            x = fsolve(self._solve_ta_ta, np.array([tension_mudline[it], 0.5]),
                       (tension_mudline[it], angle_mudline[it], rel_pos_pad_eye * lng, self.mooring_data['d'],
                        self.mooring_data['mu'], self.soil_data['su'],
                        12), factor=2)
            mask = length == lng
            tension_pad_eye[mask] = x[0]
            angle_pad_eye[mask] = x[1]

        h = np.cos(angle_pad_eye) * tension_pad_eye
        v = np.sin(angle_pad_eye) * tension_pad_eye

        a = length / mean_diameter + 0.5
        b = length / (3 * mean_diameter) + 4.5

        hor_util = h / (h_max / 1.2)
        ver_util = v / (v_max / 1.2)

        return np.power(hor_util, a) + np.power(ver_util, b), h_max, v_max, h, v

    @staticmethod
    def rounder(values):
        """
        Set value to the closest one in a list
        """
        def f(x):
            idx = np.argmin(np.abs(values - x))
            return values[idx]

        return np.frompyfunc(f, 1, 1)

    def fluke_anchors(self, tension_mudline, loc):
        """
        Function to calculate the utilization factor of drag embedded anchors.

        References:
            - ABS. (2013). Offshore anchor data for preliminary design of anchors of floating offshore wind turbines.
                American Bureau of Shipping. Retrieved from https://www.osti.gov/servlets/purl/1178273
        """
        if self.soil_type == 'clay':
            angle_anchor = np.radians(41)
        else:
            angle_anchor = np.radians(32)

        angle_mudline = np.radians(0)

        masses_mk3 = [1, 1.5, 3, 5, 7, 9, 12, 15, 20, 30]
        masses_mk5 = [1.5, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30]
        masses_mk6 = [1.5, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 30]

        utilization = list()

        for i in range(len(loc)):
            x = fsolve(self._solve_ta, np.array([10000]),
                       (tension_mudline[i], angle_mudline, angle_anchor, self.mooring_data['mu'],))
            tension_anchor = x[0]

            if self.soil_type == 'clay' and self.soil_data['su'] >= 10:
                theoretical_mass_mk3 = (tension_anchor / 229.19) ** (1 / 0.92)
                theoretical_mass_mk5 = (tension_anchor / 552.53) ** (1 / 0.92)
                theoretical_mass_mk6 = (tension_anchor / 701.49) ** (1 / 0.93)
            elif self.soil_type == 'clay' and self.soil_data['su'] < 10:
                theoretical_mass_mk3 = (tension_anchor / 161.23) ** (1 / 0.92)
                theoretical_mass_mk5 = (tension_anchor / 392.28) ** (1 / 0.92)
                theoretical_mass_mk6 = (tension_anchor / 509.96) ** (1 / 0.93)
            else:
                theoretical_mass_mk3 = (tension_anchor / 324.42) ** (1 / 0.90)
                theoretical_mass_mk5 = (tension_anchor / 686.49) ** (1 / 0.93)
                theoretical_mass_mk6 = (tension_anchor / 904.21) ** (1 / 0.92)

            if loc[i] < 10:
                mass_a = masses_mk3[loc[i]]
                utilization.append(mass_a / (theoretical_mass_mk3 / 1.3))
            elif 10 <= loc[i] < 22:
                mass_a = masses_mk5[loc[i] - 10]
                utilization.append(mass_a / (theoretical_mass_mk5 / 1.3))
            else:
                mass_a = masses_mk6[loc[i] - 22]
                utilization.append(mass_a / (theoretical_mass_mk6 / 1.3))

        return utilization

    @staticmethod
    def _solve_clay_piled_anchors(x):
        """Function to solve the ultimate horizontal capacity of anchor piles in clay"""
        res = fsolve(h_ult_clay_piled_anchors, np.array([30, 3e3]), args=(x,))
        return res[1]

    @staticmethod
    def _solve_sand_piled_anchors(x):
        """Function to solve the ultimate horizontal capacity of anchor piles in sand"""
        res = fsolve(h_ult_sand_piled_anchors, np.array([20, 3e3]), args=(x,))
        return res[1]

    def piled_anchors(self, tension_mudline, angle_mudline, diameter, length, t=None):
        """
        Function to calculate the utilization factor of anchor piles.

        References:
            - ABS. (2013). Offshore anchor data for preliminary design of anchors of floating offshore wind turbines.
                American Bureau of Shipping. Retrieved from https://www.osti.gov/servlets/purl/1178273
            - Randolph, M., & Gourvenec, S. (2017). Offshore geotechnical engineering. CRC press.
        """
        if t is None:
            t = diameter / 30 if self.soil_type == 'clay' else diameter / 25

        if self.soil_type == 'clay':
            rel_pos_pad_eye = 0.5
        else:
            rel_pos_pad_eye = 2 / 3

        d_i = diameter - 2 * t
        mean_diameter = diameter - t
        weight_anchor = (np.pi * length * mean_diameter * t + np.pi * mean_diameter ** 2 * t / 4) * (
                STEEL['Specific_weight'] - CONSTANTS['W_water'])

        one_big_array = np.zeros((len(length), 7))
        for it, lng in enumerate(np.unique(length)):
            x = fsolve(self._solve_ta_ta, np.array([tension_mudline[it], 0.5]),
                       (tension_mudline[it], angle_mudline[it], rel_pos_pad_eye * lng, self.mooring_data['d'],
                        self.mooring_data['mu'], self.soil_data['su']))
            mask = length == lng
            one_big_array[:, 0][mask] = x[0]
            one_big_array[:, 1][mask] = x[1]

        if self.soil_type == 'clay':
            one_big_array[:, 2] = length
            one_big_array[:, 3] = diameter
            one_big_array[:, 4] = CONSTANTS['NP']
            one_big_array[:, 5] = self.soil_data['su']
            one_big_array[:, 6] = rel_pos_pad_eye * length
            with Pool() as p:
                t_ult = p.map(self._solve_clay_piled_anchors, one_big_array)

            weight_plug = np.pi / 4 * d_i ** 2 * length * self.soil_data['specific_weight']

            external_shaft_fric = np.pi * diameter * length * self.soil_data['a_o'] * self.soil_data['su']
            internal_shaft_fric = np.pi * d_i * length * self.soil_data['a_i'] * self.soil_data['su']
            reverse_end_bearing = 6.7 * self.soil_data['su'] * diameter ** 2 * np.pi / 4

            v_mode_1 = weight_anchor + external_shaft_fric + reverse_end_bearing
            v_mode_2 = weight_anchor + external_shaft_fric + internal_shaft_fric
            v_mode_3 = weight_anchor + external_shaft_fric + weight_plug

            v_ult = np.amin([v_mode_1, v_mode_2, v_mode_3], axis=0)
        else:
            angle_internal_fric = np.radians(self.soil_data['angle internal friction'])
            one_big_array[:, 2] = length
            one_big_array[:, 3] = diameter
            one_big_array[:, 4] = (1 + np.sin(angle_internal_fric)) / (1 - np.sin(angle_internal_fric))
            one_big_array[:, 5] = self.soil_data['specific weight']
            one_big_array[:, 6] = rel_pos_pad_eye * length
            with Pool() as p:
                t_ult = p.map(self._solve_sand_piled_anchors, one_big_array)

            z_e = diameter / (4 * self.soil_data['K_tan_delta_o'])
            z_i = d_i / (4 * self.soil_data['K_tan_delta_i'])

            y_e = np.exp(-length / z_e) - 1 + length / z_e
            y_i = np.exp(-length / z_i) - 1 + length / z_i

            external_shaft_fric = self.soil_data['specific_weight'] * z_e ** 2 * y_e * self.soil_data[
                'K_tan_delta_o'] * np.pi * diameter
            internal_shaft_fric = self.soil_data['specific_weight'] * z_i ** 2 * y_i * self.soil_data[
                'K_tan_delta_i'] * np.pi * d_i

            v_ult = weight_anchor + external_shaft_fric + internal_shaft_fric

        v_t = one_big_array[:, 0] * np.sin(one_big_array[:, 1])
        util_v = v_t / (np.array(v_ult) / 1.3)
        util_t = one_big_array[:, 0] / (np.array(t_ult) / 1.3)

        return np.amax(np.array([util_v, util_t]), axis=0)


if __name__ == '__main__':
    soil = 'clay'

    if soil == 'clay':
        soil_data = {
            'type': 'clay',
            'su': 50,  # kPa
            'a_i': 0.64,
            'a_o': 0.64,
            'specific_weight': 9,  # kN/m3
        }
    else:
        soil_data = {
            'type': 'sand',
            'specific_weight': 9,  # kN/m3
            'K_tan_delta_i': 5,
            'K_tan_delta_o': 7,
            'angle internal fric': 33,
            'cohesion': 50,  # kPa; http://www.geotechdata.info/parameter/cohesion
            'fric con-soil': 0.6
        }

    mooring_data = {
        'line type': 'chain',
        'd': 0.127,  # m
        'mu': 0.25,  # -
        'AWB': 2.5  # -
    }

    eng_class = EngineeringCalculations(soil_data, mooring_data)
    # print(eng_class.suction_anchor([4800], [0], np.array([5.75]), np.array([18.5])))
    # print(eng_class.gravity_anchor(23100, 0, np.array([22, 25]), np.array([12, 10])))
    # print(eng_class.fluke_anchors(3800, 0, np.array([22, 25]), np.array([2, 6])))
    # print(eng_class.piled_anchors([4200, 4200], [0, 0], np.array([2.03, 2.5]), np.array([2.66, 20])))
