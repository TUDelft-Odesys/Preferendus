"""
Copyright (c) 2022. Harold Van Heukelum

currents: https://map.emodnet-physics.eu
"""
import csv
import glob
import os
import pathlib
import statistics
from calendar import month_abbr
from pprint import pprint

import numpy as np
import pandas as pd
import xarray
from matplotlib import pyplot as plt
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_3P
from windrose import WindroseAxes

HERE = pathlib.Path(__file__).parent


class EnvironmentalAnalysis:
    """
    class to do the environmental analysis
    """

    def __init__(self, project: str = None):
        if project is None:
            self.project = input('For which project is this analysis: ')
        else:
            self.project = project

        path = f'{HERE}//env_analysis/results/{self.project}'
        assert os.path.isdir(path), 'No environmental data for specified project. Please download data first'

        u100 = list()  # eastward component
        v100 = list()  # northward component
        mean_wave_direction = list()
        mean_zero_crossing_period = list()
        peak_wave_period = list()
        significant_height_wave_swell = list()
        significant_height_swell = list()
        significant_height_wind_waves = list()
        wind_months = list()
        wave_months = list()
        annual_max_waves = list()

        self.p80 = 1 - 1 / (5 * 365 * 24)
        self.p98 = 1 - 1 / (50 * 365 * 24)
        self.p99 = 1 - 1 / (100 * 365 * 24)

        self.df_ns = None
        self.df_we = None

        list_dir = os.listdir(f'{HERE}/env_analysis/results/{self.project}')
        for file in list_dir:
            if '.csv' in file:
                if '_NS' in file:
                    self.df_ns = pd.read_csv(f'./env_analysis/results/{self.project}/{file}', delimiter=';')
                else:
                    self.df_we = pd.read_csv(f'./env_analysis/results/{self.project}/{file}', delimiter=';')
            elif '.nc' in file:
                dataset = xarray.open_dataset(f'./env_analysis/results/{self.project}/{file}', decode_times=True)

                long = dataset.longitude[0]
                lat = dataset.latitude[0]

                if 'wind' in file or 'default' in file:
                    u100.append(dataset.sel(longitude=long, latitude=lat).u100.values.tolist())
                    v100.append(dataset.sel(longitude=long, latitude=lat).v100.values.tolist())
                    dates = dataset.sel(longitude=long, latitude=lat).u100.time.values
                    wind_months.append(dates.astype('datetime64[M]').astype(int) % 12 + 1)

                elif 'waves' in file or 'default' in file:
                    mean_wave_direction.append(dataset.sel(longitude=long, latitude=lat).mwd.values.tolist())
                    mean_zero_crossing_period.append(dataset.sel(longitude=long, latitude=lat).mp2.values.tolist())
                    peak_wave_period.append(dataset.sel(longitude=long, latitude=lat).pp1d.values.tolist())
                    significant_height_wave_swell.append(dataset.sel(longitude=long, latitude=lat).swh.values.tolist())
                    significant_height_swell.append(dataset.sel(longitude=long, latitude=lat).shts.values.tolist())
                    significant_height_wind_waves.append(dataset.sel(longitude=long, latitude=lat).shww.values.tolist())
                    dates = dataset.sel(longitude=long, latitude=lat).swh.time.values
                    wave_months.append(dates.astype('datetime64[M]').astype(int) % 12 + 1)

                    annual_max_waves.append(dataset.resample(time='A').max('time').swh.values.tolist()[0][0][0])
                    annual_max_waves.append(dataset.resample(time='A').max('time').swh.values.tolist()[1][0][0])
                elif 'custom' in file:
                    raise AttributeError('Unforeseen situation, analysis not yet build for custom files')
            else:
                continue

        self.show_list_dir = list_dir

        if u100 and v100:
            u100 = [j for sub in u100 for j in sub]
            v100 = [j for sub in v100 for j in sub]
            self.wind_months = [j for sub in wind_months for j in sub]
            self.ws, self.wd = self.get_wind_info(np.array(u100), np.array(v100))
            self.wind_analysis = True
        else:
            self.wind_analysis = False

        if significant_height_wave_swell:
            self.swh = [j for sub in significant_height_wave_swell for j in sub]
            self.wpp = [j for sub in peak_wave_period for j in sub]
            self.wave_months = [j for sub in wave_months for j in sub]
            self.wave_analysis = True
            self.annual_max_swh = statistics.mode(np.round_(annual_max_waves, 1))
        else:
            self.wave_analysis = False

        if self.df_ns is None and self.df_we is None:
            self.currents_analysis = False
        elif self.df_ns is None or self.df_we is None:
            print('Insufficient data for sea currents analysis, so will be skipped')
            self.currents_analysis = False
        else:
            self.currents_analysis = True

    def run(self, num_sectors: int = 16, dev_mode: bool = False):
        """
        Run the analysis
        :param num_sectors: number of sectors in which the wind rose needs to be divided
        :param dev_mode: development mode. Do not use if you do not know what it is
        :return:
        """

        if dev_mode is False:
            ip = input('Remove previous figures? y/n ')
            ip2 = input('Remove previous csv files? y/n ')
            for item in ['Wind', 'Waves']:
                if ip == 'y':
                    files = glob.glob(f'./env_analysis/figures/{self.project}/{item}/*.png')
                    for f in files:
                        os.remove(f)
                if ip2 == 'y':
                    files = glob.glob(f'./env_analysis/figures/{self.project}/{item}/csv/*.csv')
                    for f in files:
                        os.remove(f)
        else:
            for item in ['Wind', 'Waves']:
                files = glob.glob(f'./env_analysis/figures/{self.project}/{item}/*.png')
                for f in files:
                    os.remove(f)
                files = glob.glob(f'./env_analysis/figures/{self.project}/{item}/csv/*.csv')
                for f in files:
                    os.remove(f)

        data = {
            'U_P80': 0,
            'U_P98': 0,
            'U_P99': 0,
            'H_1': self.annual_max_swh,
            'H_P80': 0,
            'H_P98': 0,
            'H_P99': 0,
            'PP_1': self.get_peak_period(self.annual_max_swh),
            'PP_P80': 0,
            'PP_P98': 0,
            'PP_P99': 0,
            'C_NCM': 0,
            'C_1': 0,
            'C_50': 0,
        }

        if self.wind_analysis is False or self.wave_analysis is False:
            raise Exception("Run can currently only be used for both wind and waves env conditions "
                            "simultaneously")

        p = [self.p80, self.p98, self.p99]
        p_str = ['P80', 'P98', 'P99']

        dist_wind = self.get_dist(self.ws)
        dist_wave = self.get_dist(self.swh)

        self.create_histograms_wind_per_sector(n_sector=num_sectors, wind_speed=self.ws, wind_direction=self.wd,
                                               distribution=dist_wind)
        self.seasonal_wind(self.ws, self.wind_months)

        self.plot_histogram_waves(data=self.swh, bin_count=80, distribution=dist_wave)
        self.seasonal_hs(self.swh, self.wave_months)

        q_wind = self.calculate_extremes(dist_wind, p)
        q_wave = self.calculate_extremes(dist_wave, p)

        for it in range(len(p)):
            data[f'U_{p_str[it]}'] = q_wind[it]

            data[f'H_{p_str[it]}'] = q_wave[it]
            peak_period = self.get_peak_period(q_wave[it])
            data[f'PP_{p_str[it]}'] = peak_period

        if self.currents_analysis:
            ncm, c_1, c_50 = self.get_currents()
            data['C_NCM'] = ncm
            data['C_1'] = c_1
            data['C_50'] = c_50
        else:
            data['C_NCM'] = 1.4
            data['C_1'] = 1.4
            data['C_50'] = 1.7

        pprint(data)
        w = csv.writer(open(f"./env_analysis/figures/{self.project}/output.csv", "w"))
        for key, val in data.items():
            w.writerow([key, val])
        return data

    def get_peak_period(self, quantile):
        """

        :return:
        """
        data_pp = np.array(self.wpp)

        lst = np.asarray(self.swh)
        idx = (np.abs(lst - quantile)).argmin()
        return data_pp[idx]

    def get_wind_info(self, eastward, northward):
        """
        Extract true wind speed and direction from Eastward and Northward dataset

        :param eastward: Dataset with Eastward wind speed
        :param northward: Dataset with Northward wind speed
        :return: wind speed, wind direction
        """
        eastward = np.array(eastward)
        northward = np.array(northward)
        wind_speed = np.sqrt(np.power(eastward, 2) + np.power(northward, 2))

        mask_u100_min = eastward < 0
        mask_v100_min = northward < 0
        mask_both = np.logical_and(mask_u100_min, mask_v100_min)

        wind_direction = np.degrees(np.arctan(eastward / northward))
        wind_direction[mask_v100_min] = 180 + np.degrees(np.arctan(eastward[mask_v100_min] / northward[mask_v100_min]))
        wind_direction[mask_u100_min] = 360 + np.degrees(np.arctan(eastward[mask_u100_min] / northward[mask_u100_min]))
        wind_direction[mask_both] = 180 + np.degrees(np.arctan(eastward[mask_both] / northward[mask_both]))

        n_sector = 16
        wind_direction = wind_direction.flatten()
        wind_speed = wind_speed.flatten()

        ax = WindroseAxes.from_ax()
        ax.box(wind_direction, wind_speed, bins=np.arange(0, 36, 4), normed=True, nsector=n_sector)
        ax.set_legend(ncol=2)
        plt.savefig(fname=f'./env_analysis/figures/{self.project}/Wind/windrose.png', format='png')

        return wind_speed, wind_direction

    def get_dist(self, data):
        """

        :return:
        """
        params = Fit_Weibull_3P(failures=data, show_probability_plot=False, print_results=False)
        weibull_distribution = Weibull_Distribution(alpha=params.alpha, beta=params.beta, gamma=params.gamma)
        return weibull_distribution

    @staticmethod
    def calculate_extremes(distribution, p):
        """
        Calculate extreme values for a certain distribution

        :param distribution: Probability distribution
        :param p: Probability
        :return: quantile
        """
        quantile = distribution.quantile(p)
        return quantile

    def plot_histogram_wind(self, data, bin_count, heading, distribution):
        """
        Function to make the figures for the different locations and hub heights. Plots the histogram of the dataset,
        the Weibull pdf at z = 10m, and the Weibull pdf at hub height.

        :return: None
        """
        q80 = self.calculate_extremes(distribution, self.p80)
        q98 = self.calculate_extremes(distribution, self.p98)
        q99 = self.calculate_extremes(distribution, self.p99)

        u = np.linspace(0, q99 + 1, num=51)  # wind speed array
        w = distribution.PDF(xvals=u, show_plot=False)
        w_cdf = distribution.CDF(xvals=u, show_plot=False)

        fig, ax1 = plt.subplots()  # declare figure

        color = 'tab:red'
        ax1.set_xlabel('Wind velocity [m/s]')
        ax1.set_ylabel('Number of occurrences', color=color)
        ax1.hist(x=data, bins=bin_count, color=color, rwidth=0.85)  # plot histogram of dataset
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:green'
        ax2.plot(u, w, color=color, label=f'Weibull pdf')  # plot pdf at hub height
        ax2.vlines(q80, 0, max(w), label='P80', color='y')
        ax2.vlines(q98, 0, max(w), label='P98', color='r')
        ax2.vlines(q99, 0, max(w), label='P99', color='g')
        ax2.set_ybound(lower=0)

        plt.legend()
        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Wind/histo_wind_speed_dir={heading}.png',
                    format='png')

        plt.figure()
        fig, ax1 = plt.subplots()  # declare figure

        color = 'tab:red'
        ax1.set_xlabel('Wind velocity [m/s]')
        ax1.set_ylabel('Number of occurrences', color=color)
        ax1.hist(x=data, bins=bin_count, color=color, rwidth=0.85)  # plot histogram of dataset
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:green'
        ax2.plot(u, w_cdf, color=color, label=f'Weibull cdf')  # plot pdf at hub height
        ax2.hlines(self.p80, 0, max(u), label='P80', color='y')
        ax2.hlines(self.p98, 0, max(u), label='P98', color='r')
        ax2.hlines(self.p99, 0, max(u), label='P99', color='g')
        ax2.set_ybound(lower=0)

        plt.legend()
        fig.savefig(
            fname=f'./env_analysis/figures/{self.project}/Wind/histo_wind_speed_dir={heading}_cdf.png',
            format='png')
        return

    def plot_histogram_waves(self, data, bin_count, distribution):
        """
        Function to make the figures for the different locations and hub heights. Plots the histogram of the dataset,
        the Weibull pdf at z = 10m, and the Weibull pdf at hub height.

        :return: None
        """
        q80 = self.calculate_extremes(distribution, self.p80)
        q98 = self.calculate_extremes(distribution, self.p98)
        q99 = self.calculate_extremes(distribution, self.p99)

        h = np.linspace(0, q99 + 1, num=51)  # wave height array
        w = distribution.PDF(xvals=h, show_plot=False)
        w_cdf = distribution.CDF(xvals=h, show_plot=False)

        # figure with PDF
        plt.figure()
        fig, ax1 = plt.subplots()  # declare figure

        ax1.set_xlabel('Significant wave height [m]')
        ax1.set_ylabel('Number of occurrences')
        ax1.hist(x=data, bins=bin_count, color='r', rwidth=0.85)  # plot histogram of dataset
        ax1.set_ybound(lower=0)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.plot(h, w, label=f'Weibull PDF', color='b')  # plot pdf at hub height
        ax2.vlines(q80, 0, max(w), label='P80')
        ax2.vlines(q98, 0, max(w), label='P98')
        ax2.vlines(q99, 0, max(w), label='P99')
        ax2.set_ybound(lower=0)

        # figure with CDF
        plt.legend()
        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Waves/histo_wave_height.png',
                    format='png')

        plt.figure()
        fig, ax1 = plt.subplots()  # declare figure

        ax1.set_xlabel('Significant wave height [m]')
        ax1.set_ylabel('Number of occurrences')
        ax1.hist(x=data, bins=bin_count, color='r', rwidth=0.85)  # plot histogram of dataset
        ax1.set_ybound(lower=0)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.plot(h, w_cdf, label=f'Weibull CDF')  # plot pdf at hub height
        ax2.vlines(q80, 0, max(w), label='P80')
        ax2.vlines(q98, 0, max(w), label='P98')
        ax2.vlines(q99, 0, max(w), label='P99')
        ax2.set_ybound(lower=0)

        plt.legend()
        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Waves/histo_wave_height_cdf.png',
                    format='png')  # save figures as png-files

        return

    def create_histograms_wind_per_sector(self, n_sector, wind_speed, wind_direction, distribution):
        """
        Create histograms for all wind directions and for certain sectors
        :return: None
        """
        self.plot_histogram_wind(data=wind_speed.tolist(), bin_count=60, heading='all', distribution=distribution)

        window = 360 / n_sector
        box_list = []
        angle = 0
        for _ in range(n_sector):
            range_min = angle - window / 2
            if range_min < 0:
                range_min += 360
            range_max = angle + window / 2
            if angle == 0:
                mask = np.logical_or(wind_direction > range_min, wind_direction < range_max)
            else:
                mask = np.logical_and(wind_direction > range_min, wind_direction < range_max)
            box_list.append(wind_speed[mask].tolist())
            self.plot_histogram_wind(data=wind_speed[mask].tolist(), bin_count=60, heading=angle,
                                     distribution=distribution)
            angle += window

        assert sum(len(item) for item in box_list) == len(
            wind_direction), 'Problem with binning of wind speeds, number ' \
                             'of items in bins not equal tot total number ' \
                             'of elements in wind speed list'
        return

    def seasonal_hs(self, hs, months):
        """
        Analyse wind data per month. Creates csv-files with data per month and makes boxplot with data

        :param hs: Dataset with significant wave height
        :param months: Dataset with corresponding month numbering
        :return: None
        """
        wave_data = list()

        for j in range(12):
            wave_per_month = list()
            for i in range(len(months)):
                if months[i] == j + 1:
                    wave_per_month.append(hs[i])
            wave_data.append(wave_per_month)

        months_positions = np.arange(1, 13)
        for month in months_positions:
            df = pd.DataFrame({month_abbr[month]: wave_data[month - 1]})
            df.to_csv(f'./env_analysis/figures/{self.project}/Waves/csv/Wave_data_month_{month}.csv')

        fig, ax = plt.subplots()
        vp = ax.boxplot(wave_data, positions=months_positions, widths=0.3, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": "green", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                  "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})

        ax.set(xlim=(0, 13), xticks=np.arange(1, 13), title='Seasonal significant wave height', xlabel='Month',
               ylabel=r'Significant wave height $H_s$ [m]')
        ax.grid()

        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Waves/seasonal_hs_boxplot.png', format='png')
        return

    def seasonal_wind(self, wind, months):
        """
        Analyse wind data per month. Creates csv-files with data per month and makes boxplot with data

        :param wind: Dataset with wind speeds
        :param months: Dataset with corresponding month numbering
        :return: None
        """
        wind_data = list()

        for j in range(12):
            wind_per_month = list()
            for i in range(len(months)):
                if months[i] == j + 1:
                    wind_per_month.append(wind[i])
            wind_data.append(wind_per_month)

        months_positions = np.arange(1, 13)

        for month in months_positions:
            df = pd.DataFrame({month_abbr[month]: wind_data[month - 1]})
            df.to_csv(f'./env_analysis/figures/{self.project}/Wind/csv/Wind_data_month_{month}.csv')

        fig, ax = plt.subplots()
        ax.boxplot(wind_data, positions=months_positions, widths=0.3, patch_artist=True,
                        showmeans=False, showfliers=False,
                        medianprops={"color": "green", "linewidth": 0.5},
                        boxprops={"facecolor": "C0", "edgecolor": "white",
                                  "linewidth": 0.5},
                        whiskerprops={"color": "C0", "linewidth": 1.5},
                        capprops={"color": "C0", "linewidth": 1.5})

        ax.set(xlim=(0, 13), xticks=np.arange(1, 13), title='Seasonal significant wave height', xlabel='Month',
               ylabel=r'Significant wave height $H_s$ [m]')
        ax.grid()

        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Wind/seasonal_wind_speed_boxplot.png', format='png')
        return

    def get_waves_for_wind_speed(self, target_wind_speed: float):
        """

        :return:
        """
        wind_speed = np.array(self.ws)
        wave_height = np.array(self.swh)
        wave_period = np.array(self.wpp)

        mask1 = wind_speed > target_wind_speed - 0.1
        mask2 = wind_speed < target_wind_speed + 0.1
        mask = np.logical_and(mask1, mask2)

        data = dict()
        data['WS'] = wind_speed[mask]
        data['WH'] = wave_height[mask]
        data['WP'] = wave_period[mask]

        pf = pd.DataFrame.from_dict(data)
        pf.to_csv(f'./env_analysis/results/wind_wave_correlations/correlation_ws{round(target_wind_speed, 2)}.csv')
        return

    def get_currents(self):
        """

        :return:
        """
        self.df_ns.columns = ['time', 'speed']
        self.df_we.columns = ['time', 'speed']
        data_ns = self.df_ns[np.abs(self.df_ns['speed']) < 5]['speed'].values
        data_we = self.df_we[np.abs(self.df_we['speed']) < 5]['speed'].values

        currents_speed = np.sqrt(np.power(data_ns, 2) + np.power(data_we, 2))

        currents_speed = currents_speed.flatten()

        params = Fit_Weibull_3P(failures=currents_speed, show_probability_plot=False, print_results=False)
        weibull_distribution = Weibull_Distribution(alpha=params.alpha, beta=params.beta, gamma=params.gamma)

        c = np.linspace(0, 1)  # wind speed array
        p = weibull_distribution.PDF(xvals=c, show_plot=False)

        fig, ax1 = plt.subplots()  # declare figure

        color = 'tab:red'
        ax1.set_xlabel('Currents velocity [m/s]')
        ax1.set_ylabel('Number of occurrences', color=color)
        ax1.hist(x=currents_speed, bins=24, color=color, rwidth=0.85)  # plot histogram of dataset
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:green'
        ax2.plot(c, p, color=color, label=f'Weibull pdf')  # plot pdf at hub height
        ax2.set_ybound(lower=0)

        fig.savefig(fname=f'./env_analysis/figures/{self.project}/Currents/histo_currents_speed.png',
                    format='png')

        ncm = np.mean(currents_speed)
        c_1 = max(currents_speed)
        c_50 = self.calculate_extremes(weibull_distribution, self.p98)

        return ncm, c_1, c_50


if __name__ == '__main__':
    analysis = EnvironmentalAnalysis(project='Demo')
    analysis.run(dev_mode=True)
    # analysis.get_monthly_workability({'wind': [10.8, 12, 15, 20],
    #                                   'waves': [1, 1.75, 2, 2.5, 3]}
    #                                  )
    # for ws in [9, 11, 13, 25]:
    #     analysis.get_waves_for_wind_speed(ws)
    # analysis.get_currents()
