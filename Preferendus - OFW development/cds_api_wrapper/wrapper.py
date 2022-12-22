"""
Copyright (c) 2022. Harold Van Heukelum
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
"""
import pathlib

import cdsapi

HERE = pathlib.Path(__file__).parent


class SendRequest:
    """
    Code to request environmental data from CDS.
    """

    def __init__(self, variables=None):
        if variables is None:
            self.identifier = 'default'
            self.variables = [
                '100m_u_component_of_wind', '100m_v_component_of_wind', 'mean_wave_direction',
                'mean_zero_crossing_wave_period', 'peak_wave_period',
                'significant_height_of_combined_wind_waves_and_swell',
                'significant_height_of_total_swell', 'significant_height_of_wind_waves',
            ]
        elif variables == "wind":
            self.identifier = 'wind'
            self.variables = [
                '100m_u_component_of_wind', '100m_v_component_of_wind',
            ]
        elif variables == "waves":
            self.identifier = 'waves'
            self.variables = [
                'mean_wave_direction', 'mean_zero_crossing_wave_period', 'peak_wave_period',
                'significant_height_of_combined_wind_waves_and_swell', 'significant_height_of_total_swell',
                'significant_height_of_wind_waves',
            ]
        else:
            self.identifier = 'custom'
            assert type(variables) == list, 'Wrong input for CDS API wrapper. Custom input should be list!'
            self.variables = variables

        yellow_text = '\033[93m'
        reset_text_color = '\033[0m'
        print(yellow_text + 'The gathering of environmental data takes 30 minutes up to 4 hours. Please do not run '
                            'this package in the same thread as your main program. Preferably, schedule it to run at '
                            'night.' + reset_text_color)

    def request(self, location: dict = None, n_years: int = 42):
        """
        Send request to API, wait for response, and store result

        :param location: Location of which one wants the information
        :param n_years: How many years of data one wants
        :return: None
        """
        c = cdsapi.Client()

        request_dict = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': self.variables,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ]
        }

        if location is not None:
            assert location.get('North') > location.get('South'), 'North coordinate is smaller than South coordinate'
            assert location.get('East') > location.get('West'), 'East coordinate is smaller than West coordinate'
            request_dict['area'] = [
                location.get('North'), location.get('West'), location.get('South'),
                location.get('East'),
            ]

        available_years = [
            '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ]

        years = available_years[-n_years:]

        for i in range(0, len(years), 2):
            try:
                request_dict.update({'year': [years[i], years[i + 1]]})
                c.retrieve('reanalysis-era5-single-levels', request_dict,
                           f'{HERE}/results/download_{years[i]}_{years[i + 1]}_{self.identifier}.nc')
            except IndexError:
                request_dict.update({'year': years[i]})
                c.retrieve('reanalysis-era5-single-levels', request_dict,
                           f'{HERE}/results/download_{years[i]}_{self.identifier}.nc')
        return


if __name__ == '__main__':
    # run wave requests for all years
    wrapper = SendRequest(variables='waves')
    loc = {
        'North': 0,
        'South': 0,
        'West': 0,
        'East': 0
    }
    wrapper.request(location=loc)

    # run wind requests for all years
    wrapper = SendRequest(variables='wind')
    loc = {
        'North': 0,
        'South': 0,
        'West': 0,
        'East': 0
    }
    wrapper.request(location=loc)
