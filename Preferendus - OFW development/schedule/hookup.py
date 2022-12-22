"""
(c) Harold van Heukelum, 2022
"""


class GetHookupSchedule:
    """
    Get data hookup
    """

    def __init__(self, mapping_hookup, mapping_towing, hu_type, transit_times: dict):
        mobilization_hookup = mapping_hookup['HUprep']
        mobilization_hookup.at[1, 'time'] = transit_times['transit home - marshalling yard']
        mobilization_hookup.at[3, 'time'] = transit_times['transit to site']
        self.mobilization_hookup = mobilization_hookup

        mobilization_towing = mapping_towing['TOWprep']
        mobilization_towing.at[1, 'time'] = transit_times['transit home - fabrication yard']
        self.mobilization_towing = mobilization_towing

        if hu_type == 'full':
            hookup = mapping_hookup['HUfull']
            hookup.at[9, 'time'] = transit_times['in-site']
        else:
            hookup = mapping_hookup['HUchain_chain']
            hookup.at[6, 'time'] = transit_times['in-site']

        towing = mapping_towing['TOWfwt']
        towing.at[2, 'time'] = transit_times['towing']
        if hu_type == 'full':
            towing.at[3, 'time'] = sum(hookup.iloc[2:6]['time'])  # id 2-5 of HUfull table in Excel
        else:
            towing.at[3, 'time'] = sum(hookup.iloc[2:5]['time'])  # id 2-4 of HUchain_chain table in Excel
        towing.at[4, 'time'] = transit_times['transit site - fabrication yard']

        self.hookup = hookup
        self.towing = towing

        demobilization_hookup = mapping_hookup['HUdemob']
        demobilization_hookup.at[0, 'time'] = transit_times['transit to site']
        demobilization_hookup.at[2, 'time'] = transit_times['transit home - marshalling yard']
        self.demobilization_hookup = demobilization_hookup

        demobilization_towing = mapping_towing['TOWdemob']
        demobilization_towing.at[0, 'time'] = transit_times['transit home - fabrication yard']
        self.demobilization_towing = demobilization_towing

        self.building_blocks = self._building_blocks_time()

    def _building_blocks_time(self):
        ret = dict()
        ret['mobilization HUV'] = self.mobilization_hookup
        ret['mobilization Tug'] = self.mobilization_towing
        ret['hookup'] = self.hookup
        ret['towing'] = self.towing
        ret['demobilization HUV'] = self.demobilization_hookup
        ret['demobilization Tug'] = self.demobilization_towing
        return ret
