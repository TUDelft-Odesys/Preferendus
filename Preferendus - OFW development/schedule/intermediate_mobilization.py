"""
(c) Harold van Heukelum, 2022
"""


class GetInterMobSchedule:
    """
    Get data intermediate mobilizations
    """

    def __init__(self, mapping, times):
        inter_mob = mapping['InterMob']
        inter_mob.at[0, 'time'] = times['transit to site']
        inter_mob.at[2, 'time'] = times['transit to site']
        self.inter_mob = inter_mob

        self.offshore_bunkering = mapping['OffshoreBunkering']

        self.building_blocks = self._building_blocks_time()

    def _building_blocks_time(self):
        ret = dict()
        ret['intermediate_mobilization'] = self.inter_mob
        ret['offshore_bunkering'] = self.offshore_bunkering
        return ret
