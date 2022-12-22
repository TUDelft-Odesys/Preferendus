"""
(c) Harold van Heukelum, 2022
"""


class GetTensioningSchedule:
    """
    Get data ML stev-tensioning
    """

    def __init__(self, mapping, times: dict):
        preparation = mapping['TENprep']
        preparation.at[1, 'time'] = times['transit home - marshalling yard']
        preparation.at[2, 'time'] = times['transit to site']
        self.mobilization = preparation

        self.tensioning = mapping['TEN']

        demobilization = mapping['TENdemob']
        demobilization.at[0, 'time'] = times['transit home - marshalling yard']
        self.demobilization = demobilization

        self.building_blocks = self._building_blocks_time()

    def _building_blocks_time(self):
        ret = dict()
        ret['mobilization'] = self.mobilization
        ret['tensioning'] = self.tensioning
        ret['demobilization'] = self.demobilization
        return ret
