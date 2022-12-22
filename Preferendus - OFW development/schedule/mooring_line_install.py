"""
(c) Harold van Heukelum, 2022
"""


class GetMooringLegInstallSchedule:
    """
    Get data ML install
    """

    def __init__(self, mapping, times: dict):
        preparation = mapping['MLprep']
        preparation.at[1, 'time'] = times['transit home - marshalling yard']
        preparation.at[3, 'time'] = times['transit to site']
        self.mobilization = preparation

        self.ml_install_poly = mapping['MLpolyinstall']
        self.ml_install_chain = mapping['MLchaininstall']

        demobilization = mapping['MLdemob']
        demobilization.at[0, 'time'] = times['transit to site']
        demobilization.at[2, 'time'] = times['transit home - marshalling yard']
        self.demobilization = demobilization

        self.building_blocks = self._building_blocks_time()

    def _building_blocks_time(self):
        ret = dict()
        ret['mobilization'] = self.mobilization
        ret['ml_install_poly'] = self.ml_install_poly
        ret['ml_install_chain'] = self.ml_install_chain
        ret['demobilization'] = self.demobilization
        return ret
