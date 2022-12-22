"""
(c) Harold van Heukelum, 2022
"""


class DragEmbeddedAnchorsStevtensioningScheduler:
    """
    Get data DEA incl. stev-tensioning
    """

    def __init__(self, mapping, times: dict):
        preparation = mapping['DEASprep']
        preparation.at[1, 'time'] = times['transit home - marshalling yard']
        preparation.at[3, 'time'] = times['transit to site']
        self.mobilization = preparation

        self.installation = mapping['DEASinstall']

        demobilization = mapping['DEASdemob']
        demobilization.at[0, 'time'] = times['transit to site']
        demobilization.at[2, 'time'] = times['transit home - marshalling yard']
        self.demobilization = demobilization

        self.building_blocks = self._building_blocks_time()

    def _building_blocks_time(self):
        ret = dict()
        ret['mobilization'] = self.mobilization
        ret['anchor_installation'] = self.installation
        ret['demobilization'] = self.demobilization
        return ret
