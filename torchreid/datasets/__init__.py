from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501 import Market1501
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17
from .viper import VIPeR
from .grid import GRID
from .cuhk01 import CUHK01
from .prid450s import PRID450S
from .ilids import iLIDS
from .sensereid import SenseReID
from .veri776 import VeRi776
from .aic19_track2 import Aic19Track2
from .pku_vehicle_id import VehicleID
from .boxcars21k import Boxcars21k
from .compcars_sv_reid import CompcarsSurvReid

from .mars import Mars
from .ilidsvid import iLIDSVID
from .prid2011 import PRID2011
from .dukemtmcvidreid import DukeMTMCVidReID


__imgreid_factory = {
    'veri776': VeRi776,
    'vehicleID': VehicleID,
    'aic19track2': Aic19Track2,
    'boxcars21k': Boxcars21k,
    'compcars_reid': CompcarsSurvReid,
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'prid450s': PRID450S,
    'ilids': iLIDS,
    'sensereid': SenseReID,
}


__vidreid_factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)
