from dmg._version import __version__
from dmg.core import calc, data, post, utils
from dmg.core.data import loaders, samplers
from dmg.models import criterion, delta_models, neural_networks, phy_models
from dmg.models.model_handler import ModelHandler
from dmg.models.mts_model_handler import MtsModelHandler

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    '__version__',
    'calc',
    'data',
    'post',
    'utils',
    'loaders',
    'samplers',
    'criterion',
    'delta_models',
    'neural_networks',
    'phy_models',
    'ModelHandler',
    'MtsModelHandler',
]
