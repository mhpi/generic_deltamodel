from dMG._version import __version__
from dMG.core import calc, data, post, utils
from dMG.models import criterion, deltamodel, neural_networks, phy_models
from dMG.models.model_handler import ModelHandler

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    '__version__',
    'calc',
    'data',
    'post',
    'utils',
    'criterion',
    'deltamodel',
    'neural_networks',
    'phy_models',
    'ModelHandler',
]