# src/dmg/models/__init__.py
from . import criterion, delta_models, neural_networks, phy_models
from .model_handler import ModelHandler

try:
    from .mts_model_handler import MtsModelHandler
except ImportError:
    MtsModelHandler = None

__all__ = [
    'criterion',
    'delta_models',
    'neural_networks',
    'phy_models',
    'ModelHandler',
    'MtsModelHandler',
]
