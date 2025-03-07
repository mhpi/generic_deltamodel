from ._version import __version__
from .core import post
from .core.calc.metrics import Metrics
from .core.data import loaders, samplers
from .core.utils.factory import (
    import_data_loader,
    import_data_sampler,
    import_phy_model,
    import_trainer,
    load_loss_func,
    load_nn_model,
)
from .core.utils.path import PathBuilder
from .models import criterion, neural_networks, phy_models
from .models.differentiable_model import DeltaModel
from .models.model_handler import ModelHandler
from .trainers.base import BaseTrainer
from .trainers.ms_trainer import MsTrainer
from .trainers.trainer import Trainer

# In case setuptools scm says version is 0.0.0
assert not __version__.startswith('0.0.0')

__all__ = [
    'Metrics',
    'loaders',
    'samplers',
    'post',
    'PathBuilder',
    'import_phy_model',
    'import_data_loader',
    'import_data_sampler',
    'import_trainer',
    'load_loss_func',
    'import_nn_model',
    'criterion',
    'neural_networks',
    'phy_models',
    'DeltaModel',
    'ModelHandler',
    'BaseTrainer',
    'Trainer',
    'MsTrainer',
]
