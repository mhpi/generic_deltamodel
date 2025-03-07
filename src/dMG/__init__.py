from dMG._version import __version__

from dMG.core import post
from dMG.core.calc.metrics import Metrics
from dMG.core.data import loaders, samplers
from dMG.core.utils.factory import (
    import_data_loader,
    import_data_sampler,
    import_phy_model,
    import_trainer,
    load_loss_func,
    load_nn_model,
)
from dMG.core.utils.path import PathBuilder
from dMG.models import criterion, neural_networks, phy_models
from dMG.models.differentiable_model import DeltaModel
from dMG.models.model_handler import ModelHandler
from dMG.trainers.base import BaseTrainer
from dMG.trainers.ms_trainer import MsTrainer
from dMG.trainers.trainer import Trainer

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
    'load_nn_model',
    'criterion',
    'neural_networks',
    'phy_models',
    'DeltaModel',
    'ModelHandler',
    'BaseTrainer',
    'Trainer',
    'MsTrainer',
]
