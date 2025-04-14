from dMG._version import __version__
from dMG.core import post
from dMG.core.calc.metrics import Metrics
from dMG.core.data import loaders, samplers
from dMG.core.utils.factory import (import_data_loader, import_data_sampler,
                                    import_phy_model, import_trainer,
                                    load_loss_func, load_nn_model)
from dMG.core.utils.path import PathBuilder
from dMG.models import criterion, neural_networks, phy_models
from dMG.models.differentiable_model import DeltaModel
from dMG.models.model_handler import ModelHandler
from dMG.trainers.base import BaseTrainer
from dMG.trainers.ms_trainer import MsTrainer
from dMG.trainers.trainer import Trainer

from dMG.core.data.loaders.hydro_loader import HydroLoader
from dMG.core.data.samplers.hydro_sampler import HydroSampler
from dMG.core.utils import print_config, save_model
from dMG.core.data import create_training_grid, load_json,txt_to_array
from dMG.core.post import print_metrics
from dMG.core.post.plot_cdf import plot_cdf
from dMG.core.post.plot_geo import geoplot_single_metric
from dMG.core.post.plot_hydrograph import plot_hydrograph
from dMG.core.utils.dates import Dates


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
    'HydroLoader',
    'HydroSampler',
    'print_config',
    'create_training_grid',
    'load_json',
    'txt_to_array',
    'print_metrics',
    'plot_cdf',
    'geoplot_single_metric',
    'plot_hydrograph',
    'Dates',
    'save_model',
]
