"""Test trainers in dmg/trainers/."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
import torch

from dmg.trainers.base import BaseTrainer
from tests import get_available_classes

# Path to module directory
PATH = Path(__file__).parent.parent / 'src' / 'dmg' / 'trainers'
PKG_PATH = 'dmg.trainers'


loaders = get_available_classes(PATH, PKG_PATH, BaseTrainer)

@pytest.fixture
def config():
    """Fixture for a mock configuration dictionary."""
    return {
        'mode': 'train_test',
        'multimodel_type': 'None',
        'device': 'cpu',
        'data_loader': 'HydroLoader',
        'data_sampler': 'HydroLoader',
        'train': {
            'target': ['streamflow'],
            'epochs': 2,
            'start_epoch': 0,
            'save_epoch': 1,
        },
        'delta_model': {
            'phy_model': {
                'model': ['HBV'],
                'warm_up': 0,
                'dynamic_params': {
                    'HBV': ['parBETA', 'parBETAET'],
                },
                'forcings': ['precip', 'tmean', 'pet'],
            },
            'nn_model': {
                'model': 'LstmModel',
                'dropout': 0.5,
                'hidden_size': 32,
                'learning_rate': 0.001,
                'lr_scheduler': 'StepLR',
                'lr_scheduler_params': {'step_size': 1, 'gamma': 0.1},
                'forcings': ['precip', 'tmean', 'pet'],
                'attributes': ['p_mean'],
            },
        },
        'loss_function': 'MSELoss',
        'model_path': './tests/test_output/',
        'out_path': './tests/test_output/',
    }

@pytest.fixture
def mock_datasets():
    """Fixture for mock training, evaluation, and inference datasets."""
    train_dataset = {
        'xc_nn_norm': torch.rand(10, 5, 3),  # (time, samples, features)
        'target': torch.rand(10, 5, 1),     # (time, samples, targets)
    }
    eval_dataset = {
        'xc_nn_norm': torch.rand(10, 5, 3),
        'target': torch.rand(10, 5, 1),
    }
    dataset = {
        'xc_nn_norm': torch.rand(10, 5, 3),
    }
    return train_dataset, eval_dataset, dataset


def test_init(config):
    """Test initialization of trainer classes."""
    for trainer_class in loaders:
        trainer = trainer_class(config)
        # assert isinstance(trainer, BaseTrainer)
        assert trainer.config == config
