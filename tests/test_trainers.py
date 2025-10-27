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
def mock_datasets():
    """Fixture for mock training, evaluation, and inference datasets."""
    train_dataset = {
        'xc_nn_norm': torch.rand(10, 5, 3),  # (time, samples, features)
        'target': torch.rand(10, 5, 1),  # (time, samples, targets)
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
