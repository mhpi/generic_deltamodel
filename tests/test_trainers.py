"""Test trainers in dmg/trainers/."""

from pathlib import Path

import numpy as np
import torch

from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer
from dmg.trainers.trainer import Trainer
from tests import get_available_classes

# Path to module directory
PATH = Path(__file__).parent.parent / 'src' / 'dmg' / 'trainers'
PKG_PATH = 'dmg.trainers'


trainers = get_available_classes(PATH, PKG_PATH, BaseTrainer)


# ---------------------------------------------------------------------------- #
#  Tests
# ---------------------------------------------------------------------------- #


def test_init(config):
    """Test initialization of trainer classes."""
    for trainer_class in trainers:
        trainer = trainer_class(config)
        # assert isinstance(trainer, BaseTrainer)
        assert trainer.config == config


def test_loss_uses_double_precision_target(config):
    """The loss takes its basin stdev from the unrounded target when available."""
    config['train']['loss_function'] = {'name': 'NseBatchLoss'}
    target = torch.rand(20, 4, 1, dtype=torch.float64) * 1000.0
    dataset = {'target': target.float(), 'target_f64': target}

    model = ModelHandler(config, verbose=False)
    trainer = Trainer(config, model, train_dataset=dataset)

    expected = np.nanstd(target[:, :, 0].numpy(), axis=0)
    assert np.array_equal(trainer.loss_func.std, expected)
