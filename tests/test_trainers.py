"""Test trainers in dmg/trainers/."""

from pathlib import Path

from dmg.trainers.base import BaseTrainer
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
