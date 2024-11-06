import logging
from typing import Any, Dict

from conf.config import ModeEnum
from torch.nn import Module
from trainers.test import TestModel
from trainers.train import TrainModel

log = logging.getLogger(__name__)



def build_handler(config: Dict[str, Any], model) -> Module:
    """Build a trainer based on the mode specified in the config.
    
    Parameters
    ----------
    config : dict
        Model and experiment configuration settings.
    
    Returns
    -------
    Module
        Trainer object for the experiment.
    """
    if config['mode'] == ModeEnum.train:
        return TrainModel(config, model)
    elif config['mode'] == ModeEnum.test:
        return TestModel(config, model)
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")


def run_train_test(config: Dict[str, Any], model: Module) -> None:
    """Run a training and testing experiment.

    Parameters
    ----------
    config : dict
        Model and experiment configuration settings.
    """
    # Training
    config['mode'] = ModeEnum.train
    train_experiment_handler = build_handler(config, model)
    train_experiment_handler.run()

    # Testing
    config['mode'] = ModeEnum.test
    test_experiment_handler = build_handler(config, model)            
    test_experiment_handler.model = train_experiment_handler.model
    test_experiment_handler.run()


def run_experiment(config: Dict[str, Any], model: Module) -> None:
    """Run a single experiment.
    
    Parameters
    ----------
    config : dict
        Model and experiment configuration settings.
    """
    experiment_handler = build_handler(config, model)
    experiment_handler.run()
