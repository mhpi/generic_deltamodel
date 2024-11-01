import logging
from typing import Dict, Any

from conf.config import ModeEnum
from trainers.test import TestModel
from trainers.train import TrainModel
from torch.nn import Module

log = logging.getLogger(__name__)



def build_handler(config: Dict[str, Any]) -> Module:
    """Build a handler based on the mode specified in the configuration.
    
    Parameters
    ----------
    config : dict
        Dictionary of configuration settings.
    
    Returns
    -------
    Module
        Trainer object for the experiment.
    """
    if config['mode'] == ModeEnum.train:
        return TrainModel(config)
    elif config['mode'] == ModeEnum.test:
        return TestModel(config)
    else:
        raise ValueError(f"Unsupported mode: {config['mode']}")


def run_train_test(config: Dict[str, Any]) -> None:
    """Run a training and testing experiment.

    Parameters
    ----------
    config : dict
        Dictionary of configuration settings.
    """
    # Training
    config['mode'] = ModeEnum.train
    train_experiment_handler = build_handler(config)
    train_experiment_handler.run()

    # Testing
    config['mode'] = ModeEnum.test
    test_experiment_handler = build_handler(config)            
    test_experiment_handler.dplh_model_handler = train_experiment_handler.dplh_model_handler
    test_experiment_handler.run()


def run_experiment(config: Dict[str, Any]) -> None:
    """Run a single experiment.
    
    Parameters
    ----------
    config_dict : Dict[str, Any]
        Dictionary of configuration settings.
    """
    experiment_handler = build_handler(config)
    experiment_handler.run()
