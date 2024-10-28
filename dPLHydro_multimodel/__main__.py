"""
Script to interface with model experiments (train/test) and manage configurations.
"""

import logging
import time
from typing import Any, Dict, Union, Tuple

import numpy as np
import random
import torch
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from conf.config import Config, ModeEnum
from core.utils import (create_output_dirs, randomseed_config, set_system_spec,
                        show_args)
from experiment import build_handler
from experiment.experiment_tracker import ExperimentTracker

log = logging.getLogger(__name__)



@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        # Injest config yaml
        # NOTE: Temporarily using config_dict for better readability, may rm config.
        config, config_dict = initialize_config(cfg)
        experiment_tracker = ExperimentTracker(cfg=config)

        # Set device, dtype, output directories, and random seed.
        randomseed_config(config.random_seed)

        config.device, config.dtype = set_system_spec(config.gpu_id)
        config_dict = create_output_dirs(config_dict)

        experiment_name = config.mode
        log.info(f"RUNNING MODE: {config.mode}")
        show_args(config)

        # Execute experiment based on mode.
        if config.mode == ModeEnum.train_test:
           run_train_test(config, config_dict, experiment_tracker)
        else:
            run_experiment(config, config_dict, experiment_tracker)

        # Clean up and log elapsed time.
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {experiment_name} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
        torch.cuda.empty_cache()


def initialize_config(cfg: DictConfig) -> Tuple[Config, Dict[str, Any]]:
    """
    Convert config into a dictionary and a Config object for validation.
    """
    try:
        config_dict: Union[Dict[str, Any], Any] = OmegaConf.to_container(
            cfg, resolve=True)
        config = Config(**config_dict)
    except ValidationError as e:
        log.exception("Configuration validation error", exc_info=e)
        raise e
    return config, config_dict


def run_train_test(config: Config, config_dict: Dict[str, Any], experiment_tracker: ExperimentTracker) -> None:
    """
    Run training and testing as one experiment.
    """
    # Train phase
    config.mode = ModeEnum.train
    train_experiment_handler = build_handler(config, config_dict)
    train_experiment_handler.run(experiment_tracker=experiment_tracker)

    # Test phase
    config.mode = ModeEnum.test
    test_experiment_handler = build_handler(config, config_dict)            
    test_experiment_handler.dplh_model_handler = train_experiment_handler.dplh_model_handler
    if config_dict['ensemble_type'] != 'none':
        test_experiment_handler.ensemble_lstm = train_experiment_handler.ensemble_lstm
    test_experiment_handler.run(experiment_tracker=experiment_tracker)


def run_experiment(config: Config, config_dict: Dict[str, Any], experiment_tracker: ExperimentTracker) -> None:
    """
    Run an experiment based on the mode specified in the configuration.
    """
    experiment_handler = build_handler(config, config_dict)
    experiment_handler.run(experiment_tracker=experiment_tracker)



if __name__ == "__main__":
    main()
    print("Experiment ended.")
