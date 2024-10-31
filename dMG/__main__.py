""" Main script to run model experiments (train/test) and manage configurations. """
import logging
import time
from typing import Any, Dict, Union, Tuple

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from conf.config import Config, ModeEnum
from core.utils import (create_output_dirs, randomseed_config, set_system_spec,
                        show_args)
from trainers import build_handler

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
        config = initialize_config(cfg)

        # Set device, dtype, output directories, and random seed.
        randomseed_config(config['random_seed'])

        config['device'], config['dtype'] = set_system_spec(config['gpu_id'])
        config = create_output_dirs(config)

        exp_name = config['mode']
        log.info(f"RUNNING MODE: {exp_name}")
        show_args(config)

        # Execute experiment based on mode.
        if config['mode'] == ModeEnum.train_test:
            run_train_test(config)
        else:
            run_experiment(config)

        # Clean up and log elapsed time.
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {exp_name} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
        torch.cuda.empty_cache()


def initialize_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Convert config into a dictionary and a Config object for validation.
    """
    try:
        config = OmegaConf.to_container(cfg, resolve=True)
    except ValidationError as e:
        log.exception("Configuration validation error", exc_info=e)
        raise e
    return config


def run_train_test(config_dict: Dict[str, Any]) -> None:
    """
    Run training and testing as one experiment.
    """
    # Training
    config_dict['mode'] = ModeEnum.train
    train_experiment_handler = build_handler(config_dict)
    train_experiment_handler.run()

    # Testing
    config_dict['mode'] = ModeEnum.test
    test_experiment_handler = build_handler(config_dict)            
    test_experiment_handler.dplh_model_handler = train_experiment_handler.dplh_model_handler
    test_experiment_handler.run()


def run_experiment(config_dict: Dict[str, Any]) -> None:
    """ Run an experiment based on the mode specified in the configuration. """
    experiment_handler = build_handler(config_dict)
    experiment_handler.run()



if __name__ == "__main__":
    main()
    print("Experiment ended.")
