""" Main script to run model experiments (train/test) and manage configurations. """
import logging
import time
from typing import Any, Dict

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from conf.config import ModeEnum
from core.utils import (create_output_dirs, set_randomseed, set_system_spec,
                        print_config)
from trainers import run_train_test, run_experiment

log = logging.getLogger(__name__)



@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(config: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        # Convert yaml to dict and initialize configuration settings.
        config = initialize_config(config)

        set_randomseed(config['random_seed'])

        log.info(f"RUNNING MODE: {config['mode']}")
        print_config(config)

        # Run Trainer based on mode.
        if config['mode'] == ModeEnum.train_test:
            run_train_test(config)
        else:
            run_experiment(config)

        # Clean up and log elapsed time.
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Cleaning up...")
        torch.cuda.empty_cache()


def initialize_config(config: DictConfig) -> Dict[str, Any]:
    """Parse and initialize configuration settings.
    
    Parameters
    ----------
    config : DictConfig
        Configuration settings from Hydra.
        
    Returns
    -------
    dict
        Dictionary of configuration settings.
    """
    try:
        config = OmegaConf.to_container(config, resolve=True)
    except ValidationError as e:
        log.exception("Configuration validation error", exc_info=e)
        raise e
    
    config['device'], config['dtype'] = set_system_spec(config['gpu_id'])
    config = create_output_dirs(config)

    return config


if __name__ == "__main__":
    main()
    print("Experiment ended.")
