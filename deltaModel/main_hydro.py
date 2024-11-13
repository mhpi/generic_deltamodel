"""Main script to run model experiments (train/test) and manage configurations."""
import logging
import time

import hydra
import torch
from conf.config import ModeEnum
from core.utils import initialize_config, print_config, set_randomseed
from core.data.dataset_loading import get_dataset_dict

from models.model_handler import ModelHandler as dModel
from omegaconf import DictConfig
from trainers import run_experiment, run_train_test
from deltaModel.trainers.trainer import Trainer

log = logging.getLogger(__name__)



@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config",
)
def main(config: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        # Initialize the config and randomseed.
        config = initialize_config(config)
        set_randomseed(config['random_seed'])

        log.info(f"RUNNING MODE: {config['mode']}")
        print_config(config)

        # Initializing a dPL model and trainer objects.
        model = dModel(config) #.to(config['device'])

        log.info("Processing datasets...")
        train_dataset = get_dataset_dict(config, train=True)
        eval_dataset = get_dataset_dict(config, train=False)

        trainer = Trainer(config, model)


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


if __name__ == "__main__":
    main()
    print("Experiment ended.")
