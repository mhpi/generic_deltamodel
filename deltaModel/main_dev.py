""" Main script for running differentiable model experiments."""
import logging
import time

import hydra
import torch
from core.data.dataset_loading import get_dataset_dict
from core.utils import initialize_config, print_config, set_randomseed
# Dev imports
from models.model_handler_dev import ModelHandler as dModel
from omegaconf import DictConfig
from trainers.trainer_dev import Trainer

log = logging.getLogger(__name__)



@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config_dev",
)
def main(config: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        ### Initializations ###
        config = initialize_config(config)
        set_randomseed(config['random_seed'])

        log.info(f"RUNNING MODE: {config['mode']}")
        print_config(config)

        ### Create/Load differentiable model ###
        model = dModel(config, verbose=True) #.to(config['device'])

        ### Process datasets ###
        log.info("Processing datasets...")
        train_dataset = get_dataset_dict(config, train=True)
        eval_dataset = get_dataset_dict(config, train=False)

        ### Create Trainer object ###
        trainer = Trainer(config, model, train_dataset, eval_dataset, verbose=True)

        mode = config['mode']
        if mode == 'train':
            trainer.train()
        elif mode == 'test':
            trainer.test()
        elif mode == 'train_test':
            trainer.train()
            trainer.test()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    except Exception as e:
        log.error(f"Error: {e}")
        raise e
    
    finally:
        print("Cleaning up...")
        torch.cuda.empty_cache()
    
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed |"
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 


if __name__ == "__main__":
    main()
