""" Main script for running differentiable model experiments."""
import logging
import time

import hydra
import torch
from omegaconf import DictConfig

from core.data.data_loaders.hydro_loader import HydroDataLoader
from core.utils import initialize_config, print_config, set_randomseed
from models.model_handler import ModelHandler as dModel
from trainers.trainer import Trainer

log = logging.getLogger(__name__)


@hydra.main(
    version_base='1.3',
    config_path='conf/',
    config_name='config_multimodel',
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
        log.info("Loading dataset...")
        data_loader = HydroDataLoader(config, test_split=True, overwrite=False)

        ### Create Trainer object ###
        trainer = Trainer(
            config,
            model,
            data_loader.train_dataset,
            data_loader.eval_dataset,
            verbose=True
        )

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


if __name__ == '__main__':
    main()
