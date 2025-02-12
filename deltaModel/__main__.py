"""Main script for running differentiable model experiments."""
import logging
import time

import hydra
import torch
from omegaconf import DictConfig

from core.utils import initialize_config, print_config, set_randomseed
from core.utils.module_loaders import get_data_loader, get_trainer
from core.data.data_loaders.loader_hydro_ms import HydroMSDataLoader
from models.model_handler import ModelHandler as dModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    version_base='1.3',
    config_path='conf/',
    config_name='config_ms_dev',
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
        model = dModel(config, verbose=True)

        ### Process datasets ###
        log.info("Processing datasets...")
        data_loader = get_data_loader(config['data_loader'])
        data_loader = data_loader(config, test_split=False, overwrite=False)

        ### Create trainer object ###
        trainer = get_trainer(config['trainer'])
        trainer = trainer(
            config,
            model,
            train_dataset = data_loader.train_dataset,
            eval_dataset = data_loader.eval_dataset,
            inf_dataset = data_loader.dataset,
            verbose=True,
        )

        mode = config['mode']
        if mode == 'train':
            trainer.train()
        elif mode == 'test':
            trainer.evaluate()
        elif mode == 'train_test':
            trainer.train()
            trainer.evaluate()
        elif mode == 'predict':
            trainer.inference()
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
