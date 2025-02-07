"""Main script for running differentiable model experiments."""
import logging
import time

import hydra
import torch
from core.data.data_loaders.loader_hydro_ms import get_dataset_dict
from core.utils import initialize_config, print_config, set_randomseed
from core.utils.module_loaders import get_data_loader, get_trainer
from models.model_handler import ModelHandler as dModel
from omegaconf import DictConfig

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    version_base='1.3',
    config_path='conf/',
    config_name='config_multiscale',
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
        train_dataset = get_dataset_dict(config, train=True)
        eval_dataset = get_dataset_dict(config, train=False)

        # Trainer
        from trainers.trainer_ms import Trainer
        trainer = Trainer(config, model, train_dataset, eval_dataset, verbose=True)

        mode = config['mode']
        if mode == 'train':
            trainer.train()
        elif mode == 'test':
            trainer.evaluate()
        elif mode == 'train_test':
            trainer.train()
            trainer.evaluate()
        elif mode == 'predict':
            trainer.predict()
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
