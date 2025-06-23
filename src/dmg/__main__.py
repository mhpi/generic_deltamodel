"""
Main entry point for differentiable model experiments.

There are two ways to run dMG in command line...
1. python -m dmg (Uses default config.yaml)
2. python src/dmg/__main__.py (Uses default config.yaml)
Add flag `--config-name <config_name>` to (1) or (2) to use a different config.
"""
import logging
import time

import hydra
import torch
from omegaconf import DictConfig

from dmg.core.tune.utils import run_tuning
from dmg.core.utils.factory import import_data_loader, import_trainer
from dmg.core.utils.utils import (initialize_config, print_config,
                                  set_randomseed)
from dmg.models.model_handler import ModelHandler as dModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def run_temporal_mode(mode: str, trainer) -> None:
    """Execute the appropriate temporal mode for training, testing, or inference."""
    if mode == 'train':
        trainer.train()
    elif mode == 'test':
        trainer.evaluate()
    elif mode == 'train_test':
        trainer.train()
        trainer.evaluate()
    elif mode == 'simulation':
        trainer.inference()
    else:
        raise ValueError(f"Invalid mode: {mode}")


def run_spatial_mode(config: DictConfig, model) -> None:
    """Execute spatial testing across all holdout indices."""
    from dmg.core.utils.spatial_testing import run_spatial_testing
    run_spatial_testing(config, model)


def run_mode(config: DictConfig, model, trainer=None) -> None:
    """Execute the appropriate mode based on test type (temporal or spatial)."""
    test_type = config.get('test', {}).get('type', 'temporal')
    
    if test_type == 'spatial':
        log.info("Running spatial testing mode")
        run_spatial_mode(config, model)
    elif test_type == 'temporal':
        log.info("Running temporal testing mode")
        if trainer is None:
            raise ValueError("Trainer required for temporal mode")
        run_temporal_mode(config['mode'], trainer)
    else:
        raise ValueError(f"Invalid test type: {test_type}. Must be 'temporal' or 'spatial'")


@hydra.main(
    version_base='1.3',
    config_path='./../../conf/',
    config_name='default',
)
def main(config: DictConfig) -> None:
    """Main function to run differentiable model experiments."""
    try:
        start_time = time.perf_counter()
        
        ### Initializations ###
        config = initialize_config(config, write_path=True)
        set_randomseed(config['random_seed'])
        
        ### Do model tuning ###
        if config.get('do_tune', False):
            run_tuning(config)
            exit()
        
        test_type = config.get('test', {}).get('type', 'temporal')
        log.info(f"Running mode: {config['mode']}, Test type: {test_type}")
        print_config(config)
        
        ### Create/Load differentiable model ###
        model = dModel(config, verbose=True)
        
        ### Process datasets and create trainer for temporal mode ###
        trainer = None
        if test_type == 'temporal':
            log.info("Processing datasets for temporal testing...")
            data_loader_cls = import_data_loader(config['data_loader'])
            data_loader = data_loader_cls(config, test_split=True, overwrite=False)
            
            ### Create trainer object ###
            trainer_cls = import_trainer(config['trainer'])
            trainer = trainer_cls(
                config,
                model,
                train_dataset=data_loader.train_dataset,
                eval_dataset=data_loader.eval_dataset,
                dataset=data_loader.dataset,
                verbose=True,
            )
        
        ### Run mode ###
        run_mode(config, model, trainer)
        
    except KeyboardInterrupt:
        log.warning("|> Keyboard interrupt received. Exiting gracefully <|")
    except Exception:
        log.error("|> An error occurred <|", exc_info=True)  # Logs full traceback
    finally:
        log.info("Cleaning up resources...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes",
        )


if __name__ == '__main__':
    main()
