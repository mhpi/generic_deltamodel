"""
Main entry point for differentiable model experiments with spatial testing support.

There are two ways to run dMG in command line:
1. python -m dMG (Uses default config.yaml)
2. python src/dMG/__main__.py (Uses default config.yaml)
Add opt `--config-name <config_name>` to (1) or (2) to use a different config.
"""
import logging
import time
import hydra
import torch
from omegaconf import DictConfig
from dMG.core.utils.factory import import_data_loader, import_trainer
from dMG.core.utils.utils import (initialize_config, print_config, set_randomseed)
from dMG.models.model_handler import ModelHandler as dModel
from dMG.core.utils.spatial_testing import run_spatial_testing

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)



@hydra.main(
    version_base='1.3',
    config_path='./../../conf/',
    config_name='temporalCamels',
)
def main(config: DictConfig) -> None:
    """Main function to run differentiable model experiments."""
    try:
        start_time = time.perf_counter()
        
        ### Initializations ###
        config = initialize_config(config, write_path=True)
        set_randomseed(config['random_seed'])
        
        test_mode = config.get('test', {}).get('type', 'temporal')
        log.info(f"Running mode: {config['mode']}, Testing type: {test_mode}")
        print_config(config)
        
        ### Create/Load differentiable model ###
        model = dModel(config, verbose=True)
        
        ### Handle different testing modes ###
        if test_mode == 'temporal':
            # Standard temporal testing
            data_loader_cls = import_data_loader(config['data_loader'])
            data_loader = data_loader_cls(config, test_split=True, overwrite=False)
            
            trainer_cls = import_trainer(config['trainer'])
            trainer = trainer_cls(
                config,
                model,
                train_dataset=data_loader.train_dataset,
                eval_dataset=data_loader.eval_dataset,
                dataset=data_loader.dataset,
                verbose=True,
            )
            mode = config['mode']
            if mode == 'train':
                trainer.train()
            elif mode == 'test':
                return trainer.evaluate()
            elif mode == 'train_test':
                trainer.train()
                return trainer.evaluate()
            elif mode == 'simulation':
                trainer.inference()
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
        elif test_mode == 'spatial':
            # Spatial testing with holdout validation
            run_spatial_testing(config, model)
            
        else:
            raise ValueError(f"Invalid test_mode type: {test_mode}")
            
    except KeyboardInterrupt:
        log.warning("|> Keyboard interrupt received. Exiting gracefully <|")
    except Exception:
        log.error("|> An error occurred <|", exc_info=True)
    finally:
        log.info("Cleaning up resources...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        )


if __name__ == '__main__':
    main()