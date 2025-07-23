"""
Main entry point for differentiable model experiments with spatial testing support.

There are two ways to run dmg in command line:
1. python -m dmg (Uses default config.yaml)
2. python src/dmg/__main__.py (Uses default config.yaml)
Add opt `--config-name <config_name>` to (1) or (2) to use a different config.
"""
import logging
import time
import hydra
import torch
from omegaconf import DictConfig
from dmg.core.utils.factory import import_data_loader, import_trainer
from dmg.core.utils.utils import (initialize_config, print_config, set_randomseed)
from dmg.models.model_handler import ModelHandler as dModel
from dmg.core.utils.spatial_testing import run_spatial_testing

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _is_fire_data(config: DictConfig) -> bool:
    """Check if this is fire occurrence data."""
    # Check observations name
    obs_name = config.get('observations', {}).get('name', '').lower()
    if 'fire' in obs_name:
        return True
    
    # Check target variables
    targets = config.get('train', {}).get('target', [])
    if any('fire' in str(target).lower() for target in targets):
        return True
    
    # Check data loader type
    data_loader = config.get('data_loader', '')
    if 'grid' in data_loader.lower():
        return True
    
    return False


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
            # Check if this is fire data and use appropriate spatial testing
            if _is_fire_data(config):
                log.info("Detected fire data - using fire-specific spatial testing")
                try:
                    from dmg.core.utils.fire_spatial_testing import run_fire_spatial_testing
                    run_fire_spatial_testing(config, model)
                except ImportError:
                    log.warning("Fire spatial testing not available, falling back to standard spatial testing")
                    run_spatial_testing(config, model)
            else:
                # Standard spatial testing for streamflow/other data
                log.info("Using standard spatial testing")
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