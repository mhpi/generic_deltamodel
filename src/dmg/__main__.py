"""
Main entry point for differentiable model experiments.

To run dMG from the command line, use
-> `python -m dmg --config-name <config_name>`
Specify a config file in the `conf/` directory, or exclude the `--config-name`
flag to use the default config.
"""

import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig

from dmg._version import __version__
from dmg.core.utils.factory import import_data_loader, import_trainer
from dmg.core.utils.paths import check_experiment_exists
from dmg.core.utils.utils import initialize_config, print_config, set_randomseed
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
    elif mode == 'sim':
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
        run_spatial_mode(config, model)
    elif test_type == 'temporal':
        if trainer is None:
            raise ValueError("Trainer required for temporal mode")
        run_temporal_mode(config['mode'], trainer)


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
        check_experiment_exists(config.get('exp_name'))
        config = initialize_config(config)
        set_randomseed(config['seed'])

        ### Do model tuning ###
        if config['do_tune']:
            try:
                from dmg.core.tune.utils import run_tuning

                run_tuning(config)
                exit()
            except ImportError:
                log.error(
                    "Ray Tune is required for tuning. To install: uv pip install 'dmg[raytune]'",
                )
                return

        print_config(config)

        ### Create/Load differentiable model ###
        model = dModel(config, verbose=True)

        ### Process datasets and create trainer for temporal mode ###
        trainer = None
        if config['test']['name'] == 'temporal':
            log.info("Processing data...")
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
                verbose=config.get('verbose', False),
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
            f"Time Elapsed: {(total_time / 60):.3f} minutes",
        )


if __name__ == '__main__':
    os.environ['DMG_VERSION'] = __version__
    main()
