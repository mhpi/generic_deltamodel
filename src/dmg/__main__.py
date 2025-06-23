"""There are two ways to run dMG in command line...

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


def run_mode(mode, trainer) -> None:
    """Execute the appropriate mode for training, testing, or inference."""
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


@hydra.main(
    version_base='1.3',
    config_path='./../../conf/',
    config_name='prism',
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
            
        log.info(f"Running mode: {config['mode']}")
        print_config(config)

        ### Create/Load differentiable model ###
        model = dModel(config, verbose=True)

        ### Process datasets ###
        log.info("Processing datasets...")
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
        run_mode(config['mode'], trainer)

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
