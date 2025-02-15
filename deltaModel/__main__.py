"""Main script for running differentiable model experiments."""
import logging
import time

import hydra
import torch
from omegaconf import DictConfig

from core.utils import initialize_config, print_config, set_randomseed
from core.utils.factory import import_data_loader, import_trainer
from models.model_handler import ModelHandler as dModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def run_mode(mode: str, trainer):
    """Execute the appropriate mode for training, testing, or inference."""
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
    

@hydra.main(
    version_base='1.3',
    config_path='conf/',
    config_name='config',
)
def main(config: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        ### Initializations ###
        config = initialize_config(config)
        set_randomseed(config['random_seed'])

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
            inf_dataset=data_loader.dataset,
            verbose=True,
        )

        ### Run mode ###
        run_mode(config['mode'], trainer)
        
    except KeyboardInterrupt:
        log.warning("|> Keyboard interrupt received. Exiting gracefully <|")

    except Exception as e:
        log.error("|> An error occurred <|", exc_info=True)  # Logs full traceback
    
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
