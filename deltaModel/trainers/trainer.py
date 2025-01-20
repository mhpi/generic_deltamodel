import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm
from core.calc.metrics import Metrics
from core.data import create_training_grid
from core.utils import save_outputs, save_train_state
from core.utils.module_loaders import load_data_sampler
from models.loss_functions import get_loss_func
from models.model_handler import ModelHandler
from trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """Generic, unified Trainer for differentiable models.

    Designed after the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizer and loss function,
    and runs training and testing loops as specified.
    
    Parameters
    ----------
    config : dict
        Configuration settings for the model and experiment.
    model : Module, optional
        dPL (differentiable parameter learning) model object.
    verbose : bool, optional
        Whether to print verbose output. The default is False.

    TODO
    - cleanup docstrings and comments
    """
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.nn.Module] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = load_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False

        if 'train' in config['mode']:
            log.info(f"Initializing training mode")

            # Loss function
            self.loss_func = loss_func or get_loss_func(
                self.train_dataset['target'],
                config['loss_function'],
                config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()

            if config['dpl_model']['nn_model']['use_scheduler']:
                self.use_scheduler = True
                self.scheduler = self.init_scheduler()
            else:
                self.use_scheduler = False

            ## TODO: load saved epoch from file here
            # Resume model training from a saved epoch
            self.start_epoch = self.config['train']['start_epoch'] + 1

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer as named in config.
        
        Adding additional optimizers is possible by extending the optimizer_dict.

        TODO: Add (dynamic) support for additional optimizer parameters.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        optimizer_name = self.config['train']['optimizer']
        learning_rate = self.config['dpl_model']['nn_model']['learning_rate']

        # Dictionary mapping optimizer names to their corresponding classes
        optimizer_dict = {
            # 'SGD': torch.optim.SGD,
            # 'Adam': torch.optim.Adam,
            # 'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            # 'RMSprop': torch.optim.RMSprop,
        }

        # Fetch optimizer class
        optimizer_cls = optimizer_dict[optimizer_name]

        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized. "
                                f"Available options are: {list(optimizer_dict.keys())}")

        # Initialize
        try:
            self.optimizer = optimizer_cls(
                self.model.get_parameters(),
                lr=learning_rate,
            )
        except Exception as e:
            raise ValueError(f"Error initializing optimizer: {e}")

        return self.optimizer
    
    def init_scheduler(self) -> None:
        ## TODO: Implement learning rate scheduler
        """
        requires the optimizer

        e.g., optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        """
        return NotImplementedError
                    
    def train(self) -> None:
        """Entry point for training loop."""
        self.is_in_train = True
        self.epochs = self.config['train']['epochs']

        # Setup a training grid (number of samples, minibatches, and timesteps)
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'],
            self.config
        )

        # Training loop
        log.info(f"Training model: Beginning {self.start_epoch} of {self.config['train']['epochs']} epochs")
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.perf_counter()
            prog_str = f"Epoch {epoch}/{self.config['train']['epochs']}"

            self.current_epoch = epoch
            self.total_loss = 0.0

            # Iterate through minibatches
            for i in tqdm.tqdm(range(1, n_minibatch + 1), desc=prog_str,
                               leave=False, dynamic_ncols=True):
                self.current_batch = i

                dataset_sample = self.sampler.get_training_sample(
                    self.train_dataset,
                    n_samples,
                    n_timesteps,
                )
            
                # Forward pass through model.
                prediction = self.model(dataset_sample)
                loss = self.model.calc_loss(dataset_sample)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.total_loss += loss.item()

                if self.verbose:
                    log.info(f"Epoch {epoch} minibatch {i} loss: {loss.item()}")

            if self.use_scheduler: self.scheduler.step()

            self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

            # Save model and trainer states.
            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)
                save_train_state(
                    self.config,
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
        log.info(f"Training complete.")

    def test(self) -> None:
        """Run testing loop and save results."""
        self.is_in_test = True

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.eval_dataset['target']

        # Get start and end indices for each batch.
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Testing loop
        log.info(f"Testing Model: Forwarding on {len(batch_start)} batches")
        for i in tqdm.tqdm(range(len(batch_start)), desc="Testing", leave=False, dynamic_ncols=True):
            self.current_batch = i

            
            dataset_sample = self.sampler.get_validation_sample(
                self.eval_dataset,
                batch_start[i],
                batch_end[i],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch predictions
            model_name = self.config['dpl_model']['phy_model']['model'][0]
            prediction = {key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()}
            batch_predictions.append(prediction)

            if self.verbose:
                log.info(f"Batch {i + 1}/{len(batch_start)} processed in testing loop.")

        # Save predictions and calculate metrics
        log.info("Saving model results and calculating metrics")
        save_outputs(self.config, batch_predictions, observations)
        self.calc_metrics(batch_predictions, observations)

    def calc_metrics(
        self,
        batch_predictions: List[Dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics.
        
        Parameters
        ----------
        batch_predictions : list
            List of dictionaries containing model predictions.
        observations : torch.Tensor
            Target variable observation data.
        """
        target_name = self.config['train']['target'][0]

        pred = torch.cat([x[target_name] for x in batch_predictions], dim=1).numpy()
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)

        # Remove warm-up data
        if not self.config['dpl_model']['phy_model']['warm_up_states']:
            target = target[self.config['dpl_model']['phy_model']['warm_up']:, :]

        # Compute metrics
        metrics = Metrics(
            np.swapaxes(pred.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config['out_path'])

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: Dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log statistics after each epoch."""
        avg_loss_dict = {key: value / n_minibatch + 1 for key, value in loss_dict.items()}
        loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info(
            f"Model loss after epoch {epoch}: {loss_formated} \n"
            f"~ Runtime {elapsed:.2f} sec, {mem_aloc} Mb reserved GPU memory"
        )
