import logging
import os
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
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.
    
    Parameters
    ----------
    config : dict
        Configuration settings for the model and experiment.
    model : torch.nn.Module, optional
        Learnable model object. If not provided, a new model is initialized.
    train_dataset : dict, optional
        Training dataset dictionary.
    eval_dataset : dict, optional
        Testing/inference dataset dictionary.
    inf_dataset : dict, optional
        Inference dataset dictionary.
    loss_func : torch.nn.Module, optional
        Loss function object. If not provided, a new loss function is initialized.
    optimizer : torch.optim.Optimizer, optional
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler : torch.nn.Module, optional
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        inf_dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inf_dataset = inf_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = load_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False

        if 'train' in config['mode']:
            log.info(f"Initializing training mode")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or get_loss_func(
                self.train_dataset['target'],
                config['loss_function'],
                config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config['dpl_model']['nn_model']['lr_scheduler']:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            self.start_epoch = self.config['train']['start_epoch'] + 1
            if self.start_epoch > 1:
                log.info(f"Loading trainer states to begin epoch {self.start_epoch}") 
                self.load_states()

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a model state optimizer.
        
        Adding additional optimizers is possible by extending the optimizer_dict.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        name = self.config['train']['optimizer']
        learning_rate = self.config['dpl_model']['nn_model']['learning_rate']
        optimizer_dict = {
            # 'SGD': torch.optim.SGD,
            # 'Adam': torch.optim.Adam,
            # 'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            # 'RMSprop': torch.optim.RMSprop,
        }

        # Fetch optimizer class
        cls = optimizer_dict[name]
        if cls is None:
            raise ValueError(f"Optimizer '{name}' not recognized. "
                                f"Available options are: {list(optimizer_dict.keys())}")

        # Initialize
        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=learning_rate,
            )
        except Exception as e:
            raise ValueError(f"Error initializing optimizer: {e}")
        return self.optimizer
    
    def init_scheduler(self) -> None:
        """Initialize a learning rate scheduler for the optimizer."""
        name = self.config['dpl_model']['nn_model']['lr_scheduler']
        scheduler_dict = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            # 'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        # Fetch scheduler class
        cls = scheduler_dict[name]
        if cls is None:
            raise ValueError(f"Scheduler '{name}' not recognized. "
                                f"Available options are: {list(scheduler_dict.keys())}")
        
        # Initialize
        try:
            self.scheduler = cls(
                self.optimizer,
                **self.config['dpl_model']['nn_model']['lr_scheduler_params'],
            )
        except Exception as e:
            raise ValueError(f"Error initializing scheduler: {e}")
        return self.scheduler

    def load_states(self) -> None:
        """Load model, optimizer, and scheduler states from a checkpoint."""
        path = self.config['model_path']
        for file in os.listdir(path):
            if 'train_state' and str(self.start_epoch-1) in file:
                checkpoint = torch.load(os.path.join(path, file))
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                continue
        
        raise FileNotFoundError(f"No checkpoint for epoch {self.start_epoch-1}.")

        # Restore random states
        torch.set_rng_state(checkpoint['random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True

        # Setup a training grid (number of samples, minibatches, and timesteps)
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'],
            self.config
        )

        # Training loop
        log.info(f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs")
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.perf_counter()
            prog_str = f"Epoch {epoch}/{self.epochs}"

            self.current_epoch = epoch
            self.total_loss = 0.0

            # Iterate through minibatches.
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

            if self.verbose:
                log.info(f"\n ---- \n Epoch {epoch} total loss: {self.total_loss}")
            self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

            # Save model and trainer states.
            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)
                save_train_state(
                    self.config,
                    epoch=epoch,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    clear_prior=True,
                )
        log.info(f"Training complete.")

    def evaluate(self) -> None:
        """Run model evaluation and return metrics.
        
        Model outputs and results are also saved.
        """
        self.is_in_train = False

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.eval_dataset['target']

        # Get start and end indices for each batch.
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Testing loop
        log.info(f"Testing model: Forwarding on {len(batch_start)} batches")
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
                log.info(f"Batch {i + 1}/{len(batch_start)} evaluated.")

        # Save predictions and calculate metrics
        log.info("Saving model results and calculating metrics")
        save_outputs(self.config, batch_predictions, observations)
        self.calc_metrics(batch_predictions, observations)

    def predict(self) -> None:
        """Run model inference and return predictions."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch.
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Inference loop
        log.info(f"Running Inference: Forwarding on {len(batch_start)} batches")
        for i in tqdm.tqdm(range(len(batch_start)), desc="Inference", leave=False, dynamic_ncols=True):
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
                log.info(f"Batch {i + 1}/{len(batch_start)} processed in inference loop.")

        # Save predictions
        log.info("Saving model predictions")
        save_outputs(self.config, batch_predictions)
    
    def evaluation_loop(self) -> None:
        """Inference loop used in .evaluate() and .predict() methods."""
        return NotImplementedError
            
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
        if self.config['dpl_model']['phy_model']['warm_up_states']:
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
