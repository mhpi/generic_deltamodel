import logging
import os
import time
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dMG.core.calc.metrics import Metrics
from dMG.core.data import create_training_grid
from dMG.core.utils import save_outputs, save_train_state
from dMG.core.utils.factory import import_data_sampler, load_loss_func
from dMG.models.model_handler import ModelHandler
from dMG.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.
    
    Parameters
    ----------
    config
        Configuration settings for the model and experiment.
    model
        Learnable model object. If not provided, a new model is initialized.
    train_dataset
        Training dataset dictionary.
    eval_dataset
        Testing/inference dataset dictionary.
    dataset
        Inference dataset dictionary.
    loss_func
        Loss function object. If not provided, a new loss function is initialized.
    optimizer
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    """
    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False

        if 'train' in config['mode']:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")
            
            log.info("Initializing experiment")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or load_loss_func(
                self.train_dataset['target'],
                config['loss_function'],
                device=config['device'],
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
                self.load_states()

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.
        
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
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer
    
    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler for the optimizer.
        
        torch.optim.lr_scheduler.LRScheduler
            Initialized learning rate scheduler object.
        """
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
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint to resume
        training if a checkpoint file exists.
        """
        path = self.config['model_path']
        for file in os.listdir(path):
            # Check for state checkpoint: looks like `train_state_epoch_XX.pt`.
            if 'train_state' and (str(self.start_epoch-1) in file):
                log.info("Loading trainer states --> Resuming Training from" /
                         f" epoch {self.start_epoch}")

                checkpoint = torch.load(os.path.join(path, file))

                # Restore optimizer states
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # Restore random states
                torch.set_rng_state(checkpoint['random_state'])
                if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
                    torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])
                return
            elif 'train_state' in file:
                raise FileNotFoundError(f"Available checkpoint file {file} does" /
                                        f" not match start epoch {self.start_epoch-1}.")

        # If no checkpoint file is found for named epoch...
        raise FileNotFoundError(f"No checkpoint for epoch {self.start_epoch-1}.")

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

            # Iterate through epoch in minibatches.
            for i in tqdm.tqdm(range(1, n_minibatch + 1), desc=prog_str,
                               leave=False, dynamic_ncols=True):
                self.current_batch = i

                dataset_sample = self.sampler.get_training_sample(
                    self.train_dataset,
                    n_samples,
                    n_timesteps,
                )
            
                # Forward pass through model.
                _ = self.model(dataset_sample)
                loss = self.model.calc_loss(dataset_sample)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.total_loss += loss.item()

                if self.verbose:
                    tqdm.tqdm.write(f"Epoch {epoch}, batch {i} | loss: {loss.item()}")

            if self.use_scheduler:
                self.scheduler.step()

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
        log.info("Training complete")

    def evaluate(self) -> None:
        """Run model evaluation and return both metrics and model outputs."""
        self.is_in_train = False

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.eval_dataset['target']

        # Get start and end indices for each batch
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.eval_dataset, batch_start, batch_end)

        # Save predictions and calculate metrics
        log.info("Saving model outputs + Calculating metrics")
        save_outputs(self.config, batch_predictions, observations)
        self.predictions = self._batch_data(batch_predictions)

        # Calculate metrics
        self.calc_metrics(batch_predictions, observations)

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['predict']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions
    
    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.
        
        Parameters
        ----------
        batch_list
            List of dictionaries from each forward batch containing inputs and
            model predictions.
        target_key
            Key to extract from each batch dictionary.
        """
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data
        
        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray
    ) -> None:
        """Forward loop used in model evaluation and inference.

        Parameters
        ----------
        data
            Dictionary containing model input data.
        batch_start
            Start indices for each batch.
        batch_end
            End indices for each batch.
        """
        # Track predictions accross batches
        batch_predictions = []

        for i in tqdm.tqdm(range(len(batch_start)), desc='Forwarding', leave=False, dynamic_ncols=True):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch predictions
            model_name = self.config['dpl_model']['phy_model']['model'][0]
            prediction = {
                key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()
            }
            batch_predictions.append(prediction)
        return batch_predictions

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        target_name = self.config['train']['target'][0]
        predictions = self._batch_data(batch_predictions, target_name)
        target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)

        # Remove warm-up data
        # if self.config['dpl_model']['phy_model']['warm_up_states']:  # NOTE: remove if bug does not reoccur
        target = target[self.config['dpl_model']['phy_model']['warm_up']:, :]

        # Compute metrics
        metrics = Metrics(
            np.swapaxes(predictions.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config['out_path'])

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log statistics after each epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        loss_dict
            Dictionary containing loss values.
        n_minibatch
            Number of minibatches.
        start_time
            Start time of the epoch.
        """
        avg_loss_dict = {key: value / n_minibatch + 1 for key, value in loss_dict.items()}
        loss = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)

        log.info(
            f"Loss after epoch {epoch}: {loss} \n"
            f"~ Runtime {elapsed:.2f} s, {mem_aloc} Mb reserved GPU memory"
        )
