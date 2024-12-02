import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from deltaModel.core.calc.metrics import metrics
from core.data import (create_training_grid, get_training_sample,
                       get_validation_sample)
from core.data.dataset_loading import get_dataset_dict
from core.utils import save_outputs
from models.loss_functions import get_loss_func
from models.model_handler import ModelHandler
from torch import nn

log = logging.getLogger(__name__)



class Trainer:
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
    dataset : dict, optional
        Dataset dictionary containing forcings and attributes
    verbose : bool, optional
        Whether to print verbose output. The default is False.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            model: nn.Module = None,
            train_dataset: Optional[dict] = None,
            eval_dataset: Optional[dict] = None,
            loss_func: Optional[nn.Module] = None,
            optimizer: Optional[nn.Module] = None,
            verbose: Optional[bool] = False
        ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset or get_dataset_dict(config, train=True)
        self.test_dataset = eval_dataset or get_dataset_dict(config, train=True)
        self.verbose = verbose

        self.is_in_train = False

        if 'train' in config['mode']:
            log.info(f"Initializing loss function and optimizer")

            # Loss function initialization
            self.loss_func = loss_func or get_loss_func(self.train_dataset['target'],
                                                        config['loss_function'],
                                                        config['device'])
            self.model.loss_func = self.loss_func

            # Optimizer initialization
            self.optimizer = optimizer or self.create_optimizer()

            # Resume model training from a saved epoch
            self.start_epoch = self.config['train']['start_epoch'] + 1

    def create_optimizer(self) -> torch.optim.Optimizer:
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
                    
    def train(self) -> None:
        """Entry point for training loop."""
        self.is_in_train = True
        self.epochs = self.config['train']['epochs']

        # Setup a training grid (number of samples, minibatches, and timesteps)
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['x_nn_scaled'],
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

                dataset_sample = get_training_sample(
                    self.train_dataset,
                    n_samples,
                    n_timesteps,
                    self.config
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

            self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)

        log.info(f"Training complete.")


    def test(self) -> None:
        """Run testing loop and save results."""
        log.info(f"Testing model: {self.config['name']}")
        self.is_in_test = True

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.test_dataset['target']

        # Get start and end indices for each batch.
        n_samples = self.test_dataset['x_nn_scaled'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Testing loop
        log.info(f"Begin validation on {len(batch_start)} batches...")
        for i in tqdm.tqdm(range(len(batch_start)), desc="Testing", leave=False, dynamic_ncols=True):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = get_validation_sample(
                self.test_dataset,
                batch_start[i],
                batch_end[i],
                self.config
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
        self._calculate_metrics(batch_predictions, observations)

    def _calculate_metrics(
            self,
            batch_predictions: List[Dict[str, torch.Tensor]],
            observations: torch.Tensor
        ) -> None:
        """Calculate and save test metrics for each prediction type."""
        preds_list, obs_list, name_list = [], [], []

        # Compile flow predictions and corresponding observations
        flow_preds = torch.cat([pred['flow_sim'] for pred in batch_predictions], dim=1)
        flow_obs = observations[:, :, 0]

        # Remove warm-up period if needed
        if self.config['dpl_model']['phy_model']['warm_up_states']:
            flow_obs = flow_obs[self.config['dpl_model']['phy_model']['warm_up']:, :]

        # Add to lists for metrics computation
        preds_list.append(flow_preds.numpy())
        obs_list.append(np.expand_dims(flow_obs, 2))
        name_list.append('flow')

        # Calculate statistics and save results to CSV
        stat_dicts = [
            metrics(np.swapaxes(pred.squeeze(), 1, 0), np.swapaxes(obs.squeeze(), 1, 0))
            for pred, obs in zip(preds_list, obs_list)
        ]

        for stat_dict, name in zip(stat_dicts, name_list):
            metric_df = pd.DataFrame(
                [[np.nanmedian(stat_dict[key]), np.nanstd(stat_dict[key]), np.nanmean(stat_dict[key])]
                 for key in stat_dict.keys()],
                index=stat_dict.keys(), columns=['median', 'STD', 'mean']
            )
            metric_df.to_csv(os.path.join(self.config['testing_path'], f'metrics_{name}.csv'))

    def _log_epoch_stats(
            self,
            epoch: int,
            loss_dict: Dict[str, float],
            n_minibatch: int,
            start_time: float
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
