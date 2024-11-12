import logging
import time
from typing import Any, Dict

import torch
import tqdm
from core.data import calc_training_params, take_sample_train
from core.data.dataset_loading import get_dataset_dict
from models.loss_functions import get_loss_fn
from models.model_handler import ModelHandler
from torch.nn import Module

log = logging.getLogger(__name__)



class TrainModel:
    """Generic trainer for differentiable models.
    
    Retrieves and formats training data, initializes optimizer and loss function,
    and runs the training loop.

    Parameters
    ----------
    config : dict
        Configuration settings for the model and experiment.
    model : Module, optional
        dPL (differentiable parameter learning) model object.
    dataset : dict, optional
        Dataset dictionary containing forcings and attributes
    """
    def __init__(self, config: Dict[str, Any], model: Module = None,
                 dataset: dict = None) -> None:
        self.config = config
        
        if model is None:
            self.model = ModelHandler(config)
        else:
            self.model = model

        if dataset is None:
            log.info(f"Loading training data")
            self.dataset = get_dataset_dict(config, train=True)
        else:
            self.dataset = dataset
            
    def run(self) -> None:
        """Run training loop."""
        log.info(f"Training model: {self.config['name']}")
        
        # Setup training grid
        n_grid, n_minibatch, nt = calc_training_params(
            self.dataset['inputs_nn_scaled'],
            self.config['train_t_range'],
            self.config
            )

        # Initialize loss function and optimizer
        log.info(f"Initializing loss function and optimizer")
        self.loss_fn = get_loss_fn(self.config, self.dataset['obs'])
        self.model.loss_fn = self.loss_fn
        self.optim = torch.optim.Adadelta(
            self.model.parameters,
            lr=self.config['dpl_model']['nn_model']['learning_rate']
            )

        if self.config['train']['resume_from_checkpoint']:
            start_ep = self.config['train']['start_epoch']
        else:
            start_ep = 1

        # Training loop
        for epoch in range(start_ep, self.config['train']['epochs'] + 1):
            start_time = time.perf_counter()

            self._train_epoch(epoch, n_minibatch, n_grid, nt)
            self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

            if epoch % self.config['train']['save_epoch'] == 0:
                self.model.save_model(epoch)

    def _train_epoch(self, epoch: int, n_minibatch: int, n_grid: Any,
                     nt: int) -> None:
        """Forward over a batched epoch and compute the loss.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        minibatch_iter : int
            Number of minibatches.
        ngrid_train : Any
            Training grid.
        nt : int
            Number of timesteps.
        """
        total_loss = 0.0

        prog_str = f"Epoch {epoch}/{self.config['train']['epochs']}"

        # Iterate through minibatches
        for i in tqdm.tqdm(range(1, n_minibatch + 1), desc=prog_str,
                           leave=False, dynamic_ncols=True):
            dataset_sample = take_sample_train(self.config, self.dataset,
                                               n_grid, nt)

            # Forward pass for hydrology models.
            preds = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)
            
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            total_loss += loss.item()

            print("Batch loss: ", loss.item())

    def _log_epoch_stats(self, epoch: int, loss_dict: Dict[str, float],
                         n_minibatch: int, start_time: float) -> None:
        """Log epoch statistics.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        ep_loss_dict : dictg
            Dictionary of model losses.
        minibatch_iter : int
            Number of minibatches.
        start_time : float
            Start time of epoch.
        """
        avg_loss_dict = {key: value / n_minibatch + 1 for key, value in loss_dict.items()}
        loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in avg_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info(f"Model loss after epoch {epoch}: {loss_formated} \n~ Runtime {elapsed:.2f} sec,{mem_aloc} Mb reserved GPU memory")
                