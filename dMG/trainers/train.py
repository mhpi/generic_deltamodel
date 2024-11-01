""" Vanilla training for differentiable models. """
import logging
import time
from typing import Any, Dict

import torch
import tqdm
from core.data import n_iter_nt_ngrid, take_sample_train
from core.data.dataset_loading import get_data_dict
from core.utils import save_model
from models.model_handler import ModelHandler

log = logging.getLogger(__name__)



class TrainModel:
    """High-level training manager for differentiable models.
    
    Responsible for retrieving and formatting training data, initializing a 
    dPL (differentiable Parameter Learning) model object, *setting the optimizer
    and loss function, and running the training loop.

    TODO*: migrate optimizer and loss function setup from ModelHandler to TrainModel.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize collection of dPL hydrology models w/ optimizer.
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])
    
    def run(self) -> None:
        """High level training manager."""
        log.info(f"Training model: {self.config['name']} | Collecting training data")
        
        # Load forcings + attributes.
        self.dataset_dict, self.config = get_data_dict(self.config, train=True)

        # Setup training grid.
        ngrid_train, minibatch_iter, nt = n_iter_nt_ngrid(
            self.dataset_dict['inputs_nn_scaled'], self.config['train_t_range'], self.config
            )

        # Initialize loss function(s) and optimizer.
        log.info(f"Initializing loss function, optimizer")
        self.dplh_model_handler.init_loss_func(self.dataset_dict['obs'])
        optim = self.dplh_model_handler.optim

        start_ep = self.config['train']['start_epoch'] if self.config['train']['run_from_checkpoint'] else 1

        # Start of training.
        for epoch in range(start_ep, self.config['train']['epochs'] + 1):
            start_time = time.perf_counter()

            self._train_epoch(epoch, minibatch_iter, ngrid_train, nt, optim)
            self._log_epoch_stats(epoch, self.ep_loss_dict, minibatch_iter, start_time)

            if epoch % self.config['train']['save_epoch'] == 0:
                self.save_models(epoch)

    def _train_epoch(self, epoch: int, minibatch_iter: int, ngrid_train: Any,
                     nt: int, optim: torch.optim.Optimizer) -> None:
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
        optim : torch.optim.Optimizer
            Optimizer.
        """
        # Initialize loss dictionary.
        ep_loss_dict = {key: 0 for key in self.config['phy_model']['models']}

        prog_str = f"Epoch {epoch}/{self.config['train']['epochs']}"

        # Iterate through minibatches.
        for i in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str,
                           leave=False, dynamic_ncols=True):
            dataset_dict_sample = take_sample_train(self.config, self.dataset_dict,
                                                    ngrid_train, nt)

            # Forward pass for hydrology models.
            model_preds = self.dplh_model_handler(dataset_dict_sample)
            hydro_loss, ep_loss_dict = self.dplh_model_handler.calc_loss(ep_loss_dict)
            
            total_loss = hydro_loss
            total_loss.backward()
            optim.step()
            optim.zero_grad()

            # print("Batch loss: ", total_loss.item())
            # print("loss dict", ep_loss_dict)

        self.ep_loss_dict = ep_loss_dict

    def _log_epoch_stats(self, epoch: int, ep_loss_dict: Dict[str, float],
                         minibatch_iter: int, start_time: float) -> None:
        """Log epoch statistics.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        ep_loss_dict : dict
            Dictionary of model losses.
        minibatch_iter : int
            Number of minibatches.
        start_time : float
            Start time of epoch.
        """
        ep_loss_dict = {key: value / minibatch_iter for key, value in ep_loss_dict.items()}
        loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in ep_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info(f"Model loss after epoch {epoch}: {loss_formated} \n~ Runtime {elapsed:.2f} sec,{mem_aloc} Mb reserved GPU memory")

    def save_models(self, epoch: int, frozen_pnn: bool = False) -> None:
        """Save trained model state dict.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        frozen_pnn : bool, optional
            Flag to freeze the pNN model.
        """
        for mod in self.config['phy_model']['models']:
            save_model(self.config, self.dplh_model_handler.model_dict[mod], mod, epoch)
                