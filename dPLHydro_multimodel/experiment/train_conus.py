"""
Train a differentiable model with CONUS MERIT data.
"""
import logging
import time
from typing import Dict, Any

import torch
import tqdm

from conf.config import Config
from core.data import n_iter_nt_ngrid, take_sample_train_merit
from core.data.conus_merit_processor import get_data_dict
from core.utils import save_model
from models.model_handler import ModelHandler

log = logging.getLogger(__name__)



class TrainModel:
    """
    High-level handler for training models on CONUS MERIT data; retrieves and 
    formats training data, initializes model, sets optimizer,
    and runs training.
    """
    def __init__(self, config: Config):
        self.config = config
        
        if config['ensemble_type'] != 'none':
            raise NotImplementedError("Multimodel ensembling with for CONUS MERIT not supported.")

        # Initialize differentiable model w/ optimizer.
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])
        
    def run(self, experiment_tracker) -> None:
        """
        High-level management of ensemble/non-ensemble model training.
        """
        log.info(f"Training model: {self.config['name']} | Collecting training data")

        # Load forcings + attributes.
        self.dataset_dict, self.config = get_data_dict(self.config, train=True)
        
        # Separate out MERIT info;
        self.info_dict = dict()
        self.info_dict['gage_key'] = self.dataset_dict['gage_key']
        self.info_dict['area_info'] = self.dataset_dict['area_info']
        del self.dataset_dict['gage_key'], self.dataset_dict['area_info']

        # Setup training grid.
        nt = self.dataset_dict['inputs_nn_scaled'].shape[0]
        ngrid_train = len(self.info_dict['gage_key'])

        _, minibatch_iter, nt = n_iter_nt_ngrid(
            self.dataset_dict['inputs_nn_scaled'],
            self.config['train_t_range'],
            self.config,
            ngrid=ngrid_train
        )
        ngrid_train = len(self.info_dict['gage_key'])
        
        # Initialize loss function(s) and optimizer.
        self.dplh_model_handler.init_loss_func(self.dataset_dict['obs'])
        optim = self.dplh_model_handler.optim

        start_ep = self.config['checkpoint']['start_epoch'] if self.config['use_checkpoint'] else 1

        self.maxNMerit = 0  # For handling MERIT basin data
     
        # Start training.
        for epoch in range(start_ep, self.config['epochs'] + 1):
            start_time = time.perf_counter()

            self._train_epoch(epoch, minibatch_iter, ngrid_train, nt, optim)
            self._log_epoch_stats(epoch, self.ep_loss_dict, minibatch_iter, start_time)

            if epoch % self.config['save_epoch'] == 0:
                self.save_models(epoch)

    def _train_epoch(self, epoch: int, minibatch_iter: int, ngrid_train: Any,
                     nt: int, optim: torch.optim.Optimizer) -> None:
        """
        Forward over a mini-batched epoch and get the loss.
        """
        self.ep_loss_dict = {self.config['hydro_models'][0]: 0.0}
        prog_str = f"Epoch {epoch}/{self.config['epochs']}"

        # Iterate through minibatches.
        for i in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str,
                           leave=False, dynamic_ncols=True):
            
            dataset_dict_sample = take_sample_train_merit(self.config,
                                                          self.dataset_dict,
                                                          self.info_dict,
                                                          ngrid_train,
                                                          nt,
                                                          self.maxNMerit)

            # Forward pass
            model_pred = self.dplh_model_handler(dataset_dict_sample)
            
            # Loss calculation
            loss, self.ep_loss_dict = self.dplh_model_handler.calc_loss(self.ep_loss_dict)
             
            loss.backward()
            optim.step()
            optim.zero_grad()

            print(loss.item())

    def _log_epoch_stats(self, epoch: int, ep_loss_dict: Dict[str, float],
                         minibatch_iter: int, start_time: float) -> None:
        
        ep_loss_dict = {key: value / minibatch_iter for key, value in ep_loss_dict.items()}
        loss_formated = ", ".join(f"{key}: {value:.6f}" for key, value in ep_loss_dict.items())
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info(f"Model loss after epoch {epoch}: {loss_formated} \n~ Runtime {elapsed:.2f} sec,{mem_aloc} Mb reserved GPU memory")

    def save_models(self, epoch: int, frozen_pnn: bool = False) -> None:
        """
        Save pytorch model.
        """
        mod = self.config['hydro_models'][0]
        save_model(self.config, self.dplh_model_handler.model_dict[mod], mod, epoch)
