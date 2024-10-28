"""
For frozen pNN multimodel ensembles: Train a weighting neural network (wNN) to
dynamically ensemble pre-trained differentiable models.
"""
import logging
import time
from typing import Dict, Any

import torch
import tqdm

from conf.config import Config
from core.data import n_iter_nt_ngrid, take_sample_train
from core.data.dataset_loading import get_data_dict
from core.utils import save_model
from models.model_handler import ModelHandler
from models.multimodels.ensemble_network import EnsembleWeights

log = logging.getLogger(__name__)



class TrainWNNModel:
    """
    High-level multimodel training handler; injests pretrained differentiable hydrology models and trains a weighting
    LSTM (wNN) to dynamically join their outputs.
    """
    def __init__(self, config: Config):
        self.config = config

        # Initializing collection of trained differentiable hydrology models and weighting LSTM.
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])
        self.ensemble_lstm = EnsembleWeights(self.config).to(self.config['device'])

    def run(self, experiment_tracker) -> None:
        """
        High-level management of weighting network (wNN) model training.
        """
        log.info(f"Training model: {self.config['name']} | Collecting training data")

        self.dataset_dict, self.config = get_data_dict(self.config, train=True)

        ngrid_train, minibatch_iter, nt, batch_size = n_iter_nt_ngrid(
            self.config['train_t_range'], self.config, self.dataset_dict['inputs_nn_scaled']
            )

        # Initialize loss function(s) and optimizer.
        self.ensemble_lstm.init_loss_func(self.dataset_dict['obs'])
        self.ensemble_lstm.init_optimizer()
        optim = self.ensemble_lstm.optim

        start_ep = self.config['checkpoint']['start_epoch'] if self.config['use_checkpoint'] else 1
 
        for epoch in range(start_ep, self.config['epochs'] + 1):
            start_time = time.perf_counter()

            self._train_epoch(epoch, minibatch_iter, ngrid_train, nt, batch_size, optim)
            self._log_epoch_stats(epoch, self.ep_loss, minibatch_iter, start_time)
            
            # Save model
            if epoch % self.config['save_epoch'] == 0:
                self.save_models(epoch, frozen_pnn=True)
    
    def _train_epoch(self, epoch: int, minibatch_iter: int, ngrid_train: Any, nt: int,
                     batch_size: int, optim: torch.optim.Optimizer) -> None:
        """
        Forward over a mini-batched epoch and get the loss.
        """
        # Initialize loss dictionary.
        ep_loss = 0
        prog_str = f"Epoch {epoch}/{self.config['epochs']}"

        rb_tot = 0
        sf_tot = 0

        # Iterate through minibatches.
        for i in tqdm.tqdm(range(1, minibatch_iter + 1), desc=prog_str,
                           leave=False, dynamic_ncols=True):
            dataset_dict_sample = take_sample_train(self.config, self.dataset_dict,
                                                    ngrid_train, nt, batch_size)

            # Forward pass
            model_preds = self.dplh_model_handler(dataset_dict_sample, eval=False)
            self.ensemble_lstm(dataset_dict_sample)
        
            total_loss, loss_rb, loss_sf = self.ensemble_lstm.calc_loss(model_preds)

            rb_tot += loss_rb.item()
            sf_tot += loss_sf.item()

            

            total_loss.backward()
            optim.step()
            optim.zero_grad() # set_to_none=True actually increases runtimes.
            ep_loss += total_loss.item()

            # print(f"Current epoch loss {ep_loss} and batch loss {loss.item()}")

        print("rb loss:", rb_tot/minibatch_iter)
        print("sf loss:", sf_tot/minibatch_iter)
        
        self.ep_loss = ep_loss

    def _log_epoch_stats(self, epoch: int, ep_loss: float,
                         minibatch_iter: int, start_time: float) -> None:
        ep_loss = ep_loss/ minibatch_iter
        elapsed = time.perf_counter() - start_time
        mem_aloc = int(torch.cuda.memory_reserved(device=self.config['device']) * 0.000001)
        log.info("wNN loss after epoch {}: {:.6f} \n".format(epoch,ep_loss) +
                 "~ Runtime {:.2f} sec, {} Mb reserved GPU memory".format(elapsed,mem_aloc))
        
    def save_models(self, epoch: int, frozen_pnn: bool = False) -> None:
        """
        Save hydrology and/or weighting models.
        frozen_pnn flag specifies only saving weighting model (wNN) after
        parameterization networks (pNNs) have been frozen.
        """
        if frozen_pnn:
            save_model(self.config, self.ensemble_lstm.lstm, 'wNN', epoch)
        else:
            for mod in self.config['hydro_models']:
                save_model(self.config, self.dplh_model_handler.model_dict[mod], mod, epoch)

            if self.config['ensemble_type'] == 'free_pnn':
                save_model(self.config, self.ensemble_lstm.lstm, 'wNN', epoch)
                