import os
from typing import Dict, List

import numpy as np
import torch
from conf.config import Config
from core.calc.RangeBoundLoss import RangeBoundLoss
from core.utils.utils import find_shared_keys
from models.loss_functions import get_loss_function
from models.neural_networks.lstm_models import CudnnLstmModel

class EnsembleGenerator(torch.nn.Module):
    """
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.name = "Multimodel Ensemble Weights Generator"
        self.range_bound_loss = RangeBoundLoss(config)
        self.lstm = None
        self.weights = None
        self.weights_scaled = None
        self.weights_dict = None
        self.ensemble_pred = {}
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the LSTM model using configuration settings."""
        if self.config['use_checkpoint']:
            self._load_checkpoint()
        elif self.config['mode'] in ['test', 'test_bmi']:
            self._load_model('wNN')
        else:
            self._initialize_new_model()

    def _load_checkpoint(self) -> None:
        """Load a trained model from a checkpoint for resuming training."""
        load_path = self.config['checkpoint']['weighting_nn']
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file {load_path} not found.")
        self.lstm = torch.load(load_path).to(self.config['device'])
        self.lstm.zero_grad()
        self.lstm.train()

    def _load_model(self, model_name: str) -> None:
        """Load a model for testing."""
        file_name = f"{model_name}_model_Ep{self.config['epochs']}.pt"
        model_path = os.path.join(self.config['out_path'], file_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        self.lstm = torch.load(model_path).to(self.config['device'])

    def _initialize_new_model(self) -> None:
        """Initialize a new LSTM model."""
        self._determine_input_output_dimensions()
        self.lstm = CudnnLstmModel(
            nx=self.nx,
            ny=self.ny,
            hiddenSize=self.config['weighting_nn']['hidden_size'],
            dr=self.config['weighting_nn']['dropout']
        ).to(self.config['device'])
        self.lstm.zero_grad()
        self.lstm.train()

    def _get_input_output_dimensions(self) -> None:
        """Determine the input and output dimensions for the LSTM model."""
        self.nx = len(self.config['observations']['var_t_nn'] + \
                      self.config['observations']['var_c_nn'])
        self.ny = len(self.config['hydro_models'])
        
    def init_loss_func(self, obs: np.ndarray) -> None:
        """Initialize the loss function based on configuration and observations."""
        self.loss_func = get_loss_function(
            self.config['weighting_nn'],
            obs
        ).to(self.config['device'])


    def forward(self, dataset_dict_sample: Dict, eval: bool = False) -> Dict:
        """Perform a forward pass through the LSTM model."""
        nn_inputs = dataset_dict_sample['x_nn_scaled']
        if eval:
            self.lstm.eval()

        self.weights = self.lstm(nn_inputs)
        self._scale_weights()

        # Map scaled weights to hydrology models
        self.weights_dict = {
            mod: self.weights_scaled[self.config['warm_up']:, :, i]
            for i, mod in enumerate(self.config['hydro_models'])
        }
        return self.weights_dict
    
    def _scale_weights(self) -> None:
        if self.config['weighting_nn']['method'] == 'sigmoid':
            self.weights_scaled = torch.sigmoid(self.weights)
        elif self.config['weighting_nn']['method'] == 'softmax':
            self.weights_scaled = torch.softmax(self.weights)
        else:
            raise ValueError(self.config['weighting_nn']['method'], "is not a valid model weighting method.")

    def calc_loss(self, hydro_preds_dict: Dict, loss_dict=None) -> float:
        """
        Compute a composite loss: 
        1) Calculates range-bound loss on the lstm weights.

        2) Takes in predictions from set of hydro models, and computes a loss on the linear combination of model predictions using lstm-derived weights.
        """
        # Range-bound loss on weights.
        weights_sum = torch.sum(self.weights_scaled, dim=2)
        loss_rb = self.range_bound_loss([weights_sum])

        # Get ensembled streamflow.
        self.ensemble_models(hydro_preds_dict)

        # Loss on streamflow preds.
        loss_sf = self.loss_func(
            self.ensemble_pred['flow_sim'],
            self.dataset_dict_sample['target'],
            n_samples=self.dataset_dict_sample['batch_sample']
        )
    
        # Debugging
        # print("rb loss:", loss_rb)
        # print("stream loss:", 0.1*loss_sf)


        # Return total_loss for optimizer.
        ###### NOTE: Added e2 factor to streamflow loss to account for ~1 OoM difference.
        total_loss = loss_rb + loss_sf
        if loss_dict:
            loss_dict['wNN'] += total_loss.item()
            return total_loss, loss_dict

        # total_loss.backward()
        # self.optim2.step()
        # self.optim2.zero_grad()
        # comb_loss += total_loss.item()
        # return comb_loss

        return total_loss, loss_rb, loss_sf
    
    def ensemble_models(self, model_preds_dict: Dict[str, np.float32]) -> Dict[str, np.float32]:
        """
        Calculate composite predictions by combining individual hydrology model results scaled by learned nn weights.
        
        Returns: predictions dict with attributes
        'flow_sim', 'srflow', 'ssflow', 'gwflow', 'AET_hydro', 'PET_hydro', 'flow_sim_no_rout', 'srflow_no_rout', 'ssflow_no_rout', 'gwflow_no_rout', 'BFI_sim'
        """
        self.ensemble_pred = dict()

        # Get prediction shared between all models.
        mod_dicts = [model_preds_dict[mod] for mod in self.config['hydro_models']]
        shared_keys = find_shared_keys(*mod_dicts)

        shared_keys.remove('flow_sim_no_rout')

        for key in shared_keys:
            self.ensemble_pred[key] = 0
            for mod in self.config['hydro_models']:
                wts_size = self.weights_dict[mod].size(0)
                pred_size = model_preds_dict[mod][key].squeeze().size()
                if (wts_size != pred_size[0]) and len(pred_size) > 1:
                    # Cut out warmup data present when testing model from loaded mod file.
                    model_preds_dict[mod][key] = model_preds_dict[mod][key][self.config['warm_up']:,:]
                self.ensemble_pred[key] += self.weights_dict[mod] * model_preds_dict[mod][key].squeeze()

        return self.ensemble_pred