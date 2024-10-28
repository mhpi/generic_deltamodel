import os
import numpy as np
import torch
from typing import Dict, List

from conf.config import Config
from core.calc.RangeBoundLoss import RangeBoundLoss
from core.utils.utils import find_shared_keys
from models.loss_functions import get_loss_function
from models.neural_networks.lstm_models import CudnnLstmModel



class EnsembleWeights(torch.nn.Module):
    """
    Interface for an ensemble weighting LSTM (wNN) used for combining outputs of
    multiple hydrology models.
    """
    def __init__(self, config: Config):
        super(EnsembleWeights, self).__init__()
        self.config = config
        self.name = 'Ensemble Weighting LSTM (wNN)'
        self.range_bound_loss = RangeBoundLoss(config)
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize weighting LSTM.
        """
        if self.config['use_checkpoint']:
            # Reinitialize trained model to continue training.
            load_path = self.config['checkpoint']['weighting_nn']
            self.lstm = torch.load(load_path).to(self.config['device'])
            self.model_params = list(self.lstm.parameters())
            self.lstm.zero_grad()
            self.lstm.train()
        elif self.config['mode'] in ['test', 'test_bmi']:
            self.load_model('wNN')
        else:
            self.get_nn_model_dim()
            self.lstm = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['weighting_nn']['hidden_size'],
                dr=self.config['weighting_nn']['dropout']
            ).to(self.config['device'])
            self.model_params = list(self.lstm.parameters())
            self.lstm.zero_grad()
            self.lstm.train()
    
    def load_model(self, model) -> None:
        model_name = f"{model}_model_Ep{self.config['epochs']}.pt"
        model_path = os.path.join(self.config['output_dir'], model_name)
        try:
            self.lstm = torch.load(model_path).to(self.config['device']) 
        except:
            raise FileNotFoundError(f"Model file {model_path} was not found.")
        
    def init_loss_func(self, obs: np.float32) -> None:
        self.loss_func = get_loss_function(self.config['weighting_nn'],
                                       obs).to(self.config['device'])

    def init_optimizer(self) -> None:
        self.optim = torch.optim.Adadelta(self.model_params, lr=self.config['weighting_nn']['learning_rate'])

    def get_nn_model_dim(self) -> None:
        self.nx = len(self.config['observations']['var_t_nn'] + self.config['observations']['var_c_nn'])
        self.ny = len(self.config['hydro_models'])  # Output size of pNN

    def forward(self, dataset_dict_sample: Dict, eval=False) -> Dict:
        self.dataset_dict_sample = dataset_dict_sample

        # Get scaled mini-batch of basin forcings + attributes.
        # inputs_nn_scaled = x_nn + c_nn, forcings + basin attributes
        nn_inputs = dataset_dict_sample['inputs_nn_scaled']

        # For testing
        if eval: self.lstm.eval()

        # Forward
        self.weights = self.lstm(nn_inputs)
        self._scale_weights()

        self.weights_dict = dict()
        for i, mod in enumerate(self.config['hydro_models']):
            # Extract predictions into model dict + remove warmup period from output.
            self.weights_dict[mod] = self.weights_scaled[self.config['warm_up']:,:,i]

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
        loss_sf = self.loss_func(self.config,
                                 self.ensemble_pred['flow_sim'],
                                 self.dataset_dict_sample['obs'],
                                 igrid=self.dataset_dict_sample['iGrid']
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
