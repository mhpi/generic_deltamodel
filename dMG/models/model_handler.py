import os
from math import comb
from typing import Dict

import torch.nn
from models.differentiable_model import dPLHydroModel
from core.utils import save_model


class ModelHandler(torch.nn.Module):
    """
    Streamlines instantiation and handling of differentiable models &
    multimodel ensembles.

    Basic functions include managing: 
    - high-level model init
    - optimizer
    - loss function(s)
    - high-level forwarding

    NOTE: In addition to interfacing with experiments, this handler is a plugin
    for BMI. All PMI-interfaced models must ultimately use this handler if they
    are to be BMI compatible.
    """
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self._init_models()
        self.loss_dict = {key: 0 for key in self.config['phy_model']['models']}

        
    def _init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = {}

        if (self.config['ensemble_type'] == 'none') and (len(self.config['phy_model']['models']) > 1):
            raise ValueError("Multiple hydro models given, but ensemble type not specified. Check config.")
        
        elif self.config['train']['run_from_checkpoint']:
            # Reinitialize trained model(s).
            self.parameters = []
            for mod in self.config['phy_model']['models']:
                load_path = self.config['checkpoint'][mod]
                self.model_dict[mod] = torch.load(load_path).to(self.config['device'])
                self.parameters += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
            self.init_optimizer()

        elif self.config['mode'] in ['test']:
            for mod in self.config['phy_model']['models']:
                self.load_model(mod)

        else:
            # Initialize differentiable hydrology model(s) and bulk optimizer.
            self.parameters = []
            for mod in self.config['phy_model']['models']:

                ### TODO: change which models are set to which devices here: ###
                self.model_dict[mod] = dPLHydroModel(self.config, mod).to(self.config['device'])
                self.parameters += list(self.model_dict[mod].parameters())

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
    
    def load_model(self, model) -> None:
        model_name = str(model) + '_model_Ep' + str(self.config['train']['epochs']) + '.pt'
        model_path = os.path.join(self.config['out_path'], model_name)
        try:
            self.model_dict[model] = torch.load(model_path).to(self.config['device'])

            # Overwrite internal config if there is discontinuity:
            if self.model_dict[model].config:
                self.model_dict[model].config = self.config
        except:
            raise FileNotFoundError(f"Model file {model_path} was not found. Check configurations.")
        
    def forward(self, dataset_dict_sample, eval=False):        
        """
        Batch forward one or more differentiable hydrology models.
        """
        self.flow_out_dict = dict()
        self.dataset_dict_sample = dataset_dict_sample

        # Forward
        for mod in self.model_dict:
            ## Test/Validation
            if eval:
                self.model_dict[mod].eval()
                # torch.set_grad_enabled(False)
                self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)
                # torch.set_grad_enabled(True)
            ## Train
            else:
                self.flow_out_dict[mod] = self.model_dict[mod](dataset_dict_sample)
        return self.flow_out_dict

    def calc_loss(self, dataset: Dict[str, torch.Tensor]) -> torch.Tensor:
        comb_loss = 0.0
        for mod in self.model_dict:
            loss = self.loss_fn(self.config,
                           self.flow_out_dict[mod],
                           dataset['obs'],
                           igrid=dataset['iGrid']
                           )
            comb_loss += loss
            self.loss_dict[mod] += loss.item()
        return comb_loss

    def save_model(self, epoch: int) -> None:
        """Save trained model/ensemble model state dict.
        
        Parameters
        ----------
        epoch : int
            Current epoch.
        """
        for mod in self.config['phy_model']['models']:
            save_model(self.config, self.model.model_dict[mod], mod, epoch)
