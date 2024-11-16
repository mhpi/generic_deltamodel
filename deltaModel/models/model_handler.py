import os
from typing import Dict, List

import torch.nn
from core.utils import save_model
from models.differentiable_model import DeltaModel


class ModelHandler(torch.nn.Module):
    """Streamlines handling of differentiable models and multimodel ensembles.

    This interface additionally acts as a link to the CSDMS BMI, enabling
    compatibility with the NOAA-OWP NextGen framework.

    Basic functions include managing: 
    - high-level model init
    - loss function(s)
    - model forwarding
    - multimodel ensembles (planned)
    - multi-GPU compute (planned)

    Parameters
    ----------
    config : dict
        Configuration settings for the model.
    """
    def __init__(self, config):
        super(ModelHandler, self).__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self._init_models()
        self.loss_dict = {key: 0 for key in self.config['dpl_model']['phy_model']['model']}

        
    def _init_models(self):
        """
        Initialize and store each differentiable hydro model and optimizer in
        the multimodel.
        """
        self.model_dict = {}

        if (self.config['ensemble_type'] == 'none') and (len(self.config['dpl_model']['phy_model']['model']) > 1):
            raise ValueError("Multiple hydro models given, but ensemble type not specified. Check config.")
        
        elif self.config['train']['resume_from_checkpoint'] > 0:
            # Reinitialize trained model(s) from checkpoint.
            start_epoch = self.config['train']['start_epoch']
            try:        
                for mod in self.config['dpl_model']['phy_model']['model']:
                    save_name = str(mod) + '_model_Ep' + str(start_epoch) + '.pt'
                    load_path = os.path.join(self.config['out_path'], save_name)

                    self.model_dict[mod] = torch.load(load_path).to(self.config['device'])

                    self.model_dict[mod].zero_grad()
                    self.model_dict[mod].train()
                self.init_optimizer()
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file {load_path} was not found. Make sure model checkpoint exists for epoch {start_epoch}.")

        elif self.config['mode'] in ['test']:
            for mod in self.config['dpl_model']['phy_model']['model']:
                self.load_model(mod)

        else:
            # Initialize differentiable hydrology model(s) and bulk optimizer.
            for mod in self.config['dpl_model']['phy_model']['model']:
                self.model_dict[mod] = DeltaModel(
                    phy_model_name=mod,
                    config=self.config['dpl_model']
                    ).to(self.config['device'])

                self.model_dict[mod].zero_grad()
                self.model_dict[mod].train()
    
    def get_parameters(self) -> List[torch.Tensor]:
        """Return all model parameters."""
        self.parameters = []
        for mod in self.config['dpl_model']['phy_model']['model']:
            self.parameters += list(self.model_dict[mod].parameters())

        return self.parameters
    
    def load_model(self, model) -> None:
        model_name = str(model) + '_model_Ep' + str(self.config['test']['test_epoch']) + '.pt'
        model_path = os.path.join(self.config['out_path'], model_name)
        try:
            self.model_dict[model] = DeltaModel(
                phy_model_name=model,
                config=self.config['dpl_model']
            ).to(self.config['device'])
            self.model_dict[model].load_state_dict(torch.load(model_path))

            # Overwrite internal config if there is discontinuity:
            if self.model_dict[model].config:
                self.model_dict[model].config = self.config
        except:
            raise FileNotFoundError(f"Model file {model_path} was not found. Check configurations.")
        
    def forward(self, dataset_dict_sample, eval=False):        
        """Batch forward one or more differentiable hydrology models."""
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
            loss = self.loss_func(
                self.flow_out_dict[mod]['flow_sim'],
                dataset['target'],
                n_samples=dataset['batch_sample']
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
        for model in self.config['dpl_model']['phy_model']['model']:
            save_model(self.config, self.model_dict[model], model, epoch)

