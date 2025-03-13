from typing import Any, Dict, Optional

import torch.nn

from dMG.core.utils.factory import import_phy_model, load_nn_model
from dMG.models.neural_networks.lstm_mlp import LstmMlpModel


class DeltaModel(torch.nn.Module):
    """Default class for instantiating a differentiable model.
    
    Default modality: 
        Parameterization neural network (NN) -> Physics Model (phy_model)

        - NN: e.g., LSTM, MLP, KNN
            Learns parameters for the physics model.

        - phy_model: e.g., HBV 1.0, HBV 1.1p
            A parameterized physics model that ingests NN-generated parameters
            and produces some target output. This output is compared to some
            observation to calculate a loss used to train the NN.

    Parameters
    ----------
    phy_model_name : str, optional
        The name of the physics model. Default is None. This allows
        initialization of multiple physics models from the same config dict. If
        not provided, the first model provided in the config dict is used.
    phy_model : torch.nn.Module, optional
        The physics model. Default is None.
    nn_model : torch.nn.Module, optional
        The neural network model. Default is None.
    config : dict, optional
        The configuration dictionary. Default is None.
    device : torch.device, optional
        The device to run the model on. Default is None.
    """
    def __init__(
        self,
        phy_model_name: Optional[str] = None,
        phy_model: Optional[torch.nn.Module] = None,
        nn_model: Optional[torch.nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'Differentiable Model'
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if nn_model and phy_model:
            self.phy_model = phy_model.to(self.device)
            self.nn_model = nn_model.to(self.device)
        elif config:
            # Initialize new models.
            self.phy_model = self._init_phy_model(phy_model_name)
            self.nn_model = self._init_nn_model()
        else:
            raise ValueError("A (1) neural network and physics model or (2) configuration dictionary is required.")

        self.initialized = True
    
    def _init_phy_model(self, phy_model_name) -> torch.nn.Module:
        """Initialize a physics model."""
        if phy_model_name:
            model_name = phy_model_name
        elif self.config['phy_model']:
            model_name = self.config['phy_model']['model'][0]
        else:
            raise ValueError("A (1) physics model name or (2) model spec in a configuration dictionary is required.")

        model = import_phy_model(model_name)
        return model(self.config['phy_model'], device=self.device)
    
    def _init_nn_model(self) -> torch.nn.Module:
        """Initialize a neural network model."""
        return load_nn_model(
            self.phy_model,
            self.config,
            device=self.device
        )
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model."""
        # try:
        #     # TODO: fix dynamic class import. Currently LstmMlpModel won't be recognized 
        #     # correctly because of this, so we get
        #     # type(self.nn_model) != models.neural_networks.lstm_mlp.LstmMlpModel
        #     from LstmMlpModel import LstmMlpModel
        # except:
        #     pass
        # NN
        if type(self.nn_model).__name__ == 'LstmMlpModel':
            parameters = self.nn_model(data_dict['xc_nn_norm'], data_dict['c_nn_norm'])
        else:
            parameters = self.nn_model(data_dict['xc_nn_norm'])        

        print(f"DEBUG ----------- LSTM param mean: {parameters[0].mean().item()}")
        
        # Physics model
        predictions = self.phy_model(
            data_dict,
            parameters,
        )
        
        return predictions
