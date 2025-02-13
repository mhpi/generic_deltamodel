from typing import Any, Dict, Optional

import torch.nn

from core.utils.module_loaders import get_phy_model
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.lstmmlp_models import LSTMMLP
from models.neural_networks.mlp_models import MLPmul


class DeltaModel(torch.nn.Module):
    """Default class for instantiating a differentiable model.
    
    Default modality: 
        Parameterization neural network (NN) -> Physics Model (phy_model)

        - NN: LSTM or MLP
            Learns parameters for the physics model.

        - phy_model: e.g., HBV, HBV_1_1p, PRMS
            Injests NN-generated parameters and produces some target output.
            The target output is compared to some observation to calculate loss
            to train the NN.

    Parameters
    ----------
    phy_model_name : str, optional
        The name of the physics model. The default is None. This allows
        initialization of multiple physics models from the same config dict. If
        not provided, the first model provided in the config dict is used.
    phy_model : torch.nn.Module, optional
        The physics model. The default is None.
    nn_model : torch.nn.Module, optional
        The neural network model. The default is None.
    config : dict, optional
        The configuration dictionary. The default is None.
    device : torch.device, optional
        The device to run the model on. The default is None.
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

        model = get_phy_model(model_name)
        return model(self.config['phy_model'], device=self.device)
    
    def _init_nn_model(self) -> torch.nn.Module:
        """Initialize a NN model.

        NN to learn parameters for the physics model.
            Inputs: forcings/attributes/observed variables.
            Outputs: parameters for the physics model.
        
        TODO: Set this up as dynamic module import instead.
        """
        n_forcings = len(self.config['nn_model']['forcings'])
        n_attributes = len(self.config['nn_model']['attributes'])
        
        # Number of inputs 'x' and outputs 'y' for nn.
        self.nx = n_forcings + n_attributes
        self.ny = self.phy_model.learnable_param_count
        
        model_name = self.config['nn_model']['model']

        # Initialize the nn
        if model_name == 'LSTM':
            model = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['nn_model']['hidden_size'],
                dr=self.config['nn_model']['dropout'],
            )
        elif model_name == 'MLP':
            model = MLPmul(
                self.config,
                nx=self.nx,
                ny=self.ny,
            )
        elif model_name == 'LSTMMLP':
            model = LSTMMLP(
                nx1=self.nx,
                ny1=self.phy_model.learnable_param_count1,
                hiddeninv1=self.config['nn_model']['lstm_hidden_size'],
                nx2=n_attributes,
                ny2=self.phy_model.learnable_param_count2,
                hiddeninv2=self.config['nn_model']['mlp_hidden_size'],
                dr1=self.config['nn_model']['lstm_dropout'],
                dr2=self.config['nn_model']['mlp_dropout'],
            )
        else:
            raise ValueError(f"{model_name} is not a supported neural network model type.")
        return model.to(self.device)
    
    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model."""
        # Parameterization
        if type(self.nn_model) == LSTMMLP:
            parameters = self.nn_model(data_dict['xc_nn_norm'], data_dict['c_nn_norm'])
        else:
            parameters = self.nn_model(data_dict['xc_nn_norm'])        

        # Physics model
        predictions = self.phy_model(
            data_dict,
            parameters,
        )
        
        return predictions
