from typing import Dict

import torch.nn
from core.data import numpy_to_torch_dict
from hydroDL2 import load_model
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.mlp_models import MLPmul


class DeltaModel(torch.nn.Module):
    """Default class for instantiating a differentiable model.
    
    Default modality: 
        Parameterization NN (pNN) -> Physics Model (phy_model)

        - pNN: LSTM or MLP
            Learns parameters for the physics model.

        - phy_model: e.g., HBV, hbv_v1_1p, PRMS
            Injests pNN-generated parameters and produces some target output.
            The target output is compared to some observation to calculate loss
            to train the pNN.

    TODO: Needs more generalization.

    Parameters
    ----------
    phy_model : torch.nn.Module, optional
        The physics model. The default is None.
    nn_model : torch.nn.Module, optional
        The neural network model. The default is None.
    config : dict, optional
        The configuration dictionary. The default is None.
    """
    def __init__(self, phy_model=None, nn_model=None, phy_model_name=None,config=None, device=None):
        super(DeltaModel, self).__init__()
        self.phy_model = phy_model
        self.nn_model = nn_model
        self.phy_model_name = phy_model_name
        self.config = config
        self.nmul = 16
        self.routing = True
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            self.nmul = config['nmul']
            self.routing = config['phy_model']['routing']
        
        if phy_model is None:
            if config is not None:
                self._init_phy_model()
            else:
                raise ValueError("A physics model or configuration dictionary is required.")
            
        self.param_bounds = self.phy_model.parameter_bounds

        if nn_model is None:
            if config is not None:
                self._init_nn_model()
            else:
                raise ValueError("A neural network or configuration dictionary is required.")
        else:
            self.nx = self.nn_model.nx
            self.ny = self.nn_model.ny

        self.phy_model.to(self.device)
        self.phy_model.device = self.device
        self.nn_model.to(self.device)
        self.initialized = True

    def _init_phy_model(self):
        """Initialize a physics model.
        
        TODO: Set this up as dynamic module import instead.
        """
        if self.phy_model_name == 'HBV':
            self.hydro_model = load_model('HBV')
        elif self.phy_model_name == 'HBV_v1_1p':
            self.hydro_model = load_model('HBV_v1_1p')
        elif self.phy_model_name == 'PRMS':
            self.hydro_model = load_model('PRMS')
        else:
            raise ValueError(self.model_name, "is not a valid physics model.")
        
        self.phy_model= self.hydro_model(self.config)

    def _init_nn_model(self):
        """Initialize a pNN model.
        
        TODO: Set this up as dynamic module import instead.
        """
        # Get input/output dimensions for nn.
        self._get_nn_dims()
        
        model_name = self.config['nn_model']['model']

        # Initialize the nn
        if model_name == 'LSTM':
            self.nn_model = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['nn_model']['hidden_size'],
                dr=self.config['nn_model']['dropout']
            )
        elif model_name == 'MLP':
            self.nn_model = MLPmul(
                self.config,
                nx=self.nx,
                ny=self.ny
            )
        else:
            raise ValueError(self.config['nn_model'], "is not a valid neural network type.")

    def _get_nn_dims(self) -> None:
        """Return dimensions for pNNs."""
        # Number of variables
        n_forcings = len(self.config['nn_model']['forcings'])
        n_attributes = len(self.config['nn_model']['attributes'])
        
        # Number of parameters
        n_params = len(self.param_bounds)
        n_routing_params = len(self.phy_model.conv_routing_hydro_model_bound)
        
        # Total number of inputs and outputs for nn.
        self.nx = n_forcings + n_attributes
        self.ny = self.nmul * n_params
        if self.routing == True:
            # Add routing parameters
            self.ny += n_routing_params

    def breakdown_params(self, params_all) -> None:
        """Extract physics model parameters from pNN output."""
        params_dict = dict()
        learned_params = params_all[:, :, :self.ny]

        # Hydro params
        params_dict['hydro_params_raw'] = torch.sigmoid(
            learned_params[:, :, :len(self.param_bounds) * self.nmul]).view(
                learned_params.shape[0],
                learned_params.shape[1],
                len(self.param_bounds),
                self.nmul)
        
        # Routing params
        if self.routing == True:
            params_dict['conv_params_hydro'] = torch.sigmoid(
                learned_params[-1, :, len(self.param_bounds) * self.nmul:])
        else:
            params_dict['conv_params_hydro'] = None
        return params_dict

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> None:
        """Forward pass for the model."""
        # Convert numpy data to torch tensors.
        data_dict = numpy_to_torch_dict(data_dict, self.device)
        
        # Parameterization + unpacking for physics model.
        params_all = self.nn_model(
            data_dict['inputs_nn_scaled']
            )
        params_dict = self.breakdown_params(params_all)
        
        # Physics model
        predictions = self.phy_model(
            data_dict['x_hydro_model'],
            params_dict['hydro_params_raw'],
            routing_parameters = params_dict['conv_params_hydro'],
        )

        # Baseflow index percentage; (from Farshid)
        # Using two deep groundwater buckets: gwflow & bas_shallow
        if 'bas_shallow' in predictions.keys():
            baseflow = predictions['gwflow'] + predictions['bas_shallow']
        else:
            baseflow = predictions['gwflow']
        predictions['BFI_sim'] = 100 * (torch.sum(baseflow, dim=0) / (
                torch.sum(predictions['flow_sim'], dim=0) + 0.00001))[:, 0]

        return predictions
