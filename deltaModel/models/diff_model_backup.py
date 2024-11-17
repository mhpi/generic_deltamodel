from typing import Dict, Optional

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

        - phy_model: e.g., HBV, HBV1_1p, PRMS
            Injests pNN-generated parameters and produces some target output.
            The target output is compared to some observation to calculate loss
            to train the pNN.

    Parameters
    ----------
    pnn_model : torch.nn.Module, optional
        The neural network model. The default is None.
    phy_model : torch.nn.Module, optional
        The physics model. The default is None.
    config : dict, optional
        The configuration dictionary. The default is None.
    device : torch.device, optional
        The device to run the model on. The default is None.
    """
    def __init__(
            self,
            pnn_model: Optional[torch.nn.Module] = None,
            phy_model: Optional[torch.nn.Module] = None,
            config: Optional[dict] = None,
            device: Optional[torch.device] = None
        ) -> None:
        super().__init__()
        self.name = 'Differentiable Model (pNN -> phy_model)'
        self.config = config
        self.nmul = config.get('nmul', 1)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if pnn_model and phy_model:
            self.pnn_model = pnn_model
            self.phy_model = phy_model
        elif config:
            self.pnn_model = self._init_nn_model(config)
            self.phy_model = self._init_phy_model(config)
        else:
            raise ValueError("A (1) neural network and physics model or (2) configuration dictionary is required.")

        self.routing = config['phy_model']['routing'] or True


            
        self.param_bounds = self.phy_model.parameter_bounds

        if pnn_model is None:
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
    
    def _init_nn_model(self) -> torch.nn.Module:
        """Initialize neural network model from config or defaults.\
        TODO: Set this up as dynamic module import instead.
        """
        if self.config:
            model_type = self.config['nn_model']['type']
            if model_type == 'LSTM':
                return CudnnLstmModel(self.config)  # Pass specific config values
            elif model_type == 'MLP':
                return MLPmul(self.config)
            else:
                raise ValueError(f"{model_type} is not a supported neural network model type.")
        raise ValueError("Config required to initialize neural network model.")
    
    def _init_phy_model(self, phy_model_name: Optional[str]) -> torch.nn.Module:
        """Initialize or load a physics model based on name."""
        # Example with load_model, expand as needed
        if phy_model_name:
            return load_model(phy_model_name)
        elif self.config:
            model_name = self.config.get("phy_model_name")
            return load_model(model_name)
        else:
            raise ValueError("Physics model name or config required to initialize model.")

    def _init_phy_model(self):
        """Initialize a physics model.
        
        TODO: Set this up as dynamic module import instead.
        """
        if self.phy_model_name == 'HBV':
            self.hydro_model = load_model('HBV')
        elif self.phy_model_name == 'HBV1_1p':
            self.hydro_model = load_model('HBV1_1p')
        elif self.phy_model_name == 'PRMS':
            self.hydro_model = load_model('PRMS')
        else:
            raise ValueError(self.model_name, "is not a valid physics model.")
        
        self.phy_model= self.hydro_model(self.config)

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
            data_dict['x_nn_scaled']
            )
        params_dict = self.breakdown_params(params_all)
        
        # Physics model
        predictions = self.phy_model(
            data_dict['x_phy'],
            params_dict['hydro_params_raw'],
            routing_parameters = params_dict['conv_params_hydro'],
        )

        ##### REMOVE AS SEPERATE MODULE IN TRAINER
        # Baseflow index percentage; (from Farshid)
        # Using two deep groundwater buckets: gwflow & bas_shallow
        if 'bas_shallow' in predictions.keys():
            baseflow = predictions['gwflow'] + predictions['bas_shallow']
        else:
            baseflow = predictions['gwflow']
        predictions['BFI_sim'] = 100 * (torch.sum(baseflow, dim=0) / (
                torch.sum(predictions['flow_sim'], dim=0) + 0.00001))[:, 0]

        return predictions
