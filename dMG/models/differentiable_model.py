from ast import Dict
import torch.nn

from hydroDL2 import load_model
  
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.mlp_models import MLPmul
from models.neural_networks.ann_models import AnnModel


class dPLHydroModel(torch.nn.Module):
    """
    Default class for instantiating a differentiable hydrology model
    (i.e., parameterization NN(s) + physics model)
    """
    def __init__(self, config, model_name):
        super(dPLHydroModel, self).__init__()
        self.config = config
        self.model_name = model_name 
        self._init_model()

    def _init_model(self):
        """
        Initialize a hydrology model and any parameterization networks.
        """
        # Physics model init
        ## TODO: Set this up as dynamic module import instead.
        if self.model_name == 'HBV':
            self.hydro_model = load_model('HBV')
            self.hydro_model= self.hydro_model(self.config)
        elif self.model_name == 'HBV_11p':
            self.hydro_model = load_model('HBV_11p')
            self.hydro_model= self.hydro_model()
        elif self.model_name == 'PRMS':
            self.hydro_model = load_model('PRMS')
            self.hydro_model= self.hydro_model()
        else:
            raise ValueError(self.model_name, "is not a valid hydrology model.")

        # Get dims of pNN model(s).
        self.get_nn_model_dims()
        
        # Parameterization NN (pNN) init.
        if self.config['pnn_model']['model'] == 'LSTM':
            self.NN_model = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['pnn_model']['hidden_size'],
                dr=self.config['pnn_model']['dropout']
            )
        elif self.config['pnn_model']['model'] == 'MLP':
            self.NN_model = MLPmul(
                self.config,
                nx=self.nx,
                ny=self.ny
            )
        else:
            raise ValueError(self.config['pnn_model'], "is not a valid neural network type.")
        
    def get_nn_model_dims(self) -> None:
        """ Return dimensions for pNNs. """
        n_forc = len(self.config['observations']['nn_forcings'])
        n_attr = len(self.config['observations']['nn_attributes'])
        self.n_model_params = len(self.hydro_model.parameters_bound)
        self.n_rout_params = len(self.hydro_model.conv_routing_hydro_model_bound)
        
        self.nx = n_forc + n_attr
        self.ny = self.config['dpl_model']['nmul'] * self.n_model_params

        if self.config['routing_hydro_model'] == True:
            self.ny += self.n_rout_params

    def breakdown_params(self, params_all) -> None:
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny]

        # Hydro params
        params_dict['hydro_params_raw'] = torch.sigmoid(
            params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.config['dpl_model']['nmul']]).view(
            params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
            self.config['dpl_model']['nmul'])
        
        # Routing params
        if self.config['routing_hydro_model'] == True:
            params_dict['conv_params_hydro'] = torch.sigmoid(
                params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.config['dpl_model']['nmul']:])
        else:
            params_dict['conv_params_hydro'] = None
        return params_dict

    def forward(self, dataset_dict_sample) -> None:
        # Parameterization + unpacking for physics model.
        params_all = self.NN_model(
            dataset_dict_sample['inputs_nn_scaled'],
            # tRange=t_range,
            # seqMode=True)
            )
        params_dict = self.breakdown_params(params_all)
        
        # Physics model
        # NOTE: conv_params_hydro == rtwts or routpara in hydroDL
        flow_out = self.hydro_model(
            dataset_dict_sample['x_hydro_model'],
            params_dict['hydro_params_raw'],
            self.config,
            static_idx=self.config['phy_model']['stat_param_idx'],
            warm_up=self.config['phy_model']['warm_up'],
            routing=self.config['routing_hydro_model'],
            conv_params_hydro=params_dict['conv_params_hydro']
        )

        # Baseflow index percentage; (from Farshid)
        # Using two deep groundwater buckets: gwflow & bas_shallow
        if 'bas_shallow' in flow_out.keys():
            baseflow = flow_out['gwflow'] + flow_out['bas_shallow']
        else:
            baseflow = flow_out['gwflow']
        flow_out['BFI_sim'] = 100 * (torch.sum(baseflow, dim=0) / (
                torch.sum(flow_out['flow_sim'], dim=0) + 0.00001))[:, 0]

        return flow_out
