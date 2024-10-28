from ast import Dict
import torch.nn
from tqdm import trange
from models.hydro_models.HBV.hbv import HBVMulTDET as HBV
from models.hydro_models.HBV.hbv_capillary import HBVMulTDET as HBVcap
from models.hydro_models.HBV.hbv_waterloss import HBVMulTDET_WaterLoss as HBV_WL
from models.hydro_models.marrmot_PRMS.prms_marrmot import prms_marrmot
from models.hydro_models.marrmot_PRMS_gw0.prms_marrmot_gw0 import \
    prms_marrmot_gw0
from models.hydro_models.SACSMA.SACSMAmul import SACSMAMul
from models.hydro_models.SACSMA_with_snowpack.SACSMA_snow_mul import \
    SACSMA_snow_Mul
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
            self.hydro_model = HBV(self.config)
        elif self.model_name == 'HBV_capillary':
            self.hydro_model = HBVcap()
        elif self.model_name == 'HBV_waterLoss':
            self.hydro_model = HBV_WL()
        elif self.model_name == 'marrmot_PRMS':
            self.hydro_model = prms_marrmot()
        elif self.model_name == 'marrmot_PRMS_gw0':
            self.hydro_model = prms_marrmot_gw0()
        elif self.model_name == 'SACSMA':
            self.hydro_model = SACSMAMul()
        elif self.model_name == 'SACSMA_with_snow':
            self.hydro_model = SACSMA_snow_Mul()
        else:
            raise ValueError(self.model_name, "is not a valid hydrology model.")

        # Get dims of pNN model(s).
        self.get_nn_model_dims()
        
        # Parameterization NN (pNN) init.
        if self.config['pnn_model'] == 'LSTM':
            self.NN_model = CudnnLstmModel(
                nx=self.nx,
                ny=self.ny,
                hiddenSize=self.config['hidden_size'],
                dr=self.config['dropout']
            )
        elif self.config['pnn_model'] == 'MLP':
            self.NN_model = MLPmul(
                self.config,
                nx=self.nx,
                ny=self.ny
            )
        else:
            raise ValueError(self.config['pnn_model'], "is not a valid neural network type.")
        
        # ANN pNN init for HBV2.0...
        if (type(self.hydro_model) == HBV_WL) and (self.config['ann_opt']):
            self.ANN_model = AnnModel(
                nx = self.nx_ann,
                ny=self.ny_ann,
                hiddenSize=self.config['ann_opt']['hidden_size'],
                dropout_rate=self.config['ann_opt']['dropout']
            )

    def get_nn_model_dims(self) -> None:
        """
        Return dimensions for pNNs.
        """
        n_forc = len(self.config['observations']['var_t_nn'])
        n_attr = len(self.config['observations']['var_c_nn'])
        self.n_model_params = len(self.hydro_model.parameters_bound)
        self.n_rout_params = len(self.hydro_model.conv_routing_hydro_model_bound)
        
        if type(self.hydro_model) != HBV_WL:
            # Typical setup for all non-HBV2.0 models...
            self.nx = n_forc + n_attr
            self.ny = self.config['nmul'] * self.n_model_params

            if self.config['routing_hydro_model'] == True:
                self.ny += self.n_rout_params
        
        else:  
            # Using ANN requires different dims for pNN also.
            n_feat_ann = self.n_model_params - self.config['ann_opt']['n_features']
            self.nx = n_forc + n_attr
            self.ny = self.config['ann_opt']['nmul'] * n_feat_ann

            # ANN dims
            self.nx_ann = n_attr
            self.ny_ann = self.config['ann_opt']['nmul'] * self.config['ann_opt']['n_features']

            if self.config['routing_hydro_model'] == True:
                self.ny_ann += self.n_rout_params

    def breakdown_params(self, params_all) -> None:
        params_dict = dict()
        params_hydro_model = params_all[:, :, :self.ny]

        # Hydro params
        params_dict['hydro_params_raw'] = torch.sigmoid(
            params_hydro_model[:, :, :len(self.hydro_model.parameters_bound) * self.config['nmul']]).view(
            params_hydro_model.shape[0], params_hydro_model.shape[1], len(self.hydro_model.parameters_bound),
            self.config['nmul'])
        
        # Routing params
        if self.config['routing_hydro_model'] == True:
            params_dict['conv_params_hydro'] = torch.sigmoid(
                params_hydro_model[-1, :, len(self.hydro_model.parameters_bound) * self.config['nmul']:])
        else:
            params_dict['conv_params_hydro'] = None
        return params_dict

    def forward(self, dataset_dict_sample) -> None:
        # Data loading: either
        # if self.config['t_range']:
        #     # a single timestep or
        #     print( self.config['t_range'])
        #     # t_range = self.config['t_range']
        #     seq_mode = self.config['seq_mode']
        # else:
        #     # an entire sequence
        #     t_range = None
        #     seq_mode = True


        if type(self.hydro_model) != HBV_WL:
            # Forward non-HBV2.0 models.
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
                dataset_dict_sample['c_hydro_model'],
                params_dict['hydro_params_raw'],
                self.config,
                static_idx=self.config['static_index'],
                warm_up=self.config['warm_up'],
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
        
        else:
            # Forward HBV2.0 models.
            flow_out = self.forward_water_loss(dataset_dict_sample)

        return flow_out
    
    def forward_water_loss(self, dataset_dict_sample: Dict) -> None:
        """
        Forward for HBV2.0 water loss model with Merit + GAGESII basin data.
        """
        # Forward non-HBV2.0 models.
        # Parameterization + unpacking for physics model.
        params_raw = self.NN_model(
            dataset_dict_sample['inputs_nn_scaled'].float(),
            # tRange=t_range,
            # seqMode=True)
            )

        # HBV params
        hydro_params = torch.sigmoid(params_raw).view(
            params_raw.shape[0],
            params_raw.shape[1],
            self.n_model_params - self.config['ann_opt']['n_features'],
            self.config['nmul']
        )

        n_merit = dataset_dict_sample['c_nn'].shape[0]
        n_features = self.config['ann_opt']['n_features']
        nmul = self.config['ann_opt']['nmul']
        idx = n_features * nmul

        waterloss_params_raw = self.ANN_model(dataset_dict_sample['c_nn'].float())
        waterloss_params = waterloss_params_raw[:, :idx].view(n_merit, n_features, nmul)  # dim: Time, Gage, Para

        rout_params = waterloss_params_raw[:, idx:idx + self.n_rout_params]  # Routing para dim:[Ngage, nmul*2] or [Ngage, 2]

        # Physics model
        # NOTE: conv_params_hydro == rtwts or routpara in hydroDL
        flow_out = self.hydro_model(
            dataset_dict_sample['x_hydro_model'],
            hydro_params,
            waterloss_params,
            dataset_dict_sample['ai_batch'],
            dataset_dict_sample['ac_batch'],
            dataset_dict_sample['idx_matrix'],
            self.config,
            static_idx=self.config['static_index'],
            warm_up=self.config['warm_up'],
            routing=self.config['routing_hydro_model'],
            conv_params_hydro=rout_params
        )
        
        return flow_out
