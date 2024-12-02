import logging
import math
from abc import ABC
from typing import Callable, Dict

import torch
import torch.nn as nn
from conf.config_old import InitalizationEnum
from models.neural_networks.lstm_models import CudnnLstmModel
from models.neural_networks.mlp_models import MLPmul

log = logging.getLogger(__name__)


class Initialization(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.config = kwargs["config"]

    def kaiming_normal_initializer(self, x) -> None:
        nn.init.kaiming_normal_(x, mode=self.config.fan, nonlinearity="sigmoid")

    def xavier_normal_initializer(self, x) -> None:
        nn.init.xavier_normal_(x, gain=self.config.gain)

    @staticmethod
    def sparse_initializer(x) -> None:
        nn.init.sparse_(x, sparsity=0.5)

    @staticmethod
    def uniform_initializer(x) -> None:
        # hardcoding hidden size for now
        stdv = 1.0 / math.sqrt(6)
        nn.init.uniform_(x, a=-stdv, b=stdv)

    def forward(self, *args, **kwargs) -> Callable[[torch.Tensor], None]:
        init = self.config.initialization
        log.debug(f"Initializing weight states using the {init} function")
        if init == InitalizationEnum.kaiming_normal:
            func = self.kaiming_normal_initializer
        elif init == InitalizationEnum.kaiming_uniform:
            func = nn.init.kaiming_uniform_
        elif init == InitalizationEnum.orthogonal:
            func = nn.init.orthogonal_
        elif init == InitalizationEnum.sparse:
            func = self.sparse_initializer
        elif init == InitalizationEnum.trunc_normal:
            func = nn.init.trunc_normal_
        elif init == InitalizationEnum.xavier_normal:
            func = self.xavier_normal_initializer
        elif init == InitalizationEnum.xavier_uniform:
            func = nn.init.xavier_uniform_
        else:
            log.info("Defaulting to a uniform initialization")
            func = self.uniform_initializer
        return func


class NeuralNetwork(ABC, torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.config = kwargs["config"]
        self.Initialization = Initialization(config=self.config)

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("The forward function must be implemented")
    

def init_nn_model(phy_model, config):
    """Initialize the pNN model.
    
    Parameters
    ----------
    phy_model : torch.nn.Module
        The physics model.
    config : dict
        The configuration dictionary.
    
    Returns
    -------
    torch.nn.Module
        The initialized neural network.
    """
    n_forc = len(config['nn_model']['forcings'])
    n_attr = len(config['nn_model']['attributes'])
    n_model_params = len(phy_model.parameter_bounds)
    n_rout_params = len(phy_model.conv_routing_hydro_model_bound)
    
    nx = n_forc + n_attr
    ny = config['nmul'] * n_model_params

    if config['phy_model']['routing'] == True:
        ny += n_rout_params
    
    if config['nn_model']['model'] == 'LSTM':
        nn_model = CudnnLstmModel(
            nx=nx,
            ny=ny,
            hiddenSize=config['nn_model']['hidden_size'],
            dr=config['nn_model']['dropout']
        )
    elif config['nn_model']['model'] == 'MLP':
        nn_model = MLPmul(
            config,
            nx=nx,
            ny=ny
        )
    else:
        raise ValueError(config['nn_model'], " not supported.")
    
    return nn_model
    
