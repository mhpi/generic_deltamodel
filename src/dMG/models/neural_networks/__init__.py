import logging
from typing import Any, Dict, Optional

import torch

from dMG.core.utils.factory import load_component

log = logging.getLogger(__name__)


#------------------------------------------#
# If directory structure changes, update these module paths.
# NOTE: potentially move these to a framework config for easier access.
nn_model_dir = 'models/neural_networks'
#------------------------------------------#


def load_nn_model(
    phy_model: torch.nn.Module,
    config: Dict[str, Dict[str, Any]],
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Initialize a neural network.

    Parameters
    ----------
    phy_model
        The physics model.
    config
        The configuration dictionary.
    device
        The device to use (e.g., 'cpu' or 'cuda'). If None, the device is
        determined from the config.

    Returns
    -------
    torch.nn.Module
        An initialized neural network.
    """
    if device is None:
        device = config.get('device', 'cpu')
    
    n_forcings = len(config['nn_model']['forcings'])
    n_attributes = len(config['nn_model']['attributes'])
    n_phy_params = phy_model.learnable_param_count

    # Number of inputs 'x' and outputs 'y' for nn.
    nx = n_forcings + n_attributes
    ny = n_phy_params

    name = config['nn_model']['model']
    
    # Dynamically retrieve the model
    cls = load_component(
        name, 
        nn_model_dir,
        torch.nn.Module
    )

    # Initialize the model with the appropriate parameters
    if name in ['CudnnLstmModel']:
        model = cls(
            nx=nx,
            ny=ny,
            hidden_size=config['nn_model']['hidden_size'],
            dr=config['nn_model']['dropout'],
        )
    elif name in ['MLP']:
        model = cls(
            config,
            nx=nx,
            ny=ny,
        )
    elif name in ['LstmMlpModel']:
        model = cls(
            nx1=nx,
            ny1=config['phy_model']['nmul'] * n_phy_params,
            hiddeninv1=config['nn_model']['lstm_hidden_size'],
            nx2=n_attributes,
            ny2=phy_model.learnable_param_count2,
            hiddeninv2=config['nn_model']['mlp_hidden_size'],
            dr1=config['nn_model']['lstm_dropout'],
            dr2=config['nn_model']['mlp_dropout'],
        )
    else:
        raise ValueError(f"Model {name} is not supported.")
    
    return model.to(device)
