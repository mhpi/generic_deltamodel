import logging
from typing import Any, Dict, Optional

import torch

from ...core.utils.factory import load_component

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
    phy_model : torch.nn.Module
        The physics model.
    config : dict
        The configuration dictionary.

    Returns
    -------
    torch.nn.Module
        An initialized neural network.
    """
    if not device:
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
            hiddenSize=config['nn_model']['hidden_size'],
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
