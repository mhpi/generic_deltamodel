import logging
import os
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from dmg.core.utils import initialize_config

log = logging.getLogger(__name__)

__all__ = [
    'load_config',
    'take_data_sample',
]


def load_config(path: str) -> dict[str, Any]:
    """Parse and initialize configuration settings from yaml with Hydra.
    
    This loader is capable of handling config files in nonlinear directory
    structures.

    Parameters
    ----------
    config_path
        Path to the configuration file.
    
    Returns
    -------
    dict
        Formatted configuration settings.
    """
    # Get the parent dir of the config file.
    path_no_ext = os.path.splitext(path)[0]
    parent_path, config_name = os.path.split(path_no_ext)
    parent_path = os.path.relpath(parent_path)

    with hydra.initialize(config_path=parent_path, version_base='1.3'):
        config = hydra.compose(config_name=config_name)

    # Convert the OmegaConf object to a dict.
    config = OmegaConf.to_container(config, resolve=True)

    # Convert date ranges / set device and dtype / create output dirs.
    config = initialize_config(config)

    return config


def take_data_sample(
    config: dict,
    dataset_dict: dict[str, torch.Tensor],
    days: int = 730,
    basins: int = 100,
) -> dict[str, torch.Tensor]:
    """Take sample of data.
    
    Parameters
    ----------
    config
        Configuration settings.
    dataset_dict
        Dictionary containing dataset tensors.
    days
        Number of days to sample.
    basins
        Number of basins to sample.
    
    Returns
    -------
    dict
        Dictionary containing sampled dataset tensors.
    """
    dataset_sample = {}

    for key, value in dataset_dict.items():
        if value.ndim == 3:
            # Determine warm-up period based on the key
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['delta_model']['phy_model']['warm_up']

            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[warm_up:days, :basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        elif value.ndim == 2:
            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[:basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Adjust the 'target' tensor based on the configuration
    if ('HBV1_1p' in config['delta_model']['phy_model']['model'] and
        config['delta_model']['phy_model']['use_warmup_mode'] and
        config['multimodel_type'] == 'none'):
        pass  # Keep 'warmup' days for dHBV1.1p
    else:
        warm_up = config['delta_model']['phy_model']['warm_up']
        dataset_sample['target'] = dataset_sample['target'][warm_up:days, :basins]

    return dataset_sample
