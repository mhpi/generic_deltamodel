import logging
import os
import sys

import hydra
# sys.path.append('../../deltaModel') # Add the root directory of dMG to the path
from core.utils import initialize_config
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)



def load_config(path: str) -> DictConfig:
    """Initialize Hydra and parse model configuration yaml(s) into config dict.
    
    # This loader is capable of handling config files in nonlinear directory
    # structures.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    # Get the parent directory of the configuration file.
    path_no_ext = os.path.splitext(path)[0]
    parent_path, config_name = os.path.split(path_no_ext)
    parent_path = os.path.relpath(parent_path)


    with hydra.initialize(config_path='conf/', version_base='1.3'):
        config = hydra.compose(config_name='dhbv_config')
   
    # Convert the OmegaConf object to a dictionary.
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Convert date ranges, set device and dtype, and create output directories.
    initialize_config(config_dict)

    return config_dict
