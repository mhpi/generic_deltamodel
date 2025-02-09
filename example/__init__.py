import logging
import os
from typing import Any, Dict

import hydra
from core.utils import initialize_config
from omegaconf import OmegaConf

log = logging.getLogger(__name__)



def load_config(path: str) -> Dict[str, Any]:
    """Parse and initialize configuration settings from yaml with Hydra.
    
    This loader is capable of handling config files in nonlinear directory
    structures.

    Parameters
    ----------
    config_path : str
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
