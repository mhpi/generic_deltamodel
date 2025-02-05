import importlib.util
import os
import sys
from ast import Dict
from typing import Any, Dict, Optional

import numpy as np
import torch
from core.utils import camel_to_snake
from numpy.typing import NDArray

sys.path.append('../deltaModel/')  # for tutorials


def get_loss_func(
    obs: NDArray[np.float32],
    config: Dict[str, Any],
    device: Optional[str] = 'cpu',
) -> torch.nn.Module:
    """Dynamically load and initialize a loss function module by name.
    
    Currently only supports loss functions in the 'loss_functions' directory.

    Parameters
    ----------
    obs : np.ndarray
        The observed data array needed for some loss function initializations.
    config : dict
        The configuration dictionary, including loss function specifications.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.
    
    Returns
    -------
    torch.nn.Module
        The initialized loss function object.
    """
    # Extract loss function name and file path.
    loss_name = config['model']
    file_name = camel_to_snake(loss_name)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    # NOTE: change for debugging:
    # file_path = os.path.join(source_dir, "deltaModel", "models", f"{file_name}.py")
    file_path = os.path.join(source_dir, f"{file_name}.py")

    # Load the specified loss function module dynamically.
    try:
        spec = importlib.util.spec_from_file_location(loss_name, os.path.abspath(file_path))
        if not spec or not spec.loader:
            raise ImportError(f"Module {file_name} could not be loaded from {file_path}.")
    
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (FileNotFoundError, ImportError) as e:
        raise ImportError(f"Error loading module '{file_name}': {e}")

    # Fetch the loss function class.
    try:
        loss_func_cls = getattr(module, loss_name)
    except AttributeError:
        raise AttributeError(f"Class '{loss_name}' not found in module '{file_name}'.")

    # Initialize (NOTE: any loss function specific settings should be set here).
    try:
        loss_obj = loss_func_cls(obs, config, device)
    except Exception as e:
        raise ValueError(f"Error initializing loss function '{loss_name}': {e}")

    return loss_obj.to(device)
