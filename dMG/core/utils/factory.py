import importlib.util
import os
import sys
from ast import Dict
from pathlib import Path
from typing import Any, Dict, Optional, Type

import torch
from hydroDL2 import load_model as load_from_hydrodl
from numpy.typing import NDArray

from core.data.loaders.base import BaseLoader
from core.data.samplers.base import BaseSampler
from trainers.base import BaseTrainer

sys.path.append('../dMG/')  # for tutorials

import numpy as np
import torch
from numpy.typing import NDArray

from core.utils import camel_to_snake

#------------------------------------------#
# If directory structure changes, update these module paths.
# NOTE: potentially move these to a framework config for easier access.
loader_dir = 'core/data/loaders'
sampler_dir = 'core/data/samplers'
trainer_dir = 'trainers'
loss_func_dir = 'models/criterion'
phy_model_dir = 'models/physics_models'
nn_model_dir = 'models/neural_networks'
#------------------------------------------#


def get_dir(dir_name: str) -> Path:
    """Get the path for the given directory name."""
    dir = Path('../../' + dir_name)
    if not os.path.exists(dir):
        dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dir = dir.parent.parent / dir_name
    return dir


def load_component(
    class_name: str,
    directory: str,
    base_class: Type,
) -> Type:
    """
    Generalized loader function to dynamically import components.

    Parameters
    ----------
    class_name : str
        The name of the class to load.
    directory : str
        The subdirectory where the module is located.
    base_class : Type
        The expected base class type (e.g., torch.nn.Module).

    Returns
    -------
    Type
        The uninstantiated class.
    """
    # Remove the 'Model' suffix from class name if present
    if class_name.endswith('Model'):
        class_name_without_model = class_name[:-5]
    else:
        class_name_without_model = class_name

    # Convert from camel case to snake case for file name
    name_lower = camel_to_snake(class_name_without_model)

    parent_dir = get_dir(directory)
    source = os.path.join(parent_dir, f"{name_lower}.py")

    try:
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(class_name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_name] = module  # Add to sys.modules
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Component '{class_name}' not found in '{source}'.")

    # Confirm class is in the module and matches the base class.
    if hasattr(module, class_name):
        class_obj = getattr(module, class_name)
        if isinstance(class_obj, type) and issubclass(class_obj, base_class):
            return class_obj

    raise ImportError(f"Class '{class_name}' not found in module '{os.path.relpath(source)}' or does not subclass '{base_class.__name__}'.")


def import_phy_model(model: str, ver_name: str = None) -> Type:
    """Loads a physical model, either from HydroDL2 (hydrology) or locally."""
    try:
        return load_from_hydrodl(model, ver_name)
    except:
        return load_component(
            model,  # Pass model as name directly
            phy_model_dir,
            torch.nn.Module,
        )


def import_data_loader(name: str) -> Type:
    """Loads a data loader dynamically."""
    return load_component(
        name,
        loader_dir,
        BaseLoader,
    )


def import_data_sampler(name: str) -> Type:
    """Loads a data sampler dynamically."""
    return load_component(
        name,
        sampler_dir,
        BaseSampler,
    )


def import_trainer(name: str) -> Type:
    """Loads a trainer dynamically."""
    return load_component(
        name,
        trainer_dir,
        BaseTrainer,
    )


def load_loss_func(
    obs: NDArray[np.float32],
    config: Dict[str, Any],
    name: Optional[str] = None,
    device: Optional[str] = 'cpu',
) -> torch.nn.Module:
    """Dynamically load and initialize a loss function module by name.

    Parameters
    ----------
    obs : NDArray[np.float32]
        The observed data array needed for some loss function initializations.
    config : dict
        The configuration dictionary, including loss function specifications.
    name : str, optional
        The name of the loss function to load. The default is None, using the
        spec named in config.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.

    Returns
    -------
    torch.nn.Module
        The initialized loss function object.
    """
    if not name:
        name = config['model']
    
    # Load the loss function dynamically using the factory.
    cls = load_component(
        name,
        loss_func_dir,
        torch.nn.Module,
    )

    # Initialize (NOTE: any loss function specific settings should be set here).
    try:
        return cls(obs, config, device)
    except Exception as e:
        raise ValueError(f"Error initializing loss function '{name}': {e}")


def load_nn_model(
    phy_model: torch.nn.Module,
    config: Dict[str, Dict[str, Any]],
    ensemble_list: Optional[list] = None,
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
    ensemble_list : list, optional
        List of models to ensemble. Default is None. This will result in a
        weighting nn being initialized.
    device : str, optional
        The device to run the model on. Default is None.

    Returns
    -------
    torch.nn.Module
        An initialized neural network.
    """
    if not device:
        device = config.get('device', 'cpu')

    # Number of inputs 'x' and outputs 'y' for the nn.
    if ensemble_list:
        n_forcings = len(config['forcings'])
        n_attributes = len(config['attributes'])
        ny = len(ensemble_list)

        hidden_size = config['hidden_size']
        dr = config['dropout']
        name = config['model']
    else:
        n_forcings = len(config['nn_model']['forcings'])
        n_attributes = len(config['nn_model']['attributes'])
        n_phy_params = phy_model.learnable_param_count
        ny = n_phy_params

        name = config['nn_model']['model']

        if name not in ['LstmMlpModel']:
            hidden_size = config['nn_model']['hidden_size']
            dr = config['nn_model']['dropout']

    nx = n_forcings + n_attributes
    
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
            hiddenSize=hidden_size,
            dr=dr,
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
            ny1=phy_model.learnable_param_count1,
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
