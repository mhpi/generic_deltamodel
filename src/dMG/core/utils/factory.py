import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from hydroDL2 import load_model as load_from_hydrodl
from numpy.typing import NDArray

from dMG.core.data.loaders.base import BaseLoader
from dMG.core.data.samplers.base import BaseSampler
from dMG.trainers.base import BaseTrainer

from . import camel_to_snake

sys.path.append('../dMG/')  # for tutorials

import numpy as np

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
    base_class: type,
) -> type:
    """
    Generalized loader function to dynamically import components.

    Parameters
    ----------
    class_name
        The name of the class to load.
    directory
        The subdirectory where the module is located.
    base_class
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
    except FileNotFoundError as e:
        raise ImportError(f"Component '{class_name}' not found in '{source}'.") from e

    # Confirm class is in the module and matches the base class.
    if hasattr(module, class_name):
        class_obj = getattr(module, class_name)
        if isinstance(class_obj, type) and issubclass(class_obj, base_class):
            return class_obj

    raise ImportError(f"Class '{class_name}' not found in module '{os.path.relpath(source)}' or does not subclass '{base_class.__name__}'.")


def import_phy_model(model: str, ver_name: str = None) -> type:
    """Loads a physical model, either from HydroDL2 (hydrology) or locally."""
    try:
        return load_from_hydrodl(model, ver_name)
    except ImportError:
        return load_component(
            model,  # Pass model as name directly
            phy_model_dir,
            torch.nn.Module,
        )


def import_data_loader(name: str) -> type:
    """Loads a data loader dynamically."""
    return load_component(
        name,
        loader_dir,
        BaseLoader,
    )


def import_data_sampler(name: str) -> type:
    """Loads a data sampler dynamically."""
    return load_component(
        name,
        sampler_dir,
        BaseSampler,
    )


def import_trainer(name: str) -> type:
    """Loads a trainer dynamically."""
    return load_component(
        name,
        trainer_dir,
        BaseTrainer,
    )


def load_loss_func(
    y_obs: NDArray[np.float32],
    config: dict[str, Any],
    name: Optional[str] = None,
    device: Optional[str] = 'cpu',
) -> torch.nn.Module:
    """Dynamically load and initialize a loss function module by name.

    Parameters
    ----------
    y_obs
        The observed data array needed for some loss function initializations.
    config
        The configuration dictionary, including loss function specifications.
    name
        The name of the loss function to load. The default is None, using the
        spec named in config.
    device
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
        return cls(config, device, y_obs=y_obs)
    except (ValueError, KeyError) as e:
        raise Exception(f"'{name}': {e}") from e


s
