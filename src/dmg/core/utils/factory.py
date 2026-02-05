import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from numpy.typing import NDArray

from dmg.core.utils.utils import camel_to_snake

sys.path.append('../dmg/')  # for tutorials

import numpy as np

log = logging.getLogger(__name__)

# ------------------------------------------#
# If directory structure changes, update these module paths.
# NOTE: potentially move these to a framework config for easier access.
loader_dir = 'core/data/loaders'
sampler_dir = 'core/data/samplers'
trainer_dir = 'trainers'
loss_func_dir = 'models/criterion'
phy_model_dir = 'models/phy_models'
nn_model_dir = 'models/neural_networks'
# ------------------------------------------#


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
        sys.modules[class_name] = module
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"Component '{class_name}' not found in '{source}'.") from e

    # Confirm class is in the module and matches the base class.
    if hasattr(module, class_name):
        class_obj = getattr(module, class_name)
        if isinstance(class_obj, type) and issubclass(class_obj, base_class):
            return class_obj

    raise ImportError(
        f"Class '{class_name}' not found in module '{os.path.relpath(source)}' or does not subclass '{base_class.__name__}'.",
    )


def import_phy_model(model: str, ver_name: str = None) -> type:
    """Loads a physical model, either from HydroDL2 (hydrology) or locally."""
    try:
        import hydrodl2

        all_models = [
            m for names in hydrodl2.available_models().values() for m in names
        ]
        if model.lower() in all_models:
            return hydrodl2.load_model(model, ver_name)
        else:
            return load_component(
                model,
                phy_model_dir,
                torch.nn.Module,
            )
    except ImportError:
        log.warning("Package 'HydroDL2' not loaded. Continuing without it.")
        return load_component(
            model,  # Pass model as name directly
            phy_model_dir,
            torch.nn.Module,
        )


def import_data_loader(name: str) -> type:
    """Loads a data loader dynamically."""
    from dmg.core.data.loaders.base import BaseLoader

    return load_component(
        name,
        loader_dir,
        BaseLoader,
    )


def import_data_sampler(name: str) -> type:
    """Loads a data sampler dynamically."""
    from dmg.core.data.samplers.base import BaseSampler

    return load_component(
        name,
        sampler_dir,
        BaseSampler,
    )


def import_trainer(name: str) -> type:
    """Loads a trainer dynamically."""
    from dmg.trainers.base import BaseTrainer

    return load_component(
        name,
        trainer_dir,
        BaseTrainer,
    )


def load_criterion(
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
        name = config['name']

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


def load_nn_model(
    config: dict[str, dict[str, Any]],
    phy_model: Optional[torch.nn.Module] = None,
    ensemble_list: Optional[list] = None,
    device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Initialize a neural network.

    Parameters
    ----------
    config
        Configuration settings for the model.
    phy_model
        Physics model to format NN output.
    ensemble_list
        List of models to ensemble. Default is None. This will result in a
        weighting NN being initialized.
    device
        The device to run the model on. Default is None.

    Returns
    -------
    torch.nn.Module
        An initialized neural network.
    """
    if not device:
        device = config.get('device', 'cpu')
    if isinstance(device, torch.device):
        device = str(device)

    # Number of inputs 'x' and outputs 'y' for the nn.
    if ensemble_list:
        n_forcings = len(config['forcings'])
        n_attributes = len(config['attributes'])
        ny = len(ensemble_list)

        hidden_size = config['hidden_size']
        dr = config['dropout']
        name = config['model']
    else:
        n_forcings = len(config['nn']['forcings'])
        n_attributes = len(config['nn']['attributes'])
        name = config['nn']['name']

        if phy_model:
            ny = phy_model.learnable_param_count
        elif 'out_size' in config['nn']:
            ny = config['nn']['out_size']
        else:
            raise ValueError(
                "Output size 'out_size' must be specified in the config or"
                " physics model must be provided.",
            )

        if name not in ['LstmMlpModel']:
            hidden_size = config['nn']['hidden_size']
            dr = config['nn']['dropout']

    nx = n_forcings + n_attributes

    # Dynamically retrieve the model
    cls = load_component(
        name,
        nn_model_dir,
        torch.nn.Module,
    )

    # Initialize the model with the appropriate parameters
    if name in ['CudnnLstmModel', 'LstmModel']:
        model = cls(
            nx=nx,
            ny=ny,
            hidden_size=hidden_size,
            dr=dr,
            cache_states=config['nn']['cache_states'],
        )
    elif name in ['MLP']:
        model = cls(
            config,
            nx=nx,
            ny=ny,
        )
    elif name in ['LstmMlpModel']:
        if phy_model:
            ny1 = phy_model.learnable_param_count1
            ny2 = phy_model.learnable_param_count2
        elif ('out_size1' in config['nn']) and ('out_size2' in config['nn']):
            ny1 = config['nn']['out_size1']
            ny2 = config['nn']['out_size2']
        else:
            raise ValueError(
                "Output sizes 'out_size1' and 'out_size2' must be"
                " specified in the config or physics model must be provided.",
            )

        model = cls(
            nx1=nx,
            ny1=ny1,
            hiddeninv1=config['nn']['lstm_hidden_size'],
            nx2=n_attributes,
            ny2=ny2,
            hiddeninv2=config['nn']['mlp_hidden_size'],
            dr1=config['nn']['lstm_dropout'],
            dr2=config['nn']['mlp_dropout'],
            cache_states=config['nn']['cache_states'],
            device=device,
        )
    else:
        raise ValueError(f"Model {name} is not supported.")

    return model.to(device)
