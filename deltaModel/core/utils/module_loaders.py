import importlib.util
import os
from json import load
from pathlib import Path
from typing import Dict, List, Type

from hydroDL2 import load_model as load_hydro_model
from torch.nn import Module


def get_dir(dir_name: str) -> Path:
    """Get the path for the given directory name."""
    dir = Path('../../' + dir_name)
    if not os.path.exists(dir):
        dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dir = dir.parent.parent / dir_name
    return dir


def load_model(model: str, ver_name: str = None) -> Module:
    """Load a model from the models directory.

    Each model file in `models/` directory should only contain one model class.

    Parameters
    ----------
    model : str
        The model name.
    ver_name : str, optional
        The version name (class) of the model to load within the model file.
    
    Returns
    -------
    Module
        The uninstantiated model.
    """
    if model in ['HBV', 'HBV_1_1p', 'PRMS', 'SACSMA_with_snow']:
        return load_hydro_model(model, ver_name)
    
    # Path to the models directory
    parent_dir = get_dir('models')

    # Construct file path
    model_dir = model.split('_')[0].lower()
    model_subpath = os.path.join(model_dir, f'{model.lower()}.py')
    
    # Path to the module file in the models directory
    source = os.path.join(parent_dir, model_subpath)
    
    # Load the model dynamically as a module.
    try:
        spec = importlib.util.spec_from_file_location(model, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Model '{model}' not found.")
    
    # Retrieve the version name if specified, otherwise get the first class in the module
    if ver_name:
        cls = getattr(module, ver_name)
    else:
        # Find the first class in the module (this may not always be accurate)
        classes = [
            attr for attr in dir(module)
            if isinstance(getattr(module, attr), type) and attr != 'Any'
        ]
        if not classes:
            raise ImportError(f"Model version '{model}' not found.")
        cls = getattr(module, classes[-1])
    
    return cls



import importlib.util
import os
from typing import Type


def load_data_loader(loader_name: str) -> Type:
    """Load a data loader from the data_loaders directory.

    Parameters
    ----------
    loader_name : str
        The name of the data loader.

    Returns
    -------
    Type
        The uninstantiated data loader class.
    """
    # Path to the core/data/data_loaders directory
    parent_dir = get_dir('core/data/data_loaders')

    # Construct the file path for the loader
    loader_subpath = f"{loader_name}.py"

    # Path to the module file in the data_loaders directory
    source = os.path.join(parent_dir, loader_subpath)

    # Load the data loader dynamically as a module
    try:
        spec = importlib.util.spec_from_file_location(loader_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Data loader '{loader_name}' not found.")

    # Dynamically load the first class found in the module
    classes = [
        attr for attr in dir(module)
        if isinstance(getattr(module, attr), type) and attr != 'Any'
    ]

    if not classes:
        raise ImportError(f"No valid classes found in loader '{loader_name}'.")

    # Ensure the right class is chosen
    # Optionally, you could inspect for class names that match 'HydroDataLoader' or similar
    cls = getattr(module, classes[-1])

    # Check if the class is a subclass of BaseDataLoader to confirm it's the right type

    return cls


def load_data_sampler(sampler_name: str) -> Type:
    """Load a data loader from the data_loaders directory.

    Parameters
    ----------
    loader_name : str
        The name of the data loader.

    Returns
    -------
    Type
        The uninstantiated data loader class.
    """
    # Path to the core/data/data_loaders directory
    parent_dir = get_dir('core/data/data_samplers')

    # Construct the file path for the loader
    loader_subpath = f"{sampler_name}.py"

    # Path to the module file in the data_loaders directory
    source = os.path.join(parent_dir, loader_subpath)

    # Load the data loader dynamically as a module
    try:
        spec = importlib.util.spec_from_file_location(sampler_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Data loader '{sampler_name}' not found.")

    # Dynamically load the first class found in the module
    classes = [
        attr for attr in dir(module)
        if isinstance(getattr(module, attr), type) and attr != 'Any'
    ]

    if not classes:
        raise ImportError(f"No valid classes found in loader '{sampler_name}'.")

    # Ensure the right class is chosen
    # Optionally, you could inspect for class names that match 'HydroDataLoader' or similar
    cls = getattr(module, classes[-1])

    # Check if the class is a subclass of BaseDataLoader to confirm it's the right type

    return cls


def load_trainer(trainer_name: str) -> Type:
    """Load a data loader from the data_loaders directory.

    Parameters
    ----------
    loader_name : str
        The name of the data loader.

    Returns
    -------
    Type
        The uninstantiated data loader class.
    """
    # Path to the core/data/data_loaders directory
    parent_dir = get_dir('trainers')

    # Construct the file path for the loader
    loader_subpath = f"{trainer_name}.py"

    # Path to the module file in the data_loaders directory
    source = os.path.join(parent_dir, loader_subpath)

    # Load the data loader dynamically as a module
    try:
        spec = importlib.util.spec_from_file_location(trainer_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Data loader '{trainer_name}' not found.")

    # Dynamically load the first class found in the module
    classes = [
        attr for attr in dir(module)
        if isinstance(getattr(module, attr), type) and attr != 'Any'
    ]

    if not classes:
        raise ImportError(f"No valid classes found in loader '{trainer_name}'.")

    # Ensure the right class is chosen
    # Optionally, you could inspect for class names that match 'HydroDataLoader' or similar
    cls = getattr(module, classes[-1])

    # Check if the class is a subclass of BaseDataLoader to confirm it's the right type

    return cls
