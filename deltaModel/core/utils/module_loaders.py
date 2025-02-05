import importlib.util
import os
from pathlib import Path
from typing import Type

import torch
from core.data.data_loaders.base import BaseDataLoader
from core.data.data_samplers.base import BaseDataSampler
from hydroDL2 import load_model as load_from_hydrodl
from trainers.base import BaseTrainer


def get_dir(dir_name: str) -> Path:
    """Get the path for the given directory name."""
    dir = Path('../../' + dir_name)
    if not os.path.exists(dir):
        dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dir = dir.parent.parent / dir_name
    return dir


def get_phy_model(model: str, ver_name: str = None) -> Type:
    """
    Load a model from the models directory.
    
    Each model file in `models/` directory should contain one or more model classes.
    
    Parameters
    ----------
    model : str
        The model name.
    ver_name : str, optional
        The version name (class) of the model to load within the model file.
    
    Returns
    -------
    Type
        The uninstantiated model class.
    """
    try:
        # Attempt to load from HydroDL2.
        return load_from_hydrodl(model, ver_name)
    except:
        # Otherwise, load from local models directory.
        parent_dir = get_dir('models')
        model_dir = model.split('_')[0].lower()
        model_subpath = os.path.join(model_dir, f'{model.lower()}.py')
        
        source = os.path.join(parent_dir, model_subpath)
        
        try:
            # Load the model dynamically as a module.
            spec = importlib.util.spec_from_file_location(model, source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except FileNotFoundError:
            raise ImportError(f"Model '{model}' not found.")
        
        if ver_name:
            # Retrieve by version name, if specified.
            cls = getattr(module, ver_name)
            # Ensure the class is defined in this module and inherits from BaseModel
            if not isinstance(cls, type) or cls.__module__ != module.__name__:
                raise ImportError(f"Class '{ver_name}' is not a valid instance of '{model}'.")
            if not issubclass(cls, torch.nn.Module):
                raise ImportError(f"Class '{ver_name}' does not inherit from torch.nn.Module.")
            return cls
        else:
            # Search for a class that matches the expected naming convention and inheritance.
            target_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and (attr.__module__ == module.__name__):
                    if issubclass(attr, torch.nn.Module):
                        # TODO: This needs to be more robust.
                        target_class = attr
                        break
            if target_class is None:
                raise ImportError(f"No valid Model class found in '{model}'.")
            
            return target_class
        

def get_data_loader(loader_name: str) -> Type:
    """
    Load a data loader from the data_loaders directory.
    
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
    source = os.path.join(parent_dir, loader_subpath)
    
    try:
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(loader_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Data loader '{loader_name}' not found.")
    
    # Search for a class that matches the expected naming convention and inheritance
    target_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and (attr.__module__ == module.__name__):
            if attr_name.endswith('DataLoader') and issubclass(attr, BaseDataLoader):
                target_class = attr
                break
    if target_class is None:
        raise ImportError(f"No valid DataLoader class found in '{loader_name}'.")
    
    return target_class


def get_data_sampler(sampler_name: str) -> Type:
    """
    Load a data sampler from the data_samplers directory.
    
    Parameters
    ----------
    sampler_name : str
        The name of the data sampler.
    
    Returns
    -------
    Type
        The uninstantiated data sampler class.
    """
    # Path to the core/data/data_samplers directory
    parent_dir = get_dir('core/data/data_samplers')
    # Construct the file path for the sampler
    sampler_subpath = f"{sampler_name}.py"
    source = os.path.join(parent_dir, sampler_subpath)
    
    try:
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(sampler_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Data sampler '{sampler_name}' not found.")
    
    # Search for a class that matches the expected naming convention and inheritance
    target_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and (attr.__module__ == module.__name__):
            if attr_name.endswith('DataSampler') and issubclass(attr, BaseDataSampler):
                target_class = attr
                break
    if target_class is None:
        raise ImportError(f"No valid DataSampler class found in '{sampler_name}'.")
    
    return target_class


def get_trainer(trainer_name: str) -> Type:
    """
    Load a trainer from the trainers directory.
    
    Parameters
    ----------
    trainer_name : str
        The name of the trainer.
    
    Returns
    -------
    Type
        The uninstantiated trainer class.
    """
    # Path to the core/trainers directory
    parent_dir = get_dir('trainers')
    # Construct the file path for the trainer
    trainer_subpath = f"{trainer_name}.py"
    source = os.path.join(parent_dir, trainer_subpath)
    
    try:
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location(trainer_name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Trainer '{trainer_name}' not found.")
    
    # Search for a class that matches the expected naming convention and inheritance
    target_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and (attr.__module__ == module.__name__):
            if attr_name.endswith('Trainer') and issubclass(attr, BaseTrainer):
                target_class = attr
                break
    if target_class is None:
        raise ImportError(f"No valid Trainer class found in '{trainer_name}'.")
    
    return target_class
