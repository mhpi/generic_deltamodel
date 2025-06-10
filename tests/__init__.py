import importlib
import pkgutil
from pathlib import Path

import torch


def get_available_classes(
    path: Path,
    pkg_path: str,
    base_class: type,
) -> list[type]:
    """Dynamically import all modules from the specified directory.
        
    Parameters
    ----------
    path
        Path to the directory containing the modules.
    pkg_path
        Package-level path to the directory containing the modules.
    base_class
        The base class the modules should inherit from. However, only used here
        to exclude the base class itself (base.py) from the list of modules.

    Returns
    -------
    list[type]
        List of modules available at the specified path.
    """
    classes = []
    for _, module_name, _ in pkgutil.iter_modules([str(path)]):
        module = importlib.import_module(f"{pkg_path}.{module_name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Check if the attribute is a subclass of torch.nn.Module and not abstract
            if (
                isinstance(attr, type)
                and attr is not base_class
                and issubclass(attr, torch.nn.Module)
                and attr.__name__ != "Module"
            ):
                classes.append(attr)
    return classes
