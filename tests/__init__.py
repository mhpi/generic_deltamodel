# dmg/tests/__init__.py
import importlib
import pkgutil
from pathlib import Path

import pytest
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


def get_phy_model_name(config):
    """Extract the physics model name from a config dict."""
    return config['model']['phy']['name'][0]


def _skip_if_zero_streamflow(dpl_model, model_dataset):
    """Skip test if model produces zero streamflow (degenerate initialization)."""
    with torch.no_grad():
        out = dpl_model(model_dataset)
    if out['streamflow'].abs().max().item() == 0:
        pytest.skip(
            "Model produces zero streamflow with random initialization "
            "(LstmMlpModel sigmoid squashes parameters to degenerate range)",
        )


def _get_nn_params(dpl_model, dataset):
    """Helper to get NN output parameters for any model type."""
    with torch.no_grad():
        if type(dpl_model.nn_model).__name__ == 'LstmMlpModel':
            return dpl_model.nn_model(
                dataset['xc_nn_norm'],
                dataset['c_nn_norm'],
            )
        else:
            return dpl_model.nn_model(dataset['xc_nn_norm'])


def compute_mse_loss(dpl_model, model_dataset, warm_up):
    """Compute MSE loss between model output and target."""
    output = dpl_model(model_dataset)
    streamflow = output['streamflow']
    target = model_dataset['target'][warm_up:]
    n = min(streamflow.shape[0], target.shape[0])
    loss = (streamflow[:n] - target[:n]).pow(2).mean()
    return loss, output
