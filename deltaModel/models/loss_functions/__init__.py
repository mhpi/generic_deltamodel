# A loss function initializer that dynamically loads loss function modules.
import importlib.util
import os
import re
import sys

import numpy as np

sys.path.append('../dMG/') # for tutorials



def snake_to_camel(snake_str):
    """
    Convert snake strings (underscore word separation, lower case) to
    Camel-case strings (no word separation, capitalized first letter of a word).
    """
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def camel_to_snake(camel_str):
    """
    Find all instances where a lowercase letter is followed by an uppercase letter
    and replace them with the lowercase letter, an underscore, and the uppercase
    letter converted to lowercase.
    """
    snake_str = re.sub(r'(?<!^)(?=[A-Z])', '_', camel_str).lower()
    return snake_str


def get_loss_func(config, obs):
    """Dynamically load a loss function module from the specified file.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
        obs : np.ndarray
            The observed data.
    """
    loss_func = config['loss_function']['model']
    file_name = camel_to_snake(loss_func)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Debugging Note: The directory './deltaModel/models' should be included
    # to find the correct loss function.
    # file_path = os.path.join(source_dir, f"{file_name}.py")
    file_path = os.path.join(source_dir, "deltaModel", "models", f"{file_name}.py")

   # Load the module dynamically.
    try:
        spec = importlib.util.spec_from_file_location(loss_func, os.path.abspath(file_path))
        if spec and spec.loader:
            # module = spec.loader.load_module()
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            raise ImportError(f"Module {file_name} could not be loaded from {file_path}.")
    except (FileNotFoundError, ImportError) as e:
        raise ImportError(f"Error loading module '{file_name}': {e}")

    # # Fetch the loss fn class.
    # loss_function_default = getattr(module, config['loss_function']['model'])

     # Fetch the loss function class and initialize it.
    try:
        loss_class = getattr(module, loss_func)
    except AttributeError:
        raise AttributeError(f"Class '{loss_func}' not found in module '{file_name}'.")

    # Initialize object.
    # NOTE: Any loss functions with specific settings should have them set here.
    if loss_func in ['NseLossBatchFlow','NseSqrtLossBatchFlow']:
        std_obs_flow = np.nanstd(obs[:, :, config['train']['target'].index('00060_Mean')], axis=0)
        loss_obj = loss_function_default(std_obs_flow)
    else:
        loss_obj = loss_function_default()
    
    return loss_obj.to(config['device'])

