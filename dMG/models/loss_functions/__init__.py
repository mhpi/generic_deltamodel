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


def get_loss_function(config, obs):
    """
    Dynamically load the loss fn module from the specified file.
    """
    loss_function = config['loss_function']['model']
    file_name = camel_to_snake(loss_function)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    
    ## NOTE: for debugging `./dMG/models` must be specified. Can't figure out why.
    file_path = os.path.join(source_dir, f"{file_name}.py")

    # Load the module dynamically.
    spec = importlib.util.spec_from_file_location(loss_function, os.path.abspath(file_path))
    module = spec.loader.load_module()

    # Fetch the loss fn class.
    loss_function_default = getattr(module, config['loss_function']['model'])
    
    # Initialize object.
    # NOTE: Any loss functions with specific settings should have them set here.
    if loss_function in ['NseLossBatchFlow','NseSqrtLossBatchFlow']:
        std_obs_flow = np.nanstd(obs[:, :, config['train']['target'].index('00060_Mean')], axis=0)
        loss_obj = loss_function_default(std_obs_flow)
    else:
        loss_obj = loss_function_default()
    
    return loss_obj
