import logging
import os
import random
import re
import sys
from typing import Any, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dmg.core.utils.dates import Dates
from dmg.core.utils.path import PathBuilder

log = logging.getLogger(__name__)


def set_system_spec(config: dict) -> tuple[str, str]:
    """Set the device and data type for the model on user's system.

    Parameters
    ----------
    cuda_devices
        List of CUDA devices to use. If None, the first available device is used.

    Returns
    -------
    tuple[str, str]
        The device type and data type for the model.
    """
    if config['device'] == 'cpu':
        device = torch.device('cpu')
    elif config['device'] == 'mps':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            raise ValueError("MPS is not available on this system.")
    elif config['device'] == 'cuda':
        # Set the first device as the active device.
        if torch.cuda.is_available() and config['gpu_id'] < torch.cuda.device_count():
            device = torch.device(f'cuda:{config['gpu_id']}')
            torch.cuda.set_device(device)
        else:
            raise ValueError(
                f"Selected CUDA device {config['gpu_id']} is not available."
            )
    else:
        raise ValueError(f"Invalid device: {config['device']}")

    dtype = torch.float32
    return str(device), str(dtype)


def set_randomseed(seed=0) -> None:
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed to set. If None, a random seed is used. Default is 0.
    """
    if seed is None:
        # seed = int(np.random.uniform(low=0, high=1e6))
        pass

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        log.warning(f"Error fixing randomseed: {e}")


def initialize_config(
    config: Union[DictConfig, dict],
) -> dict[str, Any]:
    """Parse and initialize configuration settings.

    Parameters
    ----------
    config
        Configuration settings from Hydra.

    Returns
    -------
    dict
        Formatted configuration settings.
    """
    # TODO: formalize this initializer

    if type(config) is DictConfig:
        try:
            # TODO: remove for dot-access configs
            config = OmegaConf.to_container(config, resolve=True)
        except ValidationError as e:
            log.exception("Configuration validation error", exc_info=e)
            raise e

    config['device'], config['dtype'] = set_system_spec(config)

    # Convert date ranges to integer values.
    train_time = Dates(config['train'], config['model']['rho'])
    test_time = Dates(config['test'], config['model']['rho'])
    sim_time = Dates(config['sim'], config['model']['rho'])
    all_time = Dates(config['observations'], config['model']['rho'])

    exp_time_start = min(
        train_time.start_time,
        train_time.end_time,
        test_time.start_time,
        test_time.end_time,
    )
    exp_time_end = max(
        train_time.start_time,
        train_time.end_time,
        test_time.start_time,
        test_time.end_time,
    )

    config['train_time'] = [train_time.start_time, train_time.end_time]
    config['test_time'] = [test_time.start_time, test_time.end_time]
    config['sim_time'] = [sim_time.start_time, sim_time.end_time]
    config['experiment_time'] = [exp_time_start, exp_time_end]
    config['all_time'] = [all_time.start_time, all_time.end_time]

    # TODO: add this handling directly to the trainer; this is not generalizable in current form.
    # change multimodel_type type to None if none.
    if config.get('multimodel_type', '') in ['none', 'None', '']:
        config['multimodel_type'] = None

    if config['model']['nn'].get('lr_scheduler', '') in [
        'none',
        'None',
        '',
    ]:
        config['model']['nn']['lr_scheduler'] = None

    if config.get('trained_model', '') in ['none', 'None', '']:
        config['trained_model'] = ''

    # Create output directories and add path to config.
    output_dir = os.getcwd()
    config['output_dir'] = output_dir
    config['model_dir'] = os.path.join(output_dir, 'model')
    config['plot_dir'] = os.path.join(output_dir, 'plot')
    config['sim_dir'] = os.path.join(output_dir, 'sim')

    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(config['plot_dir'], exist_ok=True)
    os.makedirs(config['sim_dir'], exist_ok=True)

    # Convert string back to data type.
    config['dtype'] = eval(config['dtype'])

    # Raytune
    config['do_tune'] = config.get('do_tune', False)

    # test type
    config['test']['type'] = config.get('test', {}).get('type', 'temporal')

    return config


def save_model(
    path: str,
    model: torch.nn.Module,
    model_name: str,
    epoch: int,
    make_dir: Optional[bool] = True,
) -> None:
    """Save model state dict.

    Parameters
    ----------
    path
        Path to save the model.
    model
        Model to save.
    model_name
        Name of the model.
    epoch
        Last completed epoch of training.
    make_dir
        Create directories for saving files.
    """
    if (not os.path.exists(path)) and make_dir:
        os.makedirs(path)

    torch.save(
        model.state_dict(),
        os.path.join(path, f"{model_name.lower()}_ep{str(epoch)}.pt"),
    )


def save_train_state(
    path: str,
    epoch: int,
    optimizer: torch.nn.Module,
    scheduler: Optional[torch.nn.Module] = None,
    make_dir: Optional[bool] = True,
    clear_prior: Optional[bool] = False,
) -> None:
    """Save dict of all experiment states for training.

    Parameters
    ----------
    path
        Path to save the training state.
    epoch
        Last completed epoch of training.
    optimizer
        Optimizer state dict.
    scheduler
        Learning rate scheduler state dict.
    make_dir
        Create directories for saving files.
    clear_prior
        Clear previous saved states.
    """
    if (not os.path.exists(path)) and make_dir:
        os.makedirs(path)

    if clear_prior:
        for file in os.listdir(path):
            if 'trainer_state' in file:
                os.remove(os.path.join(path, file))

    scheduler_state = None
    cuda_state = None

    if scheduler:
        scheduler_state = scheduler.state_dict()
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()

    torch.save(
        {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_state,
            'random_state': torch.get_rng_state(),
            'cuda_state': cuda_state,
        },
        os.path.join(path, f'trainer_state_ep{str(epoch)}.pt'),
    )


def save_outputs(config, predictions, y_obs=None, create_dirs=False) -> None:
    """Save outputs from a model."""
    if torch.is_tensor(y_obs):
        y_obs = y_obs.cpu().numpy()

    if create_dirs:
        out_path = PathBuilder(config)
        out_path.write_path(config)

    if type(predictions) is list:
        # Handle a single model
        for key in predictions[0].keys():
            if len(predictions[0][key].shape) == 3:
                dim = 1
            else:
                dim = 0

            c_tensor = torch.cat([d[key] for d in predictions], dim=dim)
            file_name = key + ".npy"

            np.save(os.path.join(config['out_path'], file_name), c_tensor.numpy())

    elif type(predictions) is dict:
        # Handle multiple models
        models = config['delta_model']['phy_model']['model']
        for key in predictions[models[0]][0].keys():
            out_dict = {}

            if len(predictions[models[0]][0][key].shape) == 3:
                dim = 1
            else:
                dim = 0

            for model in models:
                out_dict[model] = torch.cat(
                    [d[key] for d in predictions[model]],
                    dim=dim,
                ).numpy()

            file_name = key + '.npy'
            np.save(os.path.join(config['out_path'], file_name), out_dict)

    else:
        raise ValueError("Invalid output format.")

    # Reading flow observation
    if y_obs is not None:
        for var in config['train']['target']:
            item_obs = y_obs[:, :, config['train']['target'].index(var)]
            file_name = var + '_obs.npy'
            np.save(os.path.join(config['out_path'], file_name), item_obs)


def load_model(config, model_name, epoch):
    """Load trained PyTorch models.

    Parameters
    ----------
    config
        Configuration dictionary with paths and model settings.
    model_name
        Name of the model to load.
    epoch
        Epoch number to load the specific state of the model.

    Returns
    -------
    torch.nn.Module
        An initialized PyTorch model.
    """
    model_name = str(model_name) + '_model_Ep' + str(epoch) + '.pt'


def print_config(config: dict[str, Any]) -> None:
    """Print the current configuration settings.

    Parameters
    ----------
    config
        Dictionary of configuration settings.

    Adapted from Jiangtao Liu.
    """
    print()
    print("\033[1m" + "Current Configuration" + "\033[0m")
    print(f"  {'Experiment Mode:':<20}{config['mode']:<20}")
    if config['multimodel_type'] is not None:
        print(f"  {'Ensemble Mode:':<20}{config['multimodel_type']:<20}")
    for i, mod in enumerate(config['model']['phy']['name']):
        print(f"  {f'Model {i + 1}:':<20}{mod:<20}")
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f"  {'Data Source:':<20}{config['observations']['name']:<20}")
    if 'train' in config['mode']:
        print(
            f"  {'Train Range :':<20}{config['train']['start_time']:<20}{config['train']['end_time']:<20}"
        )
    if 'test' in config['mode']:
        print(
            f"  {'Test Range :':<20}{config['test']['start_time']:<20}{config['test']['end_time']:<20}"
        )
    if 'simulation' in config['mode']:
        print(
            f"  {'Simulation Range :':<20}{config['sim']['start_time']:<20}{config['sim']['end_time']:<20}"
        )
    if config['train']['start_epoch'] > 0 and 'train' in config['mode']:
        print(
            f"  {'Resume training from epoch:':<20}{config['train']['start_epoch']:<20}"
        )
    print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(
        f"  {'Train Epochs:':<20}{config['train']['epochs']:<20}{'Batch Size:':<20}{config['train']['batch_size']:<20}"
    )
    if config['model']['nn']['name'] != 'LstmMlpModel':
        print(
            f"  {'Dropout:':<20}{config['model']['nn']['dropout']:<20}{'Hidden Size:':<20}{config['model']['nn']['hidden_size']:<20}"
        )
    else:
        print(
            f"  {'LSTM Dropout:':<20}{config['model']['nn']['lstm_dropout']:<20}{'LSTM Hidden Size:':<20}{config['model']['nn']['lstm_hidden_size']:<20}"
        )
        print(
            f"  {'MLP Dropout:':<20}{config['model']['nn']['mlp_dropout']:<20}{'MLP Hidden Size:':<20}{config['model']['nn']['mlp_hidden_size']:<20}"
        )
    print(
        f"  {'Warmup:':<20}{config['model']['phy'].get('warm_up', '0'):<20}{'Concurrent Models:':<20}{config['model']['phy']['nmul']:<20}"
    )
    print(f"  {'Loss Fn:':<20}{config['loss_function']['name']:<20}")
    print()

    if config['multimodel_type'] is not None:
        print("\033[1m" + "Multimodel Parameters" + "\033[0m")
        print(
            f"  {'Mosaic:':<20}{config['multimodel']['mosaic']:<20}{'Dropout:':<20}{config['multimodel']['dropout']:<20}"
        )
        print(
            f"  {'Learning Rate:':<20}{config['multimodel']['learning_rate']:<20}{'Hidden Size:':<20}{config['multimodel']['hidden_size']:<20}"
        )
        print(
            f"  {'Scaling Fn:':<20}{config['multimodel']['scaling_function']:<20}{'Loss Fn:':<20}{config['multimodel']['loss_function']:<20}"
        )
        print(
            f"  {'Range-bound Loss:':<20}{config['multimodel']['use_rb_loss']:<20}{'Loss Factor:':<20}{config['multimodel']['loss_factor']:<20}"
        )
        print()

    print("\033[1m" + 'Machine' + "\033[0m")
    print(f"  {'Use Device:':<20}{str(config['device']):<20}")
    print()


def find_shared_keys(*dicts: dict[str, Any]) -> list[str]:
    """Find keys shared between multiple dictionaries.

    Parameters
    ----------
    *dicts
        Variable number of dictionaries.

    Returns
    -------
    list[str]
        A list of keys shared between the input dictionaries.
    """
    if len(dicts) == 1:
        return []

    # Start with the keys of the first dictionary.
    shared_keys = set(dicts[0].keys())

    # Intersect with the keys of all other dictionaries.
    for d in dicts[1:]:
        shared_keys.intersection_update(d.keys())

    return list(shared_keys)


def snake_to_camel(snake_str):
    """
    Convert snake strings (underscore word separation, lower case) to
    Camel-case strings (no word separation, capitalized first letter of a word).
    """
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def camel_to_snake(camel_str):
    """
    Convert CamelCase or PascalCase strings to snake_case while properly handling
    consecutive uppercase letters (e.g., 'DiracDF' -> 'dirac_df').
    """
    return re.sub(r'([a-z])([A-Z])', r'\1_\2', camel_str).lower()


def format_resample_interval(resample: str) -> str:
    """Formats the resampling interval into a human-readable string.

    Parameters
    ----------
    resample
        The resampling interval (e.g., 'D', 'W', '3D', 'M', 'Y').

    Returns
    -------
    str
        A formatted string describing the resampling interval.
    """
    # Check if the interval contains a number (e.g., "3D")
    if any(char.isdigit() for char in resample):
        # Extract the numeric part and the unit part
        num = ''.join(filter(str.isdigit, resample))
        unit = ''.join(filter(str.isalpha, resample))

        # Map units to human-readable names
        if num == '1':
            unit_map = {
                'D': 'daily',
                'W': 'weekly',
                'M': 'monthly',
                'Y': 'yearly',
            }
        else:
            unit_map = {
                'D': 'days',
                'W': 'weeks',
                'M': 'months',
                'Y': 'years',
            }
        return f"{num} {unit_map.get(unit, unit)}"
    else:
        # Single-character intervals (e.g., "D", "W")
        unit_map = {
            'D': 'daily',
            'W': 'weekly',
            'M': 'monthly',
            'Y': 'yearly',
        }
        return unit_map.get(resample, resample)
