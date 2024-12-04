from core.data.data_samplers.base import BaseDataSampler

import numpy as np
import torch
from typing import Dict, Optional
from core.data import random_index

class HydroDataSampler(BaseDataSampler):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def load_data(self):
        # Custom implementation for loading data
        print("Loading data...")

    def preprocess_data(self):
        # Custom implementation for preprocessing data
        print("Preprocessing data...")

def select_subset(
    config: Dict,
    x: torch.Tensor,
    i_grid: np.ndarray,
    i_t: np.ndarray,
    rho: int,
    *,
    c: Optional[np.ndarray] = None,
    tuple_out: bool = False,
    has_grad: bool = False,
    warm_up: int = 0
) -> torch.Tensor:
    """Select a subset of input tensor."""
    nx = x.shape[-1]
    nt = x.shape[0]
    device = config['device']

    if x.shape[0] == len(i_grid):
        i_grid = np.arange(0, len(i_grid))
    if nt <= rho:
        i_t.fill(0)

    batch_size = i_grid.shape[0]

    if i_t is not None:
        x_tensor = torch.zeros([rho + warm_up, batch_size, nx], device=device, requires_grad=has_grad)
        for k in range(batch_size):
            temp = x[i_t[k] - warm_up:i_t[k] + rho, i_grid[k]:i_grid[k] + 1, :]
            x_tensor[:, k:k + 1, :] = temp
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel (x = Ngrid * Ntime).
            x_tensor = x[i_grid, :].float().to(device)
        else:
            # Used for rho equal to the whole length of time series.
            x_tensor = x[:, i_grid, :].float().to(device)
            rho = x_tensor.shape[0]

    if c is not None:
        c = torch.from_numpy(c).float().to(device)
        nc = c.shape[-1]
        temp = c[i_grid, :].unsqueeze(1).repeat(1, rho + warm_up, 1)
        c_tensor = temp.to(device)

        if tuple_out:
            return x_tensor, c_tensor
        return torch.cat((x_tensor, c_tensor), dim=2)

    return x_tensor

def get_training_sample(
    dataset_dict: Dict[str, np.ndarray], 
    ngrid_train: int,
    nt: int,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    """Select random sample of data for training batch."""
    warm_up = config['dpl_model']['phy_model']['warm_up']

    subset_dims = (config['train']['batch_size'], config['dpl_model']['rho'])

    i_sample, i_t = random_index(ngrid_train, nt, subset_dims, warm_up=warm_up)

    flow_obs = select_subset(config, dataset_dict['target'], i_sample, i_t,
                             config['dpl_model']['rho'], warm_up=warm_up)
    
    flow_obs = flow_obs[warm_up:, :]
    
    # Create dataset sample dict
    dataset_sample = {
        'x_phy': select_subset(
            config,
            dataset_dict['x_phy'],
            i_sample,
            i_t,
            config['dpl_model']['rho'],
            warm_up=warm_up,
        ),
        'c_phy': dataset_dict['c_phy'][i_sample],
        'c_nn': dataset_dict['c_nn'][i_sample],
        'xc_nn_norm': select_subset(
            config,
            dataset_dict['xc_nn_norm'],
            i_sample,
            i_t,
            config['dpl_model']['rho'],
            has_grad=False,
            warm_up=warm_up,
        ),
        'target': flow_obs,
        'batch_sample': i_sample,
    }
    return dataset_sample


def get_validation_sample(
    dataset_dict: Dict[str, torch.Tensor],
    i_s: int,
    i_e: int,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    """
    Take sample of data for testing batch.
    """
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['dpl_model']['phy_model']['warm_up']
            dataset_sample[key] = torch.tensor(
                value[warm_up:, i_s:i_e, :],
                dtype=torch.float32,
                device = config['device']
            )
        elif value.ndim == 2:
            dataset_sample[key] = torch.tensor(
                value[i_s:i_e, :],
                dtype=torch.float32,
                device = config['device']
            )
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    return dataset_sample


def take_sample(config: Dict, dataset_dict: Dict[str, torch.Tensor], days=730,
                basins=100) -> Dict[str, torch.Tensor]:
    """Take sample of data."""
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['dpl_model']['phy_model']['warm_up']
            dataset_sample[key] = torch.tensor(value[warm_up:days,:basins, :], dtype=torch.float32).to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = torch.tensor(value[:basins, :], dtype=torch.float32).to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Keep 'warmup' days for dHBV1.1p.
    if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
    (config['dpl_model']['phy_model']['warm_up_states']) and (config['multimodel_type'] == 'none'):
        pass
    else:
        dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:days, :basins]
    return dataset_sample
