import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from core.utils.time import trange_to_array

log = logging.getLogger(__name__)



class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def getDataTs(self, config, varLst):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, config, varLst):
        raise NotImplementedError


def random_index(ngrid: int, nt: int, dim_subset: Tuple[int, int],
                 warm_up: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(0 + warm_up, nt - rho, size=batch_size)
    return i_grid, i_t


def create_training_grid(
    x: np.ndarray,
    config: Dict[str, Any],
    n_samples: int = None
) -> Tuple[int, int, int]:
    """Define a training grid of samples x iterations per epoch x time.

    Parameters
    ----------
    x : np.ndarray
        The input data for a model.
    config : dict
        The configuration dictionary.
    n_samples : int, optional
        The number of samples to use in the training grid.
    
    Returns
    -------
    Tuple[int, int, int]
        The number of samples, the number of iterations per epoch, and the
        number of timesteps.
    """
    t_range = config['train_t_range']
    n_t = x.shape[0]

    if n_samples is None:
        n_samples = x.shape[1]

    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['dpl_model']['rho'])

    # Calculate number of iterations per epoch.
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['train']['batch_size'] * rho / n_samples
                     / (n_t - config['dpl_model']['phy_model']['warm_up']))
        )
    )
    return n_samples, n_iter_ep, n_t,


def select_subset(config: Dict,
                  x: np.ndarray,
                  i_grid: np.ndarray,
                  i_t: np.ndarray,
                  rho: int,
                  *,
                  c: Optional[np.ndarray] = None,
                  tuple_out: bool = False,
                  has_grad: bool = False,
                  warm_up: int = 0
                  ) -> torch.Tensor:
    """
    Select a subset of input array.
    """
    nx = x.shape[-1]
    nt = x.shape[0]
    if x.shape[0] == len(i_grid):   #hack
        i_grid = np.arange(0,len(i_grid))  # hack
    if nt <= rho:
        i_t.fill(0)

    batch_size = i_grid.shape[0]

    if i_t is not None:
        x_tensor = torch.zeros([rho + warm_up, batch_size, nx], requires_grad=has_grad)
        for k in range(batch_size):
            temp = x[np.arange(i_t[k] - warm_up, i_t[k] + rho), i_grid[k]:i_grid[k] + 1, :]
            x_tensor[:, k:k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel (x = Ngrid * Ntime).
            x_tensor = torch.from_numpy(x[i_grid, :]).float()
        else:
            # Used for rho equal to the whole length of time series.
            x_tensor = torch.from_numpy(x[:, i_grid, :]).float()
            rho = x_tensor.shape[0]

    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[i_grid, :], [batch_size, 1, nc]), rho + warm_up, axis=1)
        c_tensor = torch.from_numpy(temp).float()

        if tuple_out:
            if torch.cuda.is_available():
                x_tensor = x_tensor.cuda()
                c_tensor = c_tensor.cuda()
            return x_tensor, c_tensor
        return torch.cat((x_tensor, c_tensor), dim=2)

    return x_tensor.to(config['device']) if torch.cuda.is_available() else x_tensor


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

    # Remove warmup days for dHBV1.1p...
    flow_obs = select_subset(config, dataset_dict['target'], i_sample, i_t,
                             config['dpl_model']['rho'], warm_up=warm_up)
    
    # if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
    # (config['dpl_model']['phy_model']['warm_up_states']) and (config['ensemble_type'] == 'none'):
    #     pass
    # else:
    flow_obs = flow_obs[warm_up:, :]
    
    # Create dataset sample dict.
    dataset_sample = {
        'batch_sample': i_sample,
        'x_phy': select_subset(config, dataset_dict['x_phy'],
                        i_sample, i_t, config['dpl_model']['rho'],
                        warm_up=warm_up),
        'x_nn_scaled': select_subset(
            config, dataset_dict['x_nn_scaled'], i_sample, i_t,
            config['dpl_model']['rho'], has_grad=False, warm_up=warm_up),
        'c_nn': torch.tensor(dataset_dict['c_nn'][i_sample],
                             device=config['device'], dtype=torch.float32),
        'target': flow_obs,

    }

    return dataset_sample


def take_sample_test(config: Dict, dataset_dict: Dict[str, torch.Tensor], 
                     i_s: int, i_e: int) -> Dict[str, torch.Tensor]:
    """
    Take sample of data for testing batch.
    """
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'x_nn_scaled']:
                warm_up = 0
            else:
                warm_up = config['dpl_model']['phy_model']['warm_up']
            dataset_sample[key] = value[warm_up:, i_s:i_e, :].to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = value[i_s:i_e, :].to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Keep 'warmup' days for dHBV1.1p.
    if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
    (config['dpl_model']['phy_model']['use_warmup_mode']) and (config['ensemble_type'] == 'none'):
        pass
    else:
        dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:, :]

    return dataset_sample


def take_sample(config: Dict, dataset_dict: Dict[str, torch.Tensor], days=730,
                basins=100) -> Dict[str, torch.Tensor]:
    """Take sample of data."""
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'x_nn_scaled']:
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
    (config['dpl_model']['phy_model']['use_warmup_mode']) and (config['ensemble_type'] == 'none'):
        pass
    else:
        dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:days, :basins]
    return dataset_sample


def numpy_to_torch_dict(data_dict: Dict[str, 
np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    """Convert numpy data dictionary to torch tensor dictionary.

    Parameters
    ----------
    data_dict : Dict[str, np.ndarray]
        The numpy data dictionary.
    device : str
        The device to move the data to.
    """
    for key, value in data_dict.items():
        if type(value) != torch.Tensor:
            data_dict[key] = torch.tensor(
                value.copy(),  #
                dtype=torch.float32).to(device)
    return data_dict
