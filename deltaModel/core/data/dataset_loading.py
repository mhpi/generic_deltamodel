import datetime as dt
import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import torch
from core.calc.normalize import init_norm_stats, trans_norm
from core.data.array_dataset import ArrayDataset
from core.data.dataframe_dataset import DataFrameDataset


class choose_class_to_read_dataset():
    def __init__(self, config, trange, data_path):
        self.config = config
        self.trange = trange
        self.data_path = data_path
        self._get_dataset_class()
        
    def _get_dataset_class(self) -> None:
        if self.data_path.endswith(".feather") or self.data_path.endswith(".csv"):
            self.read_data = DataFrameDataset(config=self.config, tRange=self.trange, data_path=self.data_path)
        elif self.data_path.endswith(".npy") or self.data_path.endswith(".pt"):
            self.read_data = ArrayDataset(config=self.config, tRange=self.trange, data_path=self.data_path)


def load_data(config, t_range=None, train=True):
    """ Load data into dictionaries for pNN and hydro model. """
    if t_range == None:
        t_range = config['t_range']

    out_dict = dict()

    if config['observations']['name'] in ['camels_671', 'camels_531']:
        if train:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attributes = pickle.load(f)
            
            startdate =config['train']['start_time']
            enddate = config['train']['end_time']
            
        else:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attributes = pickle.load(f)
        
            startdate =config['test']['start_time']
            enddate = config['test']['end_time']
            
        all_time = pd.date_range(config['observations']['start_time_all'], config['observations']['end_time_all'], freq='d')
        new_time = pd.date_range(startdate, enddate, freq='d')
        
        index_start = all_time.get_loc(new_time[0])
        index_end = all_time.get_loc(new_time[-1]) + 1

        # Subset forcings and attributes.
        attr_subset_idx = []
        for attr in config['dpl_model']['nn_model']['attributes']:
            if attr not in config['observations']['attributes_all']:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            attr_subset_idx.append(config['observations']['attributes_all'].index(attr))

        forcings = np.transpose(forcing[:,index_start:index_end], (1,0,2))
        forcing_subset_idx = []
        for forc in config['dpl_model']['nn_model']['forcings']:
            if forc not in config['observations']['forcings_all']:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            forcing_subset_idx.append(config['observations']['forcings_all'].index(forc))
        
        forcing_phy_subset_idx = []
        for forc in config['dpl_model']['phy_model']['forcings']:
            if forc not in config['observations']['forcings_all']:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            forcing_phy_subset_idx.append(config['observations']['forcings_all'].index(forc))

        out_dict['x_nn'] = forcings[:,:, forcing_subset_idx]  # Forcings for neural network (note, slight error from indexing)
        out_dict['x_phy'] = forcings[:,:, forcing_phy_subset_idx]  # Forcings for physics model
        out_dict['c_nn'] = attributes[:, attr_subset_idx] # Attributes
        out_dict['target'] = np.transpose(target[:,index_start:index_end], (1,0,2))  # Observation target
        
        ## For running a subset (531 basins) of CAMELS.
        if config['observations']['name'] == 'camels_531':
            gage_info = np.load(config['observations']['gage_info'])

            with open(config['observations']['subset_path'], 'r') as f:
                selected_camels = json.load(f)

            [C, Ind, subset_idx] = np.intersect1d(selected_camels, gage_info, return_indices=True)

            out_dict['x_nn'] = out_dict['x_nn'][:, subset_idx, :]
            out_dict['x_phy'] = out_dict['x_phy'][:, subset_idx, :]
            out_dict['c_nn'] = out_dict['c_nn'][subset_idx, :]
            out_dict['target'] = out_dict['target'][:, subset_idx, :]
            
    else:
        # Farshid data extractions
        forcing_dataset_class = choose_class_to_read_dataset(config, t_range, config['observations']['forcing_path'])
        out_dict['x_nn'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['dpl_model']['nn_model']['forcings'])
        out_dict['x_phy'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['phy_forcings'])
        out_dict['c_nn'] = forcing_dataset_class.read_data.getDataConst(config, varLst=config['dpl_model']['nn_model']['attributes'])
        out_dict['target'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['train']['target'])
    
    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(config, c_NN_sample, obs_sample):
    varTar_NN = config['train']['target']
    if 'flow_sim' in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index('flow_sim')]
        varC_NN = config['dpl_model']['nn_model']['attributes']
        area_name = config['observations']['area_name']
        
        c_area = c_NN_sample[:, varC_NN.index(area_name)]
        area = np.expand_dims(c_area, axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index('flow_sim')] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample


def get_dataset_dict(config, train=False):
    """
    Create dictionary of datasets used by the models.
    Contains 'c_nn', 'target', 'x_phy', 'x_nn_scaled'.

    train: bool, specifies whether data is for training.
    """

    # Create stats for NN input normalizations.
    if train: 
        dataset_dict = load_data(config, config['train_t_range'])
        init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                              dataset_dict['target'])
    else:
        dataset_dict = load_data(config, config['test_t_range'], train=False)

    # Normalization
    x_nn_scaled = trans_norm(config, np.swapaxes(dataset_dict['x_nn'], 1, 0).copy(),
                             var_lst=config['dpl_model']['nn_model']['forcings'], to_norm=True)
    x_nn_scaled[x_nn_scaled != x_nn_scaled] = 0  # Remove nans

    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['dpl_model']['nn_model']['attributes'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_scaled[c_nn_scaled != c_nn_scaled] = 0  # Remove nans
    c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)

    dataset_dict['x_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
    
    # Streamflow unit conversion.
    #### MOVED FROM LOAD_DATA
    if 'flow_sim' in config['train']['target']:
        dataset_dict['target'] = converting_flow_from_ft3_per_sec_to_mm_per_day(
            config,
            dataset_dict['c_nn'],
            dataset_dict['target']
        )

    return dataset_dict



def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind1, ind2






import datetime as dt
import logging
from abc import ABC, abstractmethod
from re import I
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


def time_to_date(t, hr=False):
    """Convert time to date or datetime object.
    
    Adapted from Farshid Rahmani.
    
    Parameters
    ----------
    t : int, datetime, date
        Time object to convert.
    hr : bool
        If True, return datetime object.
    """
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("Failed to change time to date.")
    return tOut


def trange_to_array(tRange, *, step=np.timedelta64(1, "D")):
    sd = time_to_date(tRange[0])
    ed = time_to_date(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


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

    # rho is 
    subset_dims = (config['train']['batch_size'], config['dpl_model']['rho'])

    i_sample, i_t = random_index(ngrid_train, nt, subset_dims, warm_up=warm_up)

    # Remove warmup days for dHBV1.1p...
    flow_obs = select_subset(config, dataset_dict['target'], i_sample, i_t,
                             config['dpl_model']['rho'], warm_up=warm_up)
    
    # if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
    # (config['dpl_model']['phy_model']['warm_up_states']) and (config['multimodel_type_type_type'] == 'none'):
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
            if key in ['x_phy', 'x_nn_scaled']:
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
    (config['dpl_model']['phy_model']['warm_up_states']) and (config['multimodel_type'] == 'none'):
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
                value.copy(),
                dtype=torch.float32,
                device=device
            )
    return data_dict
