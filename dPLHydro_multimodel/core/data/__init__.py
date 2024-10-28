import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from core.utils.time import trange_to_array

log = logging.getLogger(__name__)



class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args, **kwargs): #-> 'Hydrofabric'
        """
        Collate function with a flexible signature to allow for different inputs
        in subclasses. Implement this method in subclasses to handle specific
        data collation logic.
        """
        raise NotImplementedError


def random_index(ngrid: int, nt: int, dim_subset: Tuple[int, int],
                 warm_up: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(0 + warm_up, nt - rho, size=batch_size)
    return i_grid, i_t


def n_iter_nt_ngrid(x: np.ndarray, t_range: Tuple[int, int],
                    config: Dict, ngrid=None) -> Tuple[int, int, int]:
    nt = x.shape[0]
    if ngrid is None:
        ngrid = x.shape[1]

    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['rho'])
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['batch_size'] * rho / ngrid
                     / (nt - config['warm_up']))
        )
    )
    return ngrid, n_iter_ep, nt,


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


def take_sample_train(config: Dict,
                      dataset_dict: Dict[str, np.ndarray], 
                      ngrid_train: int,
                      nt: int,
                      ) -> Dict[str, torch.Tensor]:
    """
    Select random sample of data for training batch.
    """
    subset_dims = (config['batch_size'], config['rho'])

    i_grid, i_t = random_index(ngrid_train, nt, subset_dims, warm_up=config['warm_up'])

    # Remove warmup days for dHBV1.1p...
    flow_obs = select_subset(config, dataset_dict['obs'], i_grid, i_t,
                             config['rho'], warm_up=config['warm_up'])
    
    if ('HBV_capillary' in config['hydro_models']) and \
    (config['hbvcap_no_warm']) and (config['ensemble_type'] == 'none'):
        pass
    else:
        flow_obs = flow_obs[config['warm_up']:, :]
    
    # Create dataset sample dict.
    dataset_sample = {
        'iGrid': i_grid,
        'inputs_nn_scaled': select_subset(
            config, dataset_dict['inputs_nn_scaled'], i_grid, i_t,
            config['rho'], has_grad=False, warm_up=config['warm_up']
        ),
        'c_nn': torch.tensor(dataset_dict['c_nn'][i_grid],
                             device=config['device'], dtype=torch.float32),
        'obs': flow_obs,
        'x_hydro_model': select_subset(config, dataset_dict['x_hydro_model'],
                                       i_grid, i_t, config['rho'], warm_up=config['warm_up']),
        'c_hydro_model': torch.tensor(dataset_dict['c_hydro_model'][i_grid],
                                       device=config['device'], dtype=torch.float32)
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
            if key in ['x_hydro_model', 'inputs_nn_scaled']:
                warm_up = 0
            else:
                warm_up = config['warm_up']
            dataset_sample[key] = value[warm_up:, i_s:i_e, :].to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = value[i_s:i_e, :].to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Keep 'warmup' days for dHBV1.1p.
    if ('hbv_capillary' in config['hydro_models']) and \
    (config['hbvcap_no_warm']) and (config['ensemble_type'] == 'none'):
        pass
    else:
        dataset_sample['obs'] = dataset_sample['obs'][config['warm_up']:, :]

    return dataset_sample


def take_sample_train_merit(config: Dict,
                    dataset_dict: Dict[str, np.ndarray],
                    info_dict: Dict[str, np.ndarray],
                    ngrid_train: int,
                    nt: int,
                    maxNMerit: int
                    ) -> Dict[str, torch.Tensor]:
    """
    Select random sample of data for training batch.

    From hydroDL for handling GAGESII + MERIT basin data.
    """
    subset_dims = (config['batch_size'], config['rho'])
    i_grid, i_t = random_index(ngrid_train, nt, subset_dims,
                               warm_up=config['warm_up'])
        
    gage_key_batch = np.array(info_dict['gage_key'])[i_grid]
    area_info = info_dict['area_info']
    merit_idx = dataset_dict['merit_idx']

    ai_batch = []
    ac_batch = []
    id_list = []
    start_id = 0

    for gage_idx, gage in enumerate(gage_key_batch):
        if(start_id+len(area_info[gage]['unitarea'])) > config['merit_batch_max']:
            print("Minibatch will be shrunk to ",gage_idx," since it has nmerit of ", start_id+len(area_info[gage]['unitarea']))

            iGrid=iGrid[:gage_idx]
            i_t = i_t[:gage_idx]
            gage_key_batch = gage_key_batch[:gage_idx]
            
            break
        
        unitarea = area_info[gage]['unitarea'] / np.array(area_info[gage]['unitarea']).sum()
        uparea = area_info[gage]['uparea']
        ai_batch.extend(unitarea)
        ac_batch.extend(uparea)
        id_list.append(range(start_id, start_id + len(unitarea)))
        
        start_id += len(unitarea)

    if(len(ai_batch)>maxNMerit): maxNMerit = len(ai_batch)

    rho = config['rho']
    warm_up = config['warm_up'] 
    
    # Init subsets
    x_hydro_sub =  np.full((rho + warm_up, len(ac_batch), dataset_dict['x_nn_scaled'].shape[-1]), np.nan)
    x_nn_sub = x_hydro_sub.copy()
    c_nn_sub = np.full((len(ac_batch), dataset_dict['c_nn_scaled'].shape[-1]),np.nan)
    
    idx_matrix = np.zeros((len(ai_batch),len(gage_key_batch)))
    for gage_idx , gage in enumerate(gage_key_batch):
        st_gage = np.array(id_list[gage_idx])
        st_merit = np.array(merit_idx[gage]).astype(int)
        
        idx_matrix[st_gage, gage_idx] = 1
        
        x_hydro_sub[:, st_gage, :] = dataset_dict['x_hydro_model'][
            i_t[gage_idx] - warm_up:i_t[gage_idx] + rho, st_merit, :
        ]
        x_nn_sub[:, st_gage, :] = dataset_dict['x_nn_scaled'][
            i_t[gage_idx] - warm_up:i_t[gage_idx] + rho, st_merit, :
        ]
        c_nn_sub[st_gage, :] = dataset_dict['c_nn_scaled'][st_merit, :]
    
    # Combine scaled (see above) pNN inputs.
    inputs_nn_scaled = np.concatenate((
        x_nn_sub, 
        np.repeat(np.expand_dims(c_nn_sub, 0), x_nn_sub.shape[0], axis=0)), axis=2
        )
     
    dataset_sample = {
        'iGrid': i_grid,
        'inputs_nn_scaled': torch.from_numpy(inputs_nn_scaled).to(config['device']),
        'c_nn': torch.from_numpy(c_nn_sub).to(config['device']),
        'x_hydro_model': torch.from_numpy(x_hydro_sub).to(config['device']),
        'obs': select_subset(config, dataset_dict['obs'], i_grid, i_t,
                             config['rho'], warm_up=config['warm_up'])[config['warm_up']:],
        'ai_batch': ai_batch,
        'ac_batch': ac_batch,
        'idx_matrix': idx_matrix
    }

    return dataset_sample


def take_sample_test_merit(config: Dict,
                           dataset_dict: Dict[str, np.ndarray],
                           i_s: int,
                           i_e: int) -> Dict[str, torch.Tensor]:
    """
    Select random sample of data for training batch.

    From hydroDL for handling GAGESII + MERIT basin data.
    """
    COMID_batch = []
    gage_area_batch = []
    gage_key_batch = np.array(dataset_dict['gage_key'])[i_s:i_e]
    area_info = dataset_dict['area_info']


    for gage in gage_key_batch:
        gage_area_batch.append(np.array(area_info[gage]['unitarea']).sum())
        COMIDs = area_info[gage]['COMID']
        COMID_batch.extend(COMIDs)

    COMID_batch_unique = list(set(COMID_batch))
    COMID_batch_unique.sort()

    
    [_, idx_batch, subidx_batch] = np.intersect1d(COMID_batch_unique,
                                                  dataset_dict['merit_all'],
                                                  return_indices=True)  

    x_hydro_batch = dataset_dict['x_hydro_model'][:, subidx_batch,:]
    x_nn_scaled_batch = dataset_dict['x_nn_scaled'][:, subidx_batch,:]
    c_nn_scaled_batch = dataset_dict['c_nn_scaled'][subidx_batch,:]
    
    obs_batch = dataset_dict['obs'][config['warm_up']:, i_s:i_e]

    ai_batch = dataset_dict['ai_all'][subidx_batch]
    ac_batch = dataset_dict['ac_all'][subidx_batch]

    idx_matrix = np.zeros((len(ai_batch),len(gage_key_batch)))
    for i, gage in enumerate(gage_key_batch):
        COMIDs = area_info[gage]['COMID']                        
        [_, _,  subidx] = np.intersect1d(COMIDs,
                                         np.array(COMID_batch_unique)[idx_batch],
                                         return_indices=True)
        idx_matrix[subidx, i] = 1/gage_area_batch[i]

    # Combine pNN inputs
    inputs_nn_scaled = np.concatenate((
        x_nn_scaled_batch, 
        np.repeat(np.expand_dims(c_nn_scaled_batch, 0), x_nn_scaled_batch.shape[0], axis=0)), axis=2
        )
     
    dataset_sample = {
        'inputs_nn_scaled': torch.from_numpy(inputs_nn_scaled).to(config['device']),
        'c_nn': torch.from_numpy(c_nn_scaled_batch).to(config['device']),
        'x_hydro_model': torch.from_numpy(x_hydro_batch).to(config['device']),
        'obs': torch.from_numpy(obs_batch).to(config['device']),
        'ai_batch': ai_batch,
        'ac_batch': ac_batch,
        'idx_matrix': idx_matrix
    }

    del x_nn_scaled_batch, c_nn_scaled_batch

    return dataset_sample
