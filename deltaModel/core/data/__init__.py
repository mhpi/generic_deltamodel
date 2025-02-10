import datetime as dt
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind2


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

    if type(t) is str:
        t = int(t.replace('/', ''))

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


def random_index(
    ngrid: int,
    nt: int,
    dim_subset: Tuple[int, int],
    warm_up: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
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
    t_range = config['train_time']
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
        'xc_nn_norm': select_subset(
            config, dataset_dict['xc_nn_norm'], i_sample, i_t,
            config['dpl_model']['rho'], has_grad=False, warm_up=warm_up),
        'c_nn': torch.tensor(dataset_dict['c_nn'][i_sample],
                             device=config['device'], dtype=torch.float32),
        'target': flow_obs,

    }

    return dataset_sample


def get_training_sample_2_0(
    dataset_dict: Dict[str, np.ndarray], 
    ngrid_train: int,
    nt: int,
    config: Dict,
) -> Dict[str, torch.Tensor]:
    """Select random sample of data for training batch."""
    warm_up = config['dpl_model']['phy_model']['warm_up']
    nsubbasin_max = config['dpl_model']['phy_model']['nsubbasin_max']
    subset_dims = (config['train']['batch_size'], config['dpl_model']['rho'])

    i_sample, i_t = random_index(ngrid_train, nt, subset_dims, warm_up=warm_up)

    gage_batch = np.array(dataset_dict['selected_gage'])[i_sample]
    subbasin_idx = dataset_dict['subbasin_idx']

    id_list = []
    start_id = 0
    for gage_idx, gage in enumerate(gage_batch):
        if(start_id+len(subbasin_idx[gage])) > nsubbasin_max:
            print("Minibatch size is shrinked to ",gage_idx,".  The maximum number of subbasin is ", start_id+len(subbasin_idx[gage]),f", out of {nsubbasin_max} in the setting ")
            
            i_sample=i_sample[:gage_idx]
            i_t = i_t[:gage_idx]
            
            gage_batch = gage_batch[:gage_idx]
            
            break

        id_list.append(range(start_id, start_id+len(subbasin_idx[gage])))
        
        start_id = start_id+len(subbasin_idx[gage])

        xTrain =  np.full((start_id,config['dpl_model']['rho']+warm_up,dataset_dict['xc_nn_norm'].shape[-1]),np.nan)
        xTrain2 = np.full((start_id,config['dpl_model']['rho']+warm_up,dataset_dict['xc_nn_norm'].shape[-1]),np.nan)
        attr2 = np.full((start_id,dataset_dict['c_nn_norm'].shape[-1]),np.nan)
        
        idx_matric = np.zeros((start_id,len(gage_batch)))
        Ai_batch = []
        Ac_batch = []
        Ele_batch = []        
        for gageidx , gage in enumerate(gage_batch):

            idx_matric[np.array(id_list[gageidx]),gageidx] = 1
            
            Ai_batch.extend(dataset_dict['Ai_all'][np.array(subbasin_idx[gage]).astype(int)]/np.array(dataset_dict['Ai_all'][np.array(subbasin_idx[gage]).astype(int)]).sum())
            Ac_batch.extend(dataset_dict['Ac_all'][np.array(subbasin_idx[gage]).astype(int)])
            Ele_batch.extend(dataset_dict['Ele_all'][np.array(subbasin_idx[gage]).astype(int)])
            xTrain[np.array(id_list[gageidx]),:,:] = dataset_dict['x_phy'][np.array(subbasin_idx[gage]).astype(int),i_t[gageidx]-warm_up:i_t[gageidx]+config['dpl_model']['rho'],:]
            xTrain2[np.array(id_list[gageidx]),:,:] = dataset_dict['xc_nn_norm'][np.array(subbasin_idx[gage]).astype(int),i_t[gageidx]-warm_up:i_t[gageidx]+config['dpl_model']['rho'],:]
            attr2[np.array(id_list[gageidx]),:] = dataset_dict['c_nn_norm'][np.array(subbasin_idx[gage]).astype(int),:]   

        xTrain_torch = torch.from_numpy(np.swapaxes(xTrain, 0,1)).to(config['device'])
        attr_torch = torch.from_numpy(attr2).to(config['device'])

        attr2_expand = np.repeat(np.expand_dims(attr2, axis=1), xTrain2.shape[1], axis=1)
        zTrain2 = np.concatenate((xTrain2,attr2_expand),axis = -1)
        zTrain2_torch = torch.from_numpy(np.swapaxes(zTrain2, 0, 1)).to(config['device'])
        




    flow_obs = select_subset(config, np.transpose(dataset_dict['target'], (1,0,2)), i_sample, i_t,
                             config['dpl_model']['rho'], warm_up=warm_up)

    flow_obs = flow_obs[warm_up:, :]
    
    # Create dataset sample dict.
    dataset_sample = {
        'batch_sample': i_sample,
        'x_phy': xTrain_torch,
        'xc_nn_norm': zTrain2_torch,
        'c_nn_norm': attr_torch,
        'Ai_batch': Ai_batch,
        'Ac_batch': Ac_batch,
        'Ele_batch': Ele_batch,
        'idx_matric': idx_matric,
        'target': flow_obs,

    }

    return dataset_sample


def get_validation_sample(
    dataset_dict: Dict[str, torch.Tensor],
    i_s: int,
    i_e: int,
    config: Dict
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
        elif value.ndim == 1:
            dataset_sample[key] = torch.tensor(
                value[i_s:i_e],
                dtype=torch.float32,
                device = config['device']
            )
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 1, 2 or 3 dimensions.")

    # Keep 'warmup' days for dHBV1.1p.
    # if ('HBV1_1p' in config['dpl_model']['phy_model']['model']) and \
    # (config['dpl_model']['phy_model']['use_warmup_mode']) and (config['multimodel_type'] == 'none'):
    #     pass
    # else:
    # dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:, :]

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
    (config['dpl_model']['phy_model']['use_warmup_mode']) and (config['multimodel_type'] == 'none'):
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


def load_json(file_path: str) -> Dict:
    """Load JSON data from a file and return it as a dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file to load.
    
    Returns
    -------
    dict
        Dictionary containing the JSON data.
    """   
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

            if isinstance(data, str):
                # If json is still a string, decode again.
                return json.loads(data) 
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file '{file_path}'.")
        return None


def txt_to_array(txt_path: str):
    """Load txt file of gage ids to numpy array."""
    with open(txt_path, 'r') as f:
        lines = f.read().strip()  # Remove extra whitespace
        lines = lines.replace("[", "").replace("]", "")  # Remove brackets
    return np.array([int(x) for x in lines.split(",")])


def timestep_resample(
    data: Union[pd.DataFrame, np.ndarray, torch.Tensor],
    resolution: str = 'M',
    method: str = 'mean',
) -> pd.DataFrame:
    """
    Resample the data to a given resolution.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        The data to resample.
    resolution : str
        The resolution to resample the data to. Default is 'M' (monthly).
    method : str
        The resampling method. Default is 'mean'.

    Returns
    -------
    pd.DataFrame
        The resampled data.
    """
    if isinstance(data, pd.DataFrame):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
        data = pd.DataFrame(data)
        data['time'] = pd.to_datetime(data['time'])
    elif isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
        data['time'] = pd.to_datetime(data['time'])
    else:
        raise ValueError(f"Data type not supported: {type(data)}")
    
    data.set_index('time', inplace=True)
    data_resample = data.resample(resolution).agg(method)

    data_resample['time'] = data_resample.index

    return data_resample
