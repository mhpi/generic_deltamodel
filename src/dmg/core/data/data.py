import datetime as dt
import json
import logging
import traceback
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def intersect(tLst1: list[np.float32], tLst2: list[np.float32]) -> list[int]:
    """Find the intersection of two lists."""
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind2


def time_to_date(t: int, hr: bool = False) -> Union[dt.date, dt.datetime]:
    """Convert time to date or datetime object.
    
    Adapted from Farshid Rahmani.
    
    Parameters
    ----------
    t
        Time object to convert.
    hr
        If True, return datetime object.
    
    Returns
    -------
    Union[dt.date, dt.datetime]
        The converted date or datetime object.
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


def trange_to_array(
    t_range: NDArray[np.float32],
    *,
    step=np.timedelta64(1, "D"),
) -> NDArray[np.float32]:
    """Convert time range to array of dates.
    
    Parameters
    ----------
    t_range
        Time range to convert.
    step
        Step size for the array.
    
    Returns
    -------
    NDArray[np.float32]
        Array of dates.
    """
    sd = time_to_date(t_range[0])
    ed = time_to_date(t_range[1])
    return np.arange(sd, ed, step)


def random_index(
    ngrid: int,
    nt: int,
    dim_subset: tuple[int, int],
    warm_up: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Generate random indices for grid and time.
    
    Parameters
    ----------
    ngrid
        Number of grid points.
    nt
        Number of time steps.
    dim_subset
        Tuple of batch size and rho.
    warm_up
        Number of warm-up time steps.
    
    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        Random grid and time indices.
    """
    batch_size, rho = dim_subset
    i_grid = np.random.randint(0, ngrid, size=batch_size)
    i_t = np.random.randint(0 + warm_up, nt - rho, size=batch_size)
    return i_grid, i_t


def create_training_grid(
    x: NDArray[np.float32],
    config: dict[str, Any],
    n_samples: int = None,
) -> tuple[int, int, int]:
    """Define a training grid of samples x iterations per epoch x time.

    Parameters
    ----------
    x
        The input data for a model.
    config
        Configuration dictionary.
    n_samples
        The number of samples to use in the training grid.
    
    Returns
    -------
    tuple[int, int, int]
        The number of samples, the number of iterations per epoch, and the
        number of timesteps.
    """
    t_range = config['train_time']
    n_t = x.shape[0]

    if n_samples is None:
        n_samples = x.shape[1]

    t = trange_to_array(t_range)
    rho = min(t.shape[0], config['delta_model']['rho'])

    # Calculate number of iterations per epoch.
    n_iter_ep = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - config['train']['batch_size'] * rho / n_samples
                     / (n_t - config['delta_model']['phy_model'].get('warm_up', 0)),
            ),
        ),
    )
    return n_samples, n_iter_ep, n_t,


def select_subset(
    config: dict,
    x: NDArray[np.float32],
    i_grid: NDArray[np.float32],
    i_t: NDArray[np.float32],
    rho: int,
    *,
    c: Optional[NDArray[np.float32]] = None,
    tuple_out: bool = False,
    has_grad: bool = False,
    warm_up: int = 0,
) -> torch.Tensor:
    """Select a subset of input array.

    Parameters
    ----------
    config
        Configuration dictionary.
    x
        Input data.
    i_grid
        Grid indices.
    i_t
        Time indices.
    rho
        Number of time steps.
    c
        Optional static data.
    tuple_out
        If True, return a tuple of tensors.
    has_grad
        If True, create tensors with gradient tracking.
    warm_up
        Number of warm-up time steps.
    
    Returns
    -------
    torch.Tensor
        The selected subset of the input array.
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


def numpy_to_torch_dict(
    data_dict: dict[str, NDArray[np.float32]],
    device: str,
) -> dict[str, torch.Tensor]:
    """Convert numpy data dictionary to torch tensor dictionary.

    Parameters
    ----------
    data_dict
        The numpy data dictionary.
    device
        The device to move the data to.
    
    Returns
    -------
    dict[str, torch.Tensor]
        The torch tensor dictionary.
    """
    for key, value in data_dict.items():
        if type(value) is torch.Tensor:
            data_dict[key] = torch.tensor(
                value.copy(),
                dtype=torch.float32,
                device=device,
            )
    return data_dict


def load_json(file_path: str) -> dict:
    """Load JSON data from a file and return it as a dictionary.
    
    Parameters
    ----------
    file_path
        Path to the JSON file to load.
    
    Returns
    -------
    dict
        Dictionary containing the JSON data.
    """
    try:
        with open(file_path) as file:
            data = json.load(file)

            if isinstance(data, str):
                # If json is still a string, decode again.
                return json.loads(data)
            return data
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: File '{file_path}' not found.") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error: Failed to decode JSON from file '{file_path}'.") from e


def txt_to_array(txt_path: str) -> NDArray[np.float32]:
    """Load txt file of gage ids to numpy array.
    
    Parameters
    ----------
    txt_path
        Path to the txt file to load.
    
    Returns
    -------
    NDArray[np.float32]
        Array of txt values converted to int.
    """
    with open(txt_path) as f:
        lines = f.read().strip()  # Remove extra whitespace
        lines = lines.replace("[", "").replace("]", "")  # Remove brackets
    return np.array([int(x) for x in lines.split(",")])


def timestep_resample(
    data: Union[pd.DataFrame, NDArray[np.float32], torch.Tensor],
    resolution: str = 'M',
    method: str = 'mean',
) -> pd.DataFrame:
    """
    Resample the data to a given resolution.

    Parameters
    ----------
    data
        The data to resample.
    resolution
        The resolution to resample the data to. Default is 'M' (monthly).
    method
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
    elif isinstance(data, NDArray[np.float32]):
        data = pd.DataFrame(data)
        data['time'] = pd.to_datetime(data['time'])
    else:
        raise ValueError(f"Data type not supported: {type(data)}")

    data.set_index('time', inplace=True)
    data_resample = data.resample(resolution).agg(method)

    data_resample['time'] = data_resample.index

    return data_resample

def split_dataset_by_basin(
    dataset: dict[str, torch.Tensor],
    config: dict[str, Any],
    holdout_index: int,
) -> tuple[Optional[dict[str, torch.Tensor]], Optional[dict[str, torch.Tensor]]]:
    """Split dataset by basin for spatial testing.
    
    NOTE: Currently this is based on the gage_split file we may want to change
    to this in the future.
    
    Parameters
    ----------
    dataset
        Dictionary of torch tensors containing the full dataset
    config
        Configuration dictionary containing test settings
    holdout_index
        Index for the current holdout experiment
        
    Returns
    -------
    Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]
        Tuple of (train_dataset, test_dataset) split by basins
    """
    if not dataset:
        log.warning("Empty dataset provided for spatial splitting")
        return dataset, dataset
    
    test_config = config.get('test', {})
    if test_config.get('type') != 'spatial':
        log.info("Not a spatial test, returning original dataset")
        return dataset, dataset
    
    try:
        extent = test_config.get('extent')
        holdout_gages = []
        
        # Determine holdout basins based on test extent type
        if extent == 'PUR':
            huc_regions = test_config.get('huc_regions', [])
            if holdout_index >= len(huc_regions):
                raise ValueError(f"Invalid holdout index {holdout_index} for {len(huc_regions)} HUC regions")
                
            holdout_hucs = huc_regions[holdout_index]
            gage_file = config['observations']['gage_split_file']
            
            log.info(f"Loading PUR split data from {gage_file}")
            gageinfo = pd.read_csv(gage_file, dtype={"huc": int, "gage": str})
            holdout_hucs_int = [int(huc) for huc in holdout_hucs]
            holdout_gages = gageinfo[gageinfo['huc'].isin(holdout_hucs_int)]['gage'].tolist()
            
        elif extent == 'PUB':
            pub_ids = test_config.get('PUB_ids', [])
            if holdout_index >= len(pub_ids):
                raise ValueError(f"Invalid holdout index {holdout_index} for {len(pub_ids)} PUB IDs")
                
            holdout_pub = pub_ids[holdout_index]
            gage_file = config['observations']['gage_split_file']
            
            log.info(f"Loading PUB split data from {gage_file}")
            gageinfo = pd.read_csv(gage_file, dtype={"PUB_ID": int, "gage": str})
            holdout_gages = gageinfo[gageinfo['PUB_ID'] == holdout_pub]['gage'].tolist()
            
        else:
            raise ValueError(f"Unsupported spatial test extent: {extent}")
        
        # Load the complete basin list
        subset_path = config['observations']['subset_path']
        log.info(f"Loading basin subset from {subset_path}")
        
        with open(subset_path) as f:
            content = f.read().strip()
            if content.startswith('[') and content.endswith(']'):
                # Handle list format
                content = content.strip('[]')
                all_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
            else:
                # Handle plain text format
                all_basins = [line.strip() for line in content.split() if line.strip()]
        
        # Create basin indices for train/test split
        holdout_gages_set = [int(str(basin).strip()) for basin in holdout_gages]
        test_indices = []
        train_indices = []
        
        for i, basin in enumerate(all_basins):
            basin_int = int(str(basin).strip())
            if basin_int in holdout_gages_set:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        if not test_indices:
            raise ValueError("No test basins found for current holdout!")
        if not train_indices:
            raise ValueError("No training basins found for current holdout!")
            
        log.info(f"Spatial split: {len(train_indices)} training basins, {len(test_indices)} test basins")
        
        # Perform the actual dataset splitting
        train_data = {}
        test_data = {}
        train_indices_tensor = torch.tensor(train_indices, device='cpu')
        test_indices_tensor = torch.tensor(test_indices, device='cpu')
        
        for key, tensor in dataset.items():
            if tensor is None:
                train_data[key] = None
                test_data[key] = None
                continue
                
            # Move to CPU for indexing
            cpu_tensor = tensor.to('cpu')
            original_device = tensor.device
            
            if len(cpu_tensor.shape) == 3:
                # Handle 3D tensors - could be [basins, time, features] or [time, basins, features]
                if cpu_tensor.shape[0] == len(all_basins):
                    # [basins, time, features] format
                    train_data[key] = cpu_tensor[train_indices_tensor]
                    test_data[key] = cpu_tensor[test_indices_tensor]
                else:
                    # [time, basins, features] format (typical for HydroLoader)
                    train_data[key] = cpu_tensor[:, train_indices_tensor]
                    test_data[key] = cpu_tensor[:, test_indices_tensor]
                    
            elif len(cpu_tensor.shape) == 2:
                # Handle 2D tensors - typically [basins, features]
                train_data[key] = cpu_tensor[train_indices_tensor]
                test_data[key] = cpu_tensor[test_indices_tensor]
            else:
                # For 1D or other tensors, keep unchanged
                train_data[key] = tensor
                test_data[key] = tensor
                continue
            
            # Move back to original device
            train_data[key] = train_data[key].to(original_device)
            test_data[key] = test_data[key].to(original_device)
        
        log.info("Successfully split dataset by basins for spatial testing")
        return train_data, test_data
        
    except Exception as e:
        log.error(traceback.format_exc())
        raise RuntimeError(f"Error splitting dataset by basin: {e}") from e
