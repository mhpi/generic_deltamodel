import logging
import os
import json
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from numpy.typing import NDArray
from dmg.core.data.loaders.load_nc import NetCDFDataset
from dmg.core.data.data import intersect
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

log = logging.getLogger(__name__)


def load_nn_data(
    config: Dict[str, Any], 
    scope: str, 
    t_range: Dict[str, str],
    nn_forcings: List[str],
    nn_attributes: List[str],
    target: List[str],  # This will be [] for FinetuneLoader
    device: str,
    nc_tool: NetCDFDataset
) -> Dict[str, np.ndarray]:
    """Load and process neural network data from NetCDF."""
    time_range = [t_range['start'].replace('/', '-'), t_range['end'].replace('/', '-')]
    warmup_days = config['delta_model']['phy_model']['warm_up']
    
    try:
        # Create a combined list of variables to load (includes target ONLY if target is not empty)
        all_variables = nn_forcings.copy()
        
        # ONLY add target variables if target list is not empty
        if target:  # This checks if target list is not empty
            for target_var in target:
                if target_var not in all_variables:
                    all_variables.append(target_var)
        
        # Load ALL stations first to avoid station ID format mismatch
        # nc2array will automatically extend the time range backwards by warmup_days
        time_series_data, static_data, date_range = nc_tool.nc2array(
            config['data_path'],
            station_ids=None,  # Load all stations
            time_range=time_range,  # Original time range - nc2array handles warmup extension
            time_series_variables=all_variables,  # Now only contains forcings for FinetuneLoader
            static_variables=nn_attributes,
            add_coords=True,
            warmup_days=warmup_days  # Let nc2array handle the warmup period
        )
        
        # Now filter stations if needed
        if 'subset_path' in config['observations']:
            subset_path = config['observations']['subset_path']
            gage_id_path = config['observations']['gage_info']
            
            with open(subset_path, 'r') as f:
                selected_basins = json.load(f)
            gage_info = np.load(gage_id_path)
            
            subset_idx = intersect(selected_basins, gage_info)
            
            # Filter the NetCDF data to match the subset
            time_series_data = time_series_data[subset_idx]
            static_data = static_data[subset_idx]
        
        # Handle coordinates (remove lat/lon from static data)
        if static_data.shape[1] >= 2:
            static_data = static_data[:, :-2]

        # Extract target and forcing data
        target_indices = []
        if target:  # Only look for target if target list is not empty
            for target_var in target:
                if target_var in all_variables:
                    target_indices.append(all_variables.index(target_var))
        
        # Extract targets from time_series_data
        target_data = None
        if target_indices:
            target_data = time_series_data[:, :, target_indices]
            log.info(f"Extracted target data with shape: {target_data.shape}")
        
        # Extract forcing data (all non-target variables that are in nn_forcings)
        forcing_indices = []
        for i, var in enumerate(all_variables):
            if var in nn_forcings:
                forcing_indices.append(i)
        
        forcing_data = time_series_data[:, :, forcing_indices] if forcing_indices else None
        
        # Transform to match expected format [time, basins, features]
        if forcing_data is not None:
            forcing_data = np.transpose(forcing_data, (1, 0, 2))
        if target_data is not None:
            target_data = np.transpose(target_data, (1, 0, 2))
        
        return {
            'x_nn': forcing_data.astype(np.float32) if forcing_data is not None else None,
            'c_nn': static_data.astype(np.float32),
            'target': target_data.astype(np.float32) if target_data is not None else None
        }
        
    except Exception as e:
        log.error(f"Error loading neural network data: {str(e)}")
        raise


def split_by_basin(dataset: Dict[str, torch.Tensor], config: Dict[str, Any], 
                  test_config: Dict[str, Any], holdout_index: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split dataset by basin for spatial testing."""
    if not dataset or not test_config or test_config.get('type') != 'spatial':
        return dataset, dataset
    
    try:
        extent = test_config.get('extent')
        holdout_gages = []
        all_basins = []  # Initialize all_basins at the start
        
        if extent == 'PUR':
            huc_regions = test_config.get('huc_regions', [])
            if holdout_index >= len(huc_regions):
                raise ValueError(f"Invalid holdout index: {holdout_index}")
                
            holdout_hucs = huc_regions[holdout_index]
            gage_file = config['observations']['gage_split_file']
            gageinfo = pd.read_csv(gage_file, dtype={"huc": int, "gage": str})
            holdout_hucs_int = [int(huc) for huc in holdout_hucs]
            holdout_gages = gageinfo[gageinfo['huc'].isin(holdout_hucs_int)]['gage'].tolist()
            
        elif extent == 'PUB':
            pub_ids = test_config.get('PUB_ids', [])
            if holdout_index >= len(pub_ids):
                raise ValueError(f"Invalid holdout index: {holdout_index}")
                
            holdout_pub = pub_ids[holdout_index]
            gage_file = config['observations']['gage_split_file']
            gageinfo = pd.read_csv(gage_file, dtype={"PUB_ID": int, "gage": str})
            holdout_gages = gageinfo[gageinfo['PUB_ID'] == holdout_pub]['gage'].tolist()
        
        # Get basin list - try subset_path first, then station_ids as fallback
        basin_file_path = None
        if 'subset_path' in config['observations']:
            basin_file_path = config['observations']['subset_path']
        elif 'station_ids' in config['observations']:
            basin_file_path = config['observations']['station_ids']
        else:
            log.error("Neither subset_path nor station_ids found in config['observations']")
            return None, None
        
        try:
            with open(basin_file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    # JSON-like format
                    content = content.strip('[]')
                    all_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                else:
                    # Plain text format (one basin ID per line or space-separated)
                    if '\n' in content:
                        # Line-separated
                        all_basins = [line.strip() for line in content.split('\n') if line.strip()]
                    else:
                        # Space-separated
                        all_basins = [item.strip() for item in content.split() if item.strip()]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.error(f"Error reading basin file {basin_file_path}: {e}")
            return None, None
        
        if not all_basins:
            log.error("No basins found in subset file")
            return None, None
        
        # Create indices for train/test split
        holdout_gages_set = set(int(str(basin).strip()) for basin in holdout_gages)
        test_indices = []
        train_indices = []
        
        for i, basin in enumerate(all_basins):
            basin_int = int(str(basin).strip())
            if basin_int in holdout_gages_set:
                test_indices.append(i)
            else:
                train_indices.append(i)
        
        if not test_indices:
            raise ValueError("No test basins found!")
        
        # Split the dataset
        train_data = {}
        test_data = {}
        train_indices_tensor = torch.tensor(train_indices, device='cpu')
        test_indices_tensor = torch.tensor(test_indices, device='cpu')
        
        for key, tensor in dataset.items():
            if tensor is None:
                continue
                
            cpu_tensor = tensor.to('cpu')
            
            if len(cpu_tensor.shape) == 3:
                if cpu_tensor.shape[0] == len(all_basins):  # [basins, time, features]
                    train_data[key] = cpu_tensor[train_indices_tensor]
                    test_data[key] = cpu_tensor[test_indices_tensor]
                else:  # [time, basins, features]
                    train_data[key] = cpu_tensor[:, train_indices_tensor]
                    test_data[key] = cpu_tensor[:, test_indices_tensor]
            elif len(cpu_tensor.shape) == 2:  # [basins, features]
                train_data[key] = cpu_tensor[train_indices_tensor]
                test_data[key] = cpu_tensor[test_indices_tensor]
            else:
                train_data[key] = tensor
                test_data[key] = tensor
            
            # Move back to original device
            train_data[key] = train_data[key].to(tensor.device)
            test_data[key] = test_data[key].to(tensor.device)
        
        return train_data, test_data
        
    except Exception as e:
        log.error(f"Error splitting dataset by basin: {e}")
        return None, None


def flow_conversion(c_nn: np.ndarray, target: np.ndarray, target_vars: List[str], 
                   nn_attributes: List[str], config: Dict[str, Any]) -> np.ndarray:
    """Convert hydraulic flow from ft3/s to mm/day."""
    target_copy = target.copy()
    
    for name in ['flow_sim', 'streamflow', 'sf', 'QObs']:
        if name in target_vars:
            target_index = target_vars.index(name)
            target_temp = target_copy[:, :, target_index].copy()
            
            try:
                area_name = config['observations']['area_name']
                basin_area = c_nn[:, nn_attributes.index(area_name)]
                area = np.expand_dims(basin_area, axis=0).repeat(target_temp.shape[0], 0)
                
                converted_flow = ((10 ** 3) * target_temp * 0.0283168 * 3600 * 24 / 
                                 (area * (10 ** 6)))
                target_copy[:, :, target_index] = converted_flow
                
            except (KeyError, ValueError) as e:
                log.warning(f"Could not convert flow units: {e}")
    
    return target_copy


def load_norm_stats(out_path: str, overwrite: bool, x_nn: np.ndarray, c_nn: np.ndarray, 
                   target: np.ndarray, nn_forcings: List[str], nn_attributes: List[str], 
                   target_vars: List[str], log_norm_vars: List[str], config: Dict[str, Any]) -> Dict[str, List[float]]:
    """Load or calculate normalization statistics if necessary."""
    if os.path.isfile(out_path) and not overwrite:
        try:
            with open(out_path, 'r') as f:
                norm_stats = json.load(f)
            # log.info(f"Loaded normalization statistics from {out_path}")
            return norm_stats
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.warning(f"Could not load norm stats: {e}")
    
    # Calculate normalization stats
    return init_norm_stats(x_nn, c_nn, target, nn_forcings, nn_attributes, 
                          target_vars, log_norm_vars, config, out_path)


def init_norm_stats(x_nn: np.ndarray, c_nn: np.ndarray, target: np.ndarray,
                   nn_forcings: List[str], nn_attributes: List[str], target_vars: List[str],
                   log_norm_vars: List[str], config: Dict[str, Any], out_path: str) -> Dict[str, List[float]]:
    """Compile and save calculations of data normalization statistics."""
    stat_dict = {}
    
    # Get basin areas from attributes
    basin_area = get_basin_area(c_nn, nn_attributes, config)

    # Forcing variable stats
    for k, var in enumerate(nn_forcings):
        try:
            if var in log_norm_vars:
                stat_dict[var] = calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = calc_norm_stats(x_nn[:, :, k])
        except Exception as e:
            log.warning(f"Error calculating stats for {var}: {e}")
            stat_dict[var] = [0, 1, 0, 1]

    # Attribute variable stats
    for k, var in enumerate(nn_attributes):
        try:
            stat_dict[var] = calc_norm_stats(c_nn[:, k])
        except Exception as e:
            log.warning(f"Error calculating stats for {var}: {e}")
            stat_dict[var] = [0, 1, 0, 1]

    # Target variable stats
    for i, name in enumerate(target_vars):
        try:
            if name in ['flow_sim', 'streamflow', 'sf']:
                stat_dict[name] = calc_norm_stats(
                    np.swapaxes(target[:, :, i:i+1], 1, 0).copy(),
                    basin_area,
                )
            else:
                stat_dict[name] = calc_norm_stats(
                    np.swapaxes(target[:, :, i:i+1], 1, 0),
                )
        except Exception as e:
            log.warning(f"Error calculating stats for {name}: {e}")
            stat_dict[name] = [0, 1, 0, 1]

    # Save statistics to file
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)
        # log.info(f"Saved normalization statistics to {out_path}")
    except Exception as e:
        log.warning(f"Could not save norm stats: {e}")
    
    return stat_dict


def calc_norm_stats(x: np.ndarray, basin_area: np.ndarray = None) -> List[float]:
    """Calculate statistics for normalization with optional basin area adjustment."""
    # Handle invalid values
    x = x.copy()
    x[x == -999] = np.nan
    if basin_area is not None:
        x[x < 0] = 0

    # Basin area normalization
    if basin_area is not None:
        nd = len(x.shape)
        if nd == 3 and x.shape[2] == 1:
            x = x[:, :, 0]
        temparea = np.tile(basin_area, (1, x.shape[1]))
        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
        x = flow

    # Flatten and exclude NaNs and invalid values
    a = x.flatten()
    if basin_area is None:
        a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
    b = a[(~np.isnan(a)) & (a != -999999)]
    if b.size == 0:
        b = np.array([0])

    # Calculate statistics
    transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
    p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
    mean = np.mean(transformed).astype(float)
    std = np.std(transformed).astype(float)

    return [p10, p90, mean, max(std, 0.001)]


def calc_gamma_stats(x: np.ndarray) -> List[float]:
    """Calculate gamma statistics for streamflow and precipitation data."""
    a = np.swapaxes(x, 1, 0).flatten()
    b = a[(~np.isnan(a))]
    
    if b.size == 0:
        return [0, 1, 0, 1]
    
    b = np.log10(np.sqrt(b) + 0.1)
    p10, p90 = np.percentile(b, [10, 90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)

    return [p10, p90, mean, max(std, 0.001)]


def get_basin_area(c_nn: np.ndarray, nn_attributes: List[str], config: Dict[str, Any]) -> Optional[np.ndarray]:
    """Get basin area from attributes."""
    try:
        area_name = config['observations']['area_name']
        basin_area = c_nn[:, nn_attributes.index(area_name)][:, np.newaxis]
        return basin_area
    except (KeyError, ValueError) as e:
        log.warning(f"No area information found: {e}. Basin area norm will not be applied.")
        return None


def normalize_data(x_nn: NDArray[np.float32], c_nn: NDArray[np.float32], 
                  nn_forcings: List[str], nn_attributes: List[str], 
                  norm_stats: Dict[str, List[float]], log_norm_vars: List[str]) -> NDArray[np.float32]:
    """Normalize data for neural network"""
    x_nn_norm = to_norm(
        np.swapaxes(x_nn, 1, 0).copy(),  # [time, basins, features] -> [basins, time, features]
        nn_forcings,
        norm_stats,
        log_norm_vars
    )
    c_nn_norm = to_norm(
        c_nn,  # [basins, features]
        nn_attributes,
        norm_stats,
        log_norm_vars
    )

    # Remove nans
    x_nn_norm[x_nn_norm != x_nn_norm] = 0
    c_nn_norm[c_nn_norm != c_nn_norm] = 0

    c_nn_norm = np.repeat(
        np.expand_dims(c_nn_norm, 0),
        x_nn_norm.shape[0],
        axis=0
    )

    xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
    del x_nn_norm, c_nn_norm

    return xc_nn_norm


def to_norm(data: NDArray[np.float32], vars: List[str], 
           norm_stats: Dict[str, List[float]], log_norm_vars: List[str]) -> NDArray[np.float32]:
    """Standard data normalization - IDENTICAL to HydroLoader."""
    if not norm_stats:
        log.warning("No normalization statistics available, using identity normalization")
        return data
        
    data_norm = np.zeros(data.shape)

    for k, var in enumerate(vars):
        if var not in norm_stats:
            log.warning(f"No normalization stats for {var}, skipping")
            continue
            
        stat = norm_stats[var]

        if len(data.shape) == 3:
            if var in log_norm_vars:
                data[:, :, k] = np.log10(np.sqrt(np.maximum(data[:, :, k], 0)) + 0.1)
            data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
        elif len(data.shape) == 2:
            if var in log_norm_vars:
                data[:, k] = np.log10(np.sqrt(np.maximum(data[:, k], 0)) + 0.1)
            data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
        else:
            raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

    # NOTE: Should be external, except altering order of first two dims
    # augments normalization...
    if len(data_norm.shape) < 3:
        return data_norm
    else:
        return np.swapaxes(data_norm, 1, 0)  # Back to [time, basins, features]


def from_norm(data_scaled: NDArray[np.float32], vars: List[str], 
             norm_stats: Dict[str, List[float]], log_norm_vars: List[str]) -> NDArray[np.float32]:
    """De-normalize data.
    
    Parameters
    ----------
    data_scaled
        Data to de-normalize.
    vars
        List of variable names in data to de-normalize.
    norm_stats
        Normalization statistics.
    log_norm_vars
        Variables that use log normalization.
    
    Returns
    -------
    NDArray[np.float32]
        De-normalized data.
    """
    data = np.zeros(data_scaled.shape)

    for k, var in enumerate(vars):
        stat = norm_stats[var]
        if len(data_scaled.shape) == 3:
            data[:, :, k] = data_scaled[:, :, k] * stat[3] + stat[2]
            if var in log_norm_vars:
                data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
        elif len(data_scaled.shape) == 2:
            data[:, k] = data_scaled[:, k] * stat[3] + stat[2]
            if var in log_norm_vars:
                data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
        else:
            raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

    if len(data.shape) < 3:
        return data
    else:
        return np.swapaxes(data, 1, 0)


def to_tensor(data: np.ndarray, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor with specified device and dtype."""
    if data is None:
        return None
    return torch.tensor(data, dtype=dtype, device=device)