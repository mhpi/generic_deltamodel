import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

from dmg.core.data.data import intersect
from dmg.core.data.loaders.load_nc import NetCDFDataset

log = logging.getLogger(__name__)


def load_nn_data(
    config: Dict[str, Any],
    scope: str,
    t_range: Dict[str, str],
    nn_forcings: List[str],
    nn_attributes: List[str],
    target: List[str],
    device: str,
    nc_tool: NetCDFDataset,
) -> Dict[str, np.ndarray]:
    """Load and process neural network data from NetCDF."""
    time_range = [t_range['start'].replace('/', '-'), t_range['end'].replace('/', '-')]
    warmup_days = config['model']['warmup']

    try:
        all_variables = nn_forcings.copy()

        if target:
            for target_var in target:
                if target_var not in all_variables:
                    all_variables.append(target_var)

        time_series_data, static_data, date_range, row_station_ids = nc_tool.nc2array(
            config['data_path'],
            station_ids=None,
            time_range=time_range,
            time_series_variables=all_variables,
            static_variables=nn_attributes,
            add_coords=True,
            warmup_days=warmup_days,
        )

        if config['observations'].get('subset_path') is not None:
            subset_path = config['observations']['subset_path']
            gage_id_path = config['observations']['gage_info']

            with open(subset_path) as f:
                selected_basins = json.load(f)
            gage_info = np.load(gage_id_path)

            subset_idx = intersect(selected_basins, gage_info)
            time_series_data = time_series_data[subset_idx]
            static_data = static_data[subset_idx]
            if row_station_ids is not None:
                row_station_ids = row_station_ids[subset_idx]

        # Remove lat/lon coords appended to static data
        if static_data.shape[1] >= 2:
            static_data = static_data[:, :-2]

        target_indices = []
        if target:
            for target_var in target:
                if target_var in all_variables:
                    target_indices.append(all_variables.index(target_var))

        target_data = None
        if target_indices:
            target_data = time_series_data[:, :, target_indices]
            log.info(f"Extracted target data with shape: {target_data.shape}")

        forcing_indices = [
            i for i, var in enumerate(all_variables) if var in nn_forcings
        ]
        forcing_data = (
            time_series_data[:, :, forcing_indices] if forcing_indices else None
        )

        # Transform to [time, basins, features]
        if forcing_data is not None:
            forcing_data = np.transpose(forcing_data, (1, 0, 2))
        if target_data is not None:
            target_data = np.transpose(target_data, (1, 0, 2))

        return {
            'x_nn': forcing_data.astype(np.float32)
            if forcing_data is not None
            else None,
            'c_nn': static_data.astype(np.float32),
            'target': target_data.astype(np.float32)
            if target_data is not None
            else None,
            'station_ids': row_station_ids,
            'date_range': date_range,
        }

    except Exception as e:
        log.error(f"Error loading neural network data: {str(e)}")
        raise


def flow_conversion(
    c_nn: np.ndarray,
    target: np.ndarray,
    target_vars: List[str],
    nn_attributes: List[str],
    config: Dict[str, Any],
) -> np.ndarray:
    """Convert hydraulic flow from ft3/s to mm/day."""
    target_copy = target.copy()

    for name in ['flow_sim', 'streamflow', 'sf', 'QObs']:
        if name in target_vars:
            target_index = target_vars.index(name)
            target_temp = target_copy[:, :, target_index].copy()

            try:
                area_name = config['observations']['area_name']
                basin_area = c_nn[:, nn_attributes.index(area_name)]
                area = np.expand_dims(basin_area, axis=0).repeat(
                    target_temp.shape[0], 0
                )

                converted_flow = (
                    (10**3) * target_temp * 0.0283168 * 3600 * 24 / (area * (10**6))
                )
                target_copy[:, :, target_index] = converted_flow

            except (KeyError, ValueError) as e:
                log.warning(f"Could not convert flow units: {e}")

    return target_copy


def load_norm_stats(
    out_path: str,
    overwrite: bool,
    x_nn: np.ndarray,
    c_nn: np.ndarray,
    target: np.ndarray,
    nn_forcings: List[str],
    nn_attributes: List[str],
    target_vars: List[str],
    log_norm_vars: List[str],
    config: Dict[str, Any],
) -> Dict[str, List[float]]:
    """Load or calculate normalization statistics if necessary."""
    if os.path.isfile(out_path) and not overwrite:
        try:
            with open(out_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            log.warning(f"Could not load norm stats: {e}")

    return init_norm_stats(
        x_nn,
        c_nn,
        target,
        nn_forcings,
        nn_attributes,
        target_vars,
        log_norm_vars,
        config,
        out_path,
    )


def init_norm_stats(
    x_nn: np.ndarray,
    c_nn: np.ndarray,
    target: np.ndarray,
    nn_forcings: List[str],
    nn_attributes: List[str],
    target_vars: List[str],
    log_norm_vars: List[str],
    config: Dict[str, Any],
    out_path: str,
) -> Dict[str, List[float]]:
    """Compile and save calculations of data normalization statistics."""
    stat_dict = {}

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
                    np.swapaxes(target[:, :, i : i + 1], 1, 0).copy(),
                    basin_area,
                )
            else:
                stat_dict[name] = calc_norm_stats(
                    np.swapaxes(target[:, :, i : i + 1], 1, 0),
                )
        except Exception as e:
            log.warning(f"Error calculating stats for {name}: {e}")
            stat_dict[name] = [0, 1, 0, 1]

    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)
    except Exception as e:
        log.warning(f"Could not save norm stats: {e}")

    return stat_dict


def calc_norm_stats(
    x: np.ndarray,
    basin_area: np.ndarray = None,
) -> List[float]:
    """Calculate statistics for normalization with optional basin area adjustment."""
    x = x.copy()
    x[x == -999] = np.nan
    if basin_area is not None:
        x[x < 0] = 0

    if basin_area is not None:
        nd = len(x.shape)
        if nd == 3 and x.shape[2] == 1:
            x = x[:, :, 0]
        temparea = np.tile(basin_area, (1, x.shape[1]))
        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10**6)) * 10**3
        x = flow

    a = x.flatten()
    if basin_area is None:
        a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
    b = a[(~np.isnan(a)) & (a != -999999)]
    if b.size == 0:
        b = np.array([0])

    transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
    p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
    mean = float(np.mean(transformed, dtype=np.float64))
    std = float(np.std(transformed, dtype=np.float64))

    return [p10, p90, mean, max(std, 0.001)]


def calc_gamma_stats(x: np.ndarray) -> List[float]:
    """Calculate gamma statistics for streamflow and precipitation data."""
    a = np.swapaxes(x, 1, 0).flatten()
    b = a[~np.isnan(a)]

    if b.size == 0:
        return [0, 1, 0, 1]

    b = np.log10(np.sqrt(b) + 0.1)
    p10, p90 = np.percentile(b, [10, 90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)

    return [p10, p90, mean, max(std, 0.001)]


def get_basin_area(
    c_nn: np.ndarray,
    nn_attributes: List[str],
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Get basin area from attributes."""
    try:
        area_name = config['observations']['area_name']
        return c_nn[:, nn_attributes.index(area_name)][:, np.newaxis]
    except (KeyError, ValueError) as e:
        log.warning(
            f"No area information found: {e}. Basin area norm will not be applied."
        )
        return None


def normalize_data(
    x_nn: NDArray[np.float32],
    c_nn: NDArray[np.float32],
    nn_forcings: List[str],
    nn_attributes: List[str],
    norm_stats: Dict[str, List[float]],
    log_norm_vars: List[str],
) -> NDArray[np.float32]:
    """Normalize and concatenate dynamic and static inputs for a neural network."""
    x_nn_norm = to_norm(
        np.swapaxes(
            x_nn, 1, 0
        ).copy(),  # [time, basins, features] -> [basins, time, features]
        nn_forcings,
        norm_stats,
        log_norm_vars,
    )
    c_nn_norm = to_norm(
        c_nn,
        nn_attributes,
        norm_stats,
        log_norm_vars,
    )

    x_nn_norm[x_nn_norm != x_nn_norm] = 0
    c_nn_norm[c_nn_norm != c_nn_norm] = 0

    c_nn_norm = np.repeat(np.expand_dims(c_nn_norm, 0), x_nn_norm.shape[0], axis=0)
    xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
    del x_nn_norm, c_nn_norm

    return xc_nn_norm


def normalize_data_split(
    x_nn: NDArray[np.float32],
    c_nn: NDArray[np.float32],
    nn_forcings: List[str],
    nn_attributes: List[str],
    norm_stats: Dict[str, List[float]],
    log_norm_vars: List[str],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Normalize dynamic and static inputs separately without concatenating.

    Parameters
    ----------
    x_nn
        Dynamic forcings of shape [time, basins, n_forcings].
    c_nn
        Static attributes of shape [basins, n_static].

    Returns
    -------
    Tuple of (x_nn_norm, c_nn_norm) with the same leading shapes as inputs.
    """
    x_nn_norm = to_norm(
        np.swapaxes(x_nn, 1, 0).copy(),  # -> [basins, time, features]
        nn_forcings,
        norm_stats,
        log_norm_vars,
    )  # to_norm swaps back -> [time, basins, features]

    c_nn_norm = to_norm(
        c_nn,
        nn_attributes,
        norm_stats,
        log_norm_vars,
    )

    x_nn_norm[x_nn_norm != x_nn_norm] = 0
    c_nn_norm[c_nn_norm != c_nn_norm] = 0

    return x_nn_norm.astype(np.float32, copy=False), c_nn_norm.astype(
        np.float32, copy=False
    )


def to_norm(
    data: NDArray[np.float32],
    vars: List[str],
    norm_stats: Dict[str, List[float]],
    log_norm_vars: List[str],
) -> NDArray[np.float32]:
    """Normalize data using Gaussian or log-Gaussian statistics.

    Parameters
    ----------
    data
        Array of shape [basins, ...] or [basins, time, features] to normalize.
        For 3-D input the first axis is basins; axes are swapped back before return.
    vars
        Ordered list of variable names corresponding to the last axis of data.
    norm_stats
        dictionary mapping variable names to [p10, p90, mean, std].
    log_norm_vars
        Variables that receive a log10(sqrt(x) + 0.1) transform before scaling.

    Returns
    -------
    NDArray[np.float32]
        Normalized array.  3-D output is returned as [time, basins, features].
    """
    if not norm_stats:
        log.warning(
            "No normalization statistics available, using identity normalization"
        )
        return data

    data = np.asarray(data, dtype=np.float32)
    data_norm = np.zeros_like(data, dtype=np.float32)

    for k, var in enumerate(vars):
        if var not in norm_stats:
            log.warning(f"No normalization stats for {var}, skipping")
            continue

        stat = norm_stats[var]
        mean, std = stat[2], stat[3]

        if len(data.shape) == 3:
            if var in log_norm_vars:
                data[:, :, k] = np.log10(np.sqrt(np.maximum(data[:, :, k], 0)) + 0.1)
            data_norm[:, :, k] = (data[:, :, k] - mean) / std
        elif len(data.shape) == 2:
            if var in log_norm_vars:
                data[:, k] = np.log10(np.sqrt(np.maximum(data[:, k], 0)) + 0.1)
            data_norm[:, k] = (data[:, k] - mean) / std
        else:
            raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

    if len(data_norm.shape) < 3:
        return data_norm
    return np.swapaxes(data_norm, 1, 0)  # Back to [time, basins, features]


def from_norm(
    data_scaled: NDArray[np.float32],
    vars: List[str],
    norm_stats: Dict[str, List[float]],
    log_norm_vars: List[str],
) -> NDArray[np.float32]:
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
    data_scaled = np.asarray(data_scaled, dtype=np.float32)
    data = np.zeros_like(data_scaled, dtype=np.float32)

    for k, var in enumerate(vars):
        stat = norm_stats[var]
        mean, std = stat[2], stat[3]

        if len(data_scaled.shape) == 3:
            denormed = data_scaled[:, :, k] * std + mean
            data[:, :, k] = (
                (np.power(10.0, denormed) - 0.1) ** 2
                if var in log_norm_vars
                else denormed
            )
        elif len(data_scaled.shape) == 2:
            denormed = data_scaled[:, k] * std + mean
            data[:, k] = (
                (np.power(10.0, denormed) - 0.1) ** 2
                if var in log_norm_vars
                else denormed
            )
        else:
            raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

    if len(data.shape) < 3:
        return data
    return np.swapaxes(data, 1, 0)


def to_tensor(
    data: np.ndarray,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert numpy array to torch tensor with specified device and dtype."""
    if data is None:
        return None
    return torch.tensor(data, dtype=dtype, device=device)
