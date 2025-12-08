
import zarr
import xarray as xr
from pathlib import Path
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from typing import Union


def get_gauge_indexes(gauge_array: np.ndarray, gauges: list) -> np.ndarray:
    """Get the indexes of gauge_array using the order of gauges."""
    df = pd.DataFrame({'gauge': gauge_array, 'local_ind': np.arange(len(gauge_array))})
    df = df.merge(pd.DataFrame({'gauge': gauges, 'global_ind': np.arange(len(gauges))}))
    return df.sort_values(by='global_ind')['local_ind'].values


def spatial_nearest_fillna(data_basins: np.ndarray, missing_basins: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Find the nearest basin (hydrofabric id) data to fill missing basins, return the new basins and data."""
    dist = np.abs(data_basins - missing_basins[:, None]).astype(np.float32)
    dist[dist == 0] = np.inf
    nearest_idx = np.argmin(dist, axis=1)
    filled_data = data[nearest_idx]
    return np.concatenate([data_basins, missing_basins], axis=0), np.concatenate([data, filled_data], axis=0)


def temporal_nearest_fillna(data):
    """Fill NaN values in a 2D numpy array using the nearest non-NaN value in the same row.
    Test:
    a = np.random.rand(3, 4)
    a[[0, 0], [0, 1]] = np.nan
    a[[1, 1], [1, 2]] = np.nan
    a[[2, 2], [1, 3]] = np.nan
    temporal_nearest_fillna(a)
    """
    df = pd.DataFrame(data)
    df_ffill = df.ffill(axis=1)   # forward fill
    df_bfill = df.bfill(axis=1)   # backward fill
    return np.nanmean(np.stack([df_ffill.values, df_bfill.values]), axis=0)


def save_nc_file(data: dict[str, np.ndarray], units: dict[str, str], gauges: np.ndarray, filename: Union[str, Path],
                 times: np.ndarray = None, ds_attrs: dict = None) -> None:
    xr_dict = {}
    for key, value in data.items():
        if times is None:
            xr_dict[key] = xr.DataArray(value, dims=("gauge"),
                                        coords={"gauge": gauges},
                                        name=key,
                                        attrs={"units": units[key], "long_name": key})
        else:
            xr_dict[key] = xr.DataArray(value, dims=("gauge", "time"),
                                        coords={"gauge": gauges, "time": times},
                                        name=key,
                                        attrs={"units": units[key], "long_name": key})
    ds = xr.Dataset(xr_dict)
    ds.attrs = ds_attrs if ds_attrs is not None else {}
    ds.to_netcdf(filename, format="NETCDF4", engine="netcdf4",
                 encoding={var: {"zlib": False, "complevel": 0, "shuffle": False, "dtype": "float32"} for var in ds.data_vars})


path_hf_daily = '/nfs/data/wby5078/CONUS3000/subzones_1980_2019'
hf_data = zarr.open_group(path_hf_daily, mode="r")


### 1. CAMELS
path_aorc = '/nfs/data/wby5078/Camels_hourly/AORC/'
path_save = '/nfs/data/wby5078/Camels_hourly/forcing/'
path_topo = '/nfs/data/wby5078/Camels_hourly/gage_hf.json'
gage_hf = json.load(open(path_topo, 'r'))
sorted_basins = np.sort(gage_hf['nodes'])


def run_one_year(year: int):
    ### reac hourly precipitation and fill na
    ds_hour = xr.open_zarr(Path(path_aorc) / str(year) / 'APCP_surface.zarr', consolidated=True)
    P_hour = ds_hour['APCP_surface'].to_numpy()
    basin_hour = ds_hour['divide_id'].to_numpy()
    fill_basins = np.setdiff1d(gage_hf['nodes'], basin_hour)
    basin_hour, P_hour = spatial_nearest_fillna(basin_hour, fill_basins, P_hour)
    P_hour = temporal_nearest_fillna(P_hour)

    ### read daily focing
    start = time.time()
    Temp = []
    PET = []
    basin_forcing = []
    for group in tqdm(list(hf_data.group_keys())):
        ds_forcing = xr.open_zarr(path_hf_daily, group=group, consolidated=True)
        time_start = pd.to_datetime(f'{year}-01-01')
        time_end = pd.to_datetime(f'{year}-12-31')
        time_idx = (pd.to_datetime(ds_forcing['time']) >= time_start) & (pd.to_datetime(ds_forcing['time']) <= time_end)
        basin_ids = pd.to_numeric(pd.Series(ds_forcing['divide_id'].to_numpy()).str.replace('cat-', '', regex=False), errors='coerce').astype(int)
        basin_ids = basin_ids[basin_ids.isin(gage_hf['nodes'])]
        idx = basin_ids.index.to_numpy()
        basin_ids = basin_ids.to_numpy()
        Temp.append(ds_forcing['Temp'][idx, time_idx].to_numpy())
        PET.append(ds_forcing['PET'][idx, time_idx].to_numpy())
        basin_forcing.append(basin_ids)
    Temp = np.concatenate(Temp, axis=0)
    PET = np.concatenate(PET, axis=0)
    basin_forcing = np.concatenate(basin_forcing, axis=0)
    print(time.time() - start)
    # drop duplicated in different zone chunks
    no_dup_idx = np.sort(np.unique(basin_forcing, return_index=True)[1])
    basin_forcing = basin_forcing[no_dup_idx]
    Temp = Temp[no_dup_idx]
    PET = PET[no_dup_idx]
    # fillna for Temp and PET
    nan_mask = np.any(np.isnan(Temp), axis=1)
    fill_basins = basin_forcing[nan_mask]
    Temp_ids, Temp_filled = spatial_nearest_fillna(basin_forcing[~nan_mask], fill_basins, Temp[~nan_mask])
    Temp_filled = temporal_nearest_fillna(Temp_filled)
    nan_mask = np.any(np.isnan(PET), axis=1)
    fill_basins = basin_forcing[nan_mask]
    PET_ids, PET_filled = spatial_nearest_fillna(basin_forcing[~nan_mask], fill_basins, PET[~nan_mask])
    PET_filled = temporal_nearest_fillna(PET_filled)

    ### align basin order
    P_hour = P_hour[get_gauge_indexes(basin_hour, sorted_basins)]
    Temp_filled = Temp_filled[get_gauge_indexes(Temp_ids, sorted_basins)]
    PET_filled = PET_filled[get_gauge_indexes(PET_ids, sorted_basins)]

    data = {
        'P': P_hour.astype(np.float32),
        'T': np.tile(Temp_filled.astype(np.float32)[:, :, None], (1, 1, 24)).reshape(Temp_filled.shape[0], -1),
        'PET': np.tile((PET_filled / 24).astype(np.float32)[:, :, None], (1, 1, 24)).reshape(PET_filled.shape[0], -1),
    }
    units = {'P': 'mm/hr', 'T': 'C', 'PET': 'mm/hr'}
    times = np.arange(P_hour.shape[1])
    filename = Path(path_save) / f'forcing_{year}.nc'
    save_nc_file(data=data, units=units, gauges=sorted_basins, filename=filename, times=times)



for year in range(1990, 2019):
    run_one_year(year)


