
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from typing import Union
import zarr


def get_gauge_indexes(gauge_array: np.ndarray, gauges: list) -> np.ndarray:
    df = pd.DataFrame({'gauge': gauge_array, 'local_ind': np.arange(len(gauge_array))})
    df = df.merge(pd.DataFrame({'gauge': gauges, 'global_ind': np.arange(len(gauges))}))
    return df.sort_values(by='global_ind')['local_ind'].values


def cfs_to_mm_day(cfs: np.ndarray, area_sq_km: np.ndarray) -> np.ndarray:
    if len(cfs.shape) == 2:
        area = area_sq_km[:, None]
    else:
        area = area_sq_km
    return cfs * 2.446576 / area


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
                 encoding={var: {"zlib": True, "complevel": 4} for var in ds.data_vars})


save_path = '/nfs/data/wby5078/Texas_floods/'
texas_gauge_info = '/data/wby5078/raw_data/texas_raw/info_stations_TX_595.csv'
texas_runoff_utc = '/data/wby5078/raw_data/texas_raw/hourly_data/discharge_processed_utc0'
runoff_start = '1991-10-01 00:00:00'  # utc0
runoff_end = '2025-07-17 00:00:00'

gauge_info = pd.read_csv(Path(texas_gauge_info))
gauge_info['gauge_id'] = gauge_info['STAID'].astype(str).str.zfill(8)
zarr_runoff_texas = zarr.open_group(Path(texas_runoff_utc))

gauges = sorted(gauge_info[gauge_info['gauge_id'].isin(zarr_runoff_texas['GAGEID'][:])]['gauge_id'])
gauge_indexes = get_gauge_indexes(zarr_runoff_texas['GAGEID'][:], gauges)
runoff = zarr_runoff_texas['observation'][:, gauge_indexes].T
gauge_indexes = get_gauge_indexes(gauge_info['gauge_id'].values, gauges)
runoff = cfs_to_mm_day(runoff, gauge_info.loc[gauge_indexes, 'DRAIN_SQKM'].values)
runoff = runoff / 24  # mm/day to mm/hour

units = {'runoff': 'mm/hour'}
times = np.arange(runoff.shape[1])
ds_attrs = {'period': f'{runoff_start} to {runoff_end} UTC0'}
save_nc_file(data={'runoff': runoff}, units=units, gauges=gauges, filename=Path(save_path) / 'runoff.nc', times=times, ds_attrs=ds_attrs)



