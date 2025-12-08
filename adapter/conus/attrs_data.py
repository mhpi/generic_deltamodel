
'''
Do not need this!!! hydrofabric_attrs.nc already has all basin attributes for hydrofabric.

'''


import zarr
import xarray as xr
from pathlib import Path
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def get_gauge_indexes(gauge_array: np.ndarray, gauges: list) -> np.ndarray:
    """Get the indexes of gauge_array using the order of gauges."""
    df = pd.DataFrame({'gauge': gauge_array, 'local_ind': np.arange(len(gauge_array))})
    df = df.merge(pd.DataFrame({'gauge': gauges, 'global_ind': np.arange(len(gauges))}))
    return df.sort_values(by='global_ind')['local_ind'].values


path_hf_daily = '/nfs/data/wby5078/CONUS3000/subzones_1980_2019'
hf_data = zarr.open_group(path_hf_daily, mode="r")


path_topo = '/nfs/data/wby5078/Camels_hourly/gage_hf.json'
gage_hf = json.load(open(path_topo, 'r'))
sorted_basins = np.sort(gage_hf['nodes'])
path_save = ''

# Read basin attributes
basin_attrs = []
data_dict = {}
for group in tqdm(list(hf_data.group_keys())):
    ds_attrs = xr.open_zarr(path_hf_daily, group=f'{group}/attrs', consolidated=True)
    basin_ids = pd.to_numeric(pd.Series(ds_attrs['divide_id'].to_numpy()).str.replace('cat-', '', regex=False), errors='coerce').astype(int)
    idx = get_gauge_indexes(basin_ids, sorted_basins)
    basin_ids = basin_ids[idx].to_numpy()
    basin_attrs.append(basin_ids)
    for var in ds_attrs.data_vars:
        if var not in data_dict:
            data_dict[var] = []
        data_dict[var].append(ds_attrs[var].values[idx])
basin_attrs = np.concatenate(basin_attrs, axis=0)
for var in data_dict:
    data_dict[var] = np.concatenate(data_dict[var], axis=0)
# Remove duplicates
no_dup_idx = np.sort(np.unique(basin_attrs, return_index=True)[1])
basin_attrs = basin_attrs[no_dup_idx]
for var in data_dict:
    data_dict[var] = data_dict[var][no_dup_idx]



gage = np.array([f'cat-{bid}' for bid in basin_attrs], dtype=object)
attrs = list(data_dict.keys())
N = len(gage)
K = len(attrs)
M = np.vstack([np.asarray(data_dict[k], dtype=np.float32) for k in attrs])  # (K, N)
da = xr.DataArray(
    M,
    dims=("attr", "gage"),
    coords={"attr": np.asarray(attrs, dtype=object), "gage": np.asarray(gage, dtype=object)},
    name="__xarray_dataarray_variable__"  # this becomes your data variable name
)
ds = da.to_dataset()
ds.to_netcdf(path_save, format="NETCDF4", engine="netcdf4")


