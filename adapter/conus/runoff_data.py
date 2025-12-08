
import xarray as xr
from pathlib import Path
import numpy as np
from typing import Union
import pandas as pd


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


#-------------------------------------- CAMELS ---------------------------------------#

### runoff data
save_path = '/nfs/data/wby5078/Camels_hourly/'
runoff = xr.open_dataset(Path('/data/wby5078/raw_data/camels_houly/') / 'forcing_freq_1h.nc')
gauges = runoff['gauge'].data
runoff_data = runoff['runoff'].data / 24
times = np.arange(runoff_data.shape[1])
units = {'runoff': 'mm/hour'}
ds_attrs = {'period': f'1979-01-01T13:00:00 to 2019-03-14T12:00:00 UTC0'}
save_nc_file(data={'runoff': runoff_data}, units=units, gauges=gauges, filename=Path(save_path) / 'runoff.nc', times=times, ds_attrs=ds_attrs)


### gage info
df = pd.read_csv(Path('/nfs/data/wby5078/Camels_hourly/') / 'camels_topo.txt', sep=';')
df['STAID'] = df['gauge_id']
df['DRAIN_SQKM'] = df['area_gages2']
df.to_csv(Path(save_path) / 'gage_info.csv', index=False)

#------------------------------------- CAMELSH ---------------------------------------#

import geopandas as gpd

df_avail = pd.read_csv(Path('/nfs/data/wby5078/CAMELSH/') / 'info.csv', dtype={'STAID': str})
df_avail[df_avail['data_availability [hrs]'] > 0]
pts = gpd.read_file(Path('/nfs/data/wby5078/CONUS3000/gagesII_9322_sept30_2011/') / 'gagesII_9322_sept30_2011.shp')
pts = pts[pts['STAID'].isin(df_avail.loc[df_avail['data_availability [hrs]'] > 0, 'STAID'])].reset_index(drop=True)
pts.dtypes
pts['DRAIN_SQKM'].describe()
np.max((pts.geometry.to_crs(epsg=4326).x - pts['LNG_GAGE']).abs())
np.max((pts.geometry.to_crs(epsg=4326).y - pts['LAT_GAGE']).abs())




