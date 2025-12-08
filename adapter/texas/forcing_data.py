
import pandas as pd
from pathlib import Path
import time
import polars as pl
import numpy as np
import xarray as xr
from typing import Union


def get_gauge_indexes(gauge_array: np.ndarray, gauges: list) -> np.ndarray:
    """Get the indexes of gauge_array using the order of gauges."""
    df = pd.DataFrame({'gauge': gauge_array, 'local_ind': np.arange(len(gauge_array))})
    df = df.merge(pd.DataFrame({'gauge': gauges, 'global_ind': np.arange(len(gauges))}))
    return df.sort_values(by='global_ind')['local_ind'].values


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


#----------------------------------------- Texas -----------------------------------------#

def create_distributed_forcing_nc(forcing_path: str, save_path: str, year: int, gauges: np.ndarray, pet: np.ndarray, all_times: pd.DatetimeIndex,
                                  nonnan_indexes1: np.ndarray, nonnan_indexes2: np.ndarray) -> None:
    start = time.time()
    df_precip = pl.read_csv(
        Path(forcing_path) / f'{year}' / 'RAINRATE.csv',
        has_header=False,
        infer_schema_length=1000,
        use_pyarrow=False,
        ignore_errors=True,
    )
    P = df_precip.to_numpy()
    if year <= 2012:
        P = P[nonnan_indexes1, :] * 3600  # mm/hr
    else:
        P = P[nonnan_indexes2, :] * 3600

    df_temp = pl.read_csv(
        Path(forcing_path) / f'{year}' / 'T2D.csv',
        has_header=False,
        infer_schema_length=1000,
        use_pyarrow=False,
        ignore_errors=True,
    )
    Temp = df_temp.to_numpy()
    if year <= 2012:
        Temp = Temp[nonnan_indexes1, :] - 273.15  # C
    else:
        Temp = Temp[nonnan_indexes2, :] - 273.15  # C

    time_indexes = np.where(all_times.year == year)[0]
    PET = pet[time_indexes, :][:, nonnan_indexes2].T  # mm/hr

    assert P.shape == PET.shape
    assert Temp.shape == PET.shape
    assert np.isnan(P).sum() == 0
    assert np.isnan(Temp).sum() == 0
    assert np.isnan(PET).sum() == 0

    data = {'P': P.astype(np.float32), 'T': Temp.astype(np.float32), 'PET': PET.astype(np.float32)}
    units = {'P': 'mm/hr', 'T': 'C', 'PET': 'mm/hr'}
    times = np.arange(P.shape[1])
    filename = Path(save_path) / f'forcing_{year}.nc'
    save_nc_file(data=data, units=units, gauges=gauges, filename=filename, times=times)
    print(f"Saved forcing for year {year} in {time.time() - start:.2f} seconds.")


forcing_path = '/nfs/data/hjj5218/data/GAGES_2/TX_flood/hf_TX_index_hour/'
save_path = '/nfs/data/wby5078/Texas_floods/forcing/'
pet = np.load(Path(forcing_path) / 'pet_hour.npy', mmap_mode="r")
pet_start_time = '1980-01-01 00:00:00'
all_times = pd.date_range(start=pet_start_time, periods=pet.shape[0], freq='h')
df_info1 = pd.read_csv(Path('/nfs/data/wby5078/Texas_floods/info_hf_1991-2012.csv'))
df_info2 = pd.read_csv(Path('/nfs/data/wby5078/Texas_floods/info_hf_2013-2025.csv'))
nonnan_indexes2 = np.where(~np.isnan(pet[0, :]))[0]
gauges = df_info2['COMID'].to_numpy()[nonnan_indexes2]
nonnan_indexes1 = get_gauge_indexes(df_info1['comid'].values, gauges)




def _run_year(y: int):
    """
    Thin wrapper that calls your existing function using module-level globals.
    With 'spawn', this module is re-imported in each child, so 'pet' (mmap),
    'all_times', etc. are re-opened inside the child — no giant pickles.
    """
    return create_distributed_forcing_nc(
        forcing_path=forcing_path,
        save_path=save_path,
        year=int(y),
        gauges=gauges,
        pet=pet,
        all_times=all_times,
        nonnan_indexes1=nonnan_indexes1,
        nonnan_indexes2=nonnan_indexes2,
    )

if __name__ == "__main__":

    for y in range(1991, 2013):
        print(f"Starting year {y}")
        _run_year(y)

