
import os

os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import time
import s3fs
import pickle
import argparse

import xarray as xr
import numpy as np
import pandas as pd
import json
import gc
from pathlib import Path
import resource, sys


from multiprocessing.pool import ThreadPool
import dask.delayed
dask.config.set(pool=ThreadPool(24))



def get_peak_rss_bytes():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak = ru.ru_maxrss
    if sys.platform == "darwin":
        return int(peak)  # bytes on macOS
    else:
        return int(peak) * 1024  # kilobytes on Linux -> bytes


parser = argparse.ArgumentParser(description="Process NOAA data by year.")
parser.add_argument('--num_year', type=int, default=2000, help='Year of the dataset to process.')
parser.add_argument('--basin_chunk', type=int, default=2000, help='Chunk size along pt after isel')
parser.add_argument('--save_chunk', type=int, default=500, help='Spatial chunk size to save')
args = parser.parse_args()
num_year = args.num_year
basin_chunk = args.basin_chunk
save_chunk = args.save_chunk



grid_thres = 100000000
# num_year = 2003
# basin_chunk = 10
# save_chunk = 500
index_file = rf"/nfs/data/wby5078/CONUS3000/AORC/index_dict.pkl"
output_dir = rf"/nfs/data/wby5078/CONUS3000/AORC/{num_year}"

with open(index_file, "rb") as f:
    index_dict = pickle.load(f)
station_ids = index_dict['station_ids']  # caution: array(['cat-5', 'cat-1e+05', 'cat-5e+05', 'cat-8e+05'], dtype=object)!!
station_ids = pd.to_numeric(pd.Series(station_ids).str.replace('cat-', '', regex=False), errors='coerce').astype(int)

# with open(os.path.join('/nfs/data/wby5078/Camels_hourly', "gage_hf.json"), "r") as f:
#     gage_hf = json.load(f)
# all_basins = gage_hf['nodes']
num_grids = pd.Series([len(row) for row in index_dict["row_list"]])
all_basins = station_ids[num_grids[num_grids <= grid_thres].index].reset_index(drop=True).tolist()
all_basins = all_basins[:4000]


# basin_chunk = 10000
# grid_len = []
# for basin_start in range(0, len(all_basins), basin_chunk):
#     basin_end = min(basin_start + basin_chunk, len(all_basins))
#     selected_basins = all_basins[basin_start:basin_end]
#     idx = station_ids[station_ids.isin(selected_basins)].index.values
#     row_list = [index_dict["row_list"][i] for i in idx]
#     row_list = [item for sublist in row_list for item in sublist]
#     grid_len.append(len(row_list))
# pd.Series(grid_len).describe()


variable_list = ["APCP_surface"]
# "APCP_surface", "DSWRF_surface", "TMP_2maboveground", "DLWRF_surface", "PRES_surface", "SPFH_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground"


def groupby_np(data, bins=None, groups=None, interval=None, mean=True):
    """
    group datasets with index by first dimension
    e.g.
    datasets = np.array([[0,1,2],
                    [30,4,5],
                    [6,7,8],
                    [9,10,11]])
    groups = [0,1,0,3]

    equal to ([0,1,2]+[6,7,8])/2
            ([3,4,5])/1
            ([9,10,11])/1

    result:
            [ 3.,  4.,  5.],
            [30.,  4.,  5.],
            [ 9., 10., 11.]]

    interval = [2,3,4]
    bins = [2,5,9]: 0-2, 2-5, 5-9
    """
    if not interval is None:
        bins = np.cumsum(interval)
        bins = np.insert(bins, 0, 0)

    if not groups is None:
        _ndx = np.argsort(groups)
        _id, _pos, g_count = np.unique(groups[_ndx], return_index=True, return_counts=True)
        bins = np.cumsum(g_count)
        bins = np.insert(bins, 0, 0)
        data = data[_ndx]

    bins = bins[:-1]

    # processing nan
    mask = np.isnan(data)
    data[mask] = 0
    g_count = np.add.reduceat(~mask, bins)

    g_sum = np.add.reduceat(data, bins, axis=0)
    g_mean = g_sum / g_count

    if mean:
        return g_mean
    else:
        return g_sum



if not os.path.exists(output_dir):
    os.makedirs(output_dir)
_s3 = s3fs.S3FileSystem(anon=True)
files = [s3fs.S3Map(
    root=f's3://noaa-nws-aorc-v1-1-1km/{num_year}.zarr',
    s3=_s3, check=False, )
]


for var_name in variable_list:
    # var_name = "APCP_surface"
    print("processing: ", var_name)
    start_time = time.time()

    ds = xr.open_mfdataset(files, engine="zarr", parallel=True, consolidated=True)
    ds = ds.sortby('latitude', ascending=False)
    num_time = ds.time.size
    da_var = ds[var_name]  # shape: (time, lat, lon)

    for basin_start in range(0, len(all_basins), basin_chunk):
        basin_end = min(basin_start + basin_chunk, len(all_basins))
        selected_basins = all_basins[basin_start:basin_end]
        idx = station_ids[station_ids.isin(selected_basins)].index.values

        col_list = [index_dict["col_list"][i] for i in idx]
        row_list = [index_dict["row_list"][i] for i in idx]
        index_list_num = [len(row) for row in col_list]
        row_list = [item for sublist in row_list for item in sublist]
        col_list = [item for sublist in col_list for item in sublist]
        print(f"basin_start {basin_start}, n grids: {len(row_list)}")

        # start = time.time()
        row = xr.DataArray(np.asarray(row_list, dtype=np.int64), dims="pt")
        col = xr.DataArray(np.asarray(col_list, dtype=np.int64), dims="pt")
        pts = da_var.isel(latitude=row, longitude=col).transpose("time", "pt")
        sub_data_sel = pts.compute().to_numpy()  # keeps Dask lazy until compute
        sub_data_sel = np.transpose(sub_data_sel, (1, 0))  # grids, time
        # print(f"{var_name} load, time: ", time.time() - start)

        # start = time.time()
        data_sel_mean = groupby_np(sub_data_sel, interval=index_list_num, mean=True)  # basins, time
        # print(f"{var_name} groupby, time: ", time.time() - start)

        times = pd.date_range(start=f"{num_year}-01-01 00:00:00", end=f"{num_year}-12-31 23:00:00", freq="h")
        divide_id = station_ids[idx].values
        output_path = Path(output_dir) / f"{var_name}_{basin_start}.zarr"
        ds_out = xr.Dataset(
            {
                var_name: (("divide_id", "time"), data_sel_mean),
            },
            coords={
                "divide_id": divide_id,
                "time": times,
            },
        )
        ds_out = ds_out.chunk({"divide_id": save_chunk, "time": len(times)})
        ds_out.to_zarr(output_path, mode="w")

        del pts, sub_data_sel, data_sel_mean, ds_out
        gc.collect()

        peak_bytes = get_peak_rss_bytes()
        print(f"[MEM] Peak RSS: {peak_bytes / 1e9:.2f} GB")

    ds.close()
    del da_var, ds
    gc.collect()

    print("total time: ", time.time() - start_time)

'''
python distributedDS/adapter/aorc_data.py --num_year 1990 &
python distributedDS/adapter/aorc_data.py --num_year 1991 &
python distributedDS/adapter/aorc_data.py --num_year 1992 &
python distributedDS/adapter/aorc_data.py --num_year 1993 &
python distributedDS/adapter/aorc_data.py --num_year 1994 &
wait

profiling:
2000 basins, 96340 grids, 100s, 24.2G mem
chunk=2000 & discard >100000 basins, max grids=174434
chunk=2000, max grids=523212

'''

