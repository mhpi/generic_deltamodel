"""Mirror of distributedDS/tests/test_model.py to verify forward capability in dMG.

@Wencong Yang, Leo Lonzarich
"""

import os
import sys
from tqdm import tqdm
import math
import torch
import yaml
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import json
from dmg.models import MtsModelHandler as ModelHandler
from conf.utils import convert_nested


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

model_version = 3
epoch = 30

forward_save_path = '/projects/mhpi/leoglonz/ciroh-ua/dmg/hf_outputs/'
forward_save_path = forward_save_path + f'h-dhbv2_{model_version}_Qprimeprime_fixed/'
os.makedirs(forward_save_path, exist_ok=True)

with open(
    Path(
        f'/projects/mhpi/yxs275/hourly_model/DShourly/trainedModel/h-dhbv2_{model_version}/'
    )
    / 'config.json'
) as f:
    config = json.load(f)
config = convert_nested(config)
obs_cfg_path = os.path.join(
    PROJECT_ROOT,
    "dmg",
    "conf",
    "observations_camels.yaml",
)

obs_config = yaml.safe_load(open(Path(obs_cfg_path)))
config['observations'] = obs_config
config['save_path'] = (
    f'/projects/mhpi/yxs275/hourly_model/DShourly/trainedModel/h-dhbv2_{model_version}/'
)
config['model_path'] = (
    f'/projects/mhpi/yxs275/hourly_model/DShourly/trainedModel/h-dhbv2_{model_version}/model/'
)
config['train']['target'] = ['Qs']


with open(
    Path('/projects/mhpi/yxs275/hourly_model/DShourly/trainedModel')
    / 'preprocessor_quantile0_1990_2003.json'
) as f:
    config_stats = json.load(f)

eps = 1e-6
mean_dyn_hourly = np.asarray(
    config_stats['mean']['dyn_input'], dtype=np.float32
)  ## daily and hour same?
std_dyn_hourly = np.asarray(config_stats['std']['dyn_input'], dtype=np.float32)

mean_dyn_daily = np.asarray(
    config_stats['mean']['dyn_input_daily'], dtype=np.float32
)  ## daily and hour same?
std_dyn_daily = np.asarray(config_stats['std']['dyn_input_daily'], dtype=np.float32)

mean_attr = np.asarray(config_stats['mean']['static_input'], dtype=np.float32)
std_attr = np.asarray(config_stats['std']['static_input'], dtype=np.float32)

mean_attr_rout = np.asarray(config_stats['mean']['rout_static_input'], dtype=np.float32)
std_attr_rout = np.asarray(config_stats['std']['rout_static_input'], dtype=np.float32)

while mean_dyn_hourly.ndim < 3:
    mean_dyn_hourly = mean_dyn_hourly[np.newaxis, ...]
    std_dyn_hourly = std_dyn_hourly[np.newaxis, ...]

while mean_dyn_daily.ndim < 3:
    mean_dyn_daily = mean_dyn_daily[np.newaxis, ...]
    std_dyn_daily = std_dyn_daily[np.newaxis, ...]

while mean_attr.ndim < 2:
    mean_attr = mean_attr[np.newaxis, ...]
    std_attr = std_attr[np.newaxis, ...]

while mean_attr_rout.ndim < 2:
    mean_attr_rout = mean_attr_rout[np.newaxis, ...]
    std_attr_rout = std_attr_rout[np.newaxis, ...]

zTest_hour_time = pd.date_range('2009-01-01', '2018-10-01', freq='1h')[:-1]
zTest_time = pd.date_range('2004-10-01', '2018-09-30')
zTest_full_time = pd.date_range('2004-10-01 00:00:00', '2018-10-01 00:00:00', freq='h')[
    :-1
]

### manual data
var_x_list = config['delta_model']['nn_model']['high_freq_model']['forcings']
var_c_list = config['delta_model']['nn_model']['high_freq_model']['attributes']
var_c_list2 = config['delta_model']['nn_model']['high_freq_model']['attributes2']
attrs_ds_all = xr.open_dataset(
    '/projects/mhpi/yxs275/hourly_model/mtsHBV/data/CAMELS_HFs_attr_new.nc'
)
gauge_chunk_size = 500
n_gauges = attrs_ds_all.dims["gauge"]
attrs_ds_all.close()

n_chunks = math.ceil(n_gauges / gauge_chunk_size)
print(f"Total gauges: {n_gauges}, chunks of {gauge_chunk_size}: {n_chunks}")

n_gages = gauge_chunk_size
n_units = gauge_chunk_size
dt = 1.0 / 24


def hourly_forward(input):
    """Hourly forward function for a chunk of gauges on a specific device."""
    ichunk, device = input

    model = ModelHandler(config=config, device=device, verbose=True)
    model.load_model(epoch=epoch)
    # new
    model.dpl_model.phy_model.high_freq_model.use_distr_routing = False

    g_start = ichunk * gauge_chunk_size
    g_end = min((ichunk + 1) * gauge_chunk_size, n_gauges)
    print("Working on chunk ", ichunk, " ", g_start, " ", g_end, " on GPU ", device)

    hourly_x_path = f'/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_hourly_{g_start:05d}_{g_end - 1:05d}.nc'
    daily_x_path = f'/gpfs/yxs275/data/hourly/CAMELS_HF/forcing/forcing_1990_2018_gauges_daily_{g_start:05d}_{g_end - 1:05d}.nc'
    print("Reading ", hourly_x_path)
    print("Reading ", daily_x_path)
    hourly_x = xr.open_dataset(hourly_x_path).sel(
        time=zTest_full_time
    )  ## pet from hargreaves, runoff from camels
    # hourly_x['P'] = hourly_x['P']
    # hourly_x['PET'] = hourly_x['PET']

    STAIDs = hourly_x['gauge'].values
    daily_x = (
        xr.open_dataset(daily_x_path)
        .sel(gauge=STAIDs, time=zTest_time)
        .transpose("gauge", "time")
    )
    attrs_ds = xr.open_dataset(
        '/projects/mhpi/yxs275/hourly_model/mtsHBV/data/CAMELS_HFs_attr_new.nc'
    ).sel(gauge=STAIDs)
    hourly_x = hourly_x.rename({"T": "Temp"})
    daily_x = daily_x.rename({"T": "Temp"})

    hourly_forcing = [np.expand_dims(hourly_x[x].values, axis=-1) for x in var_x_list]
    hourly_forcing = np.concatenate(hourly_forcing, axis=-1)
    # hourly_x.close
    daily_forcing = [np.expand_dims(daily_x[x].values, axis=-1) for x in var_x_list]
    daily_forcing = np.concatenate(daily_forcing, axis=-1)
    # daily_x.close

    attr = [np.expand_dims(attrs_ds[x].values, axis=-1) for x in var_c_list]
    attr = np.concatenate(attr, axis=-1)

    attr_rout = [np.expand_dims(attrs_ds[x].values, axis=-1) for x in var_c_list2]
    attr_rout = np.concatenate(attr_rout, axis=-1)

    # attrs_ds.close

    daily_forcing_norm = (daily_forcing - mean_dyn_daily) / (std_dyn_daily + eps)
    hourly_forcing_norm = (hourly_forcing - mean_dyn_hourly) / (std_dyn_hourly + eps)

    # da_hourly = xr.DataArray(
    #     hourly_forcing_norm,
    #     coords={
    #         "gage": STAIDs,
    #         "time": zTest_full_time,
    #         "variable": var_x_list,
    #     },
    #     dims=("gage", "time","variable"),
    # )

    # # daily sum (change to .mean() if you want daily mean)
    # da_daily = da_hourly.resample(time="1D").mean()

    # daily_forcing_norm = da_daily.values          # shape: (n_gage, n_day)

    attr_norm = (attr - mean_attr) / (std_attr + eps)

    attr_norm_rout = (attr_rout - mean_attr_rout) / (std_attr_rout + eps)

    window_size_hour = 7 * 24
    lookback_days = 365
    window_size_day = 7

    start_index = zTest_full_time.get_loc(zTest_hour_time[0])
    n_hour = zTest_full_time.shape[0]

    # n_pairs = 10000
    # rs = len(config['delta_model']['nn_model']['high_freq_model']['attributes2'])

    hourly_predict = []
    for i in tqdm(range(start_index, n_hour, window_size_hour)):
        end_i = min(i + window_size_hour, n_hour)
        current_window_size = end_i - i
        if current_window_size < 1:
            continue

        start_hour = zTest_full_time[i]

        start_daily = start_hour - pd.Timedelta(days=(lookback_days - window_size_day))
        # start_daily = start_hour - pd.Timedelta(days = (lookback_days)-2*window_size_day)
        if start_daily < zTest_time[0]:
            start_index = i
            continue

        start_daily_index = zTest_time.get_loc(start_daily)
        x_phy_high_freq = torch.from_numpy(
            hourly_forcing[:, i - current_window_size : i + current_window_size, :]
        ).permute([1, 0, 2])  ## 7 daily for warmup?
        # x_phy_high_freq =  torch.from_numpy(hourly_forcing[:,i:i+2*current_window_size,:]).permute([1,0,2])
        # x_phy_low_freq  = torch.from_numpy(daily_forcing[:,start_daily_index:start_daily_index + lookback_days - 2*window_size_day,:]).permute([1,0,2])
        x_phy_low_freq = torch.from_numpy(
            daily_forcing[
                :,
                start_daily_index : start_daily_index
                + lookback_days
                - 2 * window_size_day,
                :,
            ]
        ).permute([1, 0, 2])

        xc_nn_norm_high_freq = torch.from_numpy(
            hourly_forcing_norm[:, i - current_window_size : i + current_window_size, :]
        ).permute([1, 0, 2])
        # xc_nn_norm_high_freq = torch.from_numpy(hourly_forcing_norm[:,i:i+2*current_window_size,:]).permute([1,0,2])
        xc_nn_norm_low_freq = torch.from_numpy(
            daily_forcing_norm[
                :,
                start_daily_index : start_daily_index
                + lookback_days
                - 2 * window_size_day,
                :,
            ]
        ).permute([1, 0, 2])
        # xc_nn_norm_low_freq = torch.from_numpy(daily_forcing_norm[:,start_daily_index:start_daily_index + lookback_days,:] ).permute([1,0,2])

        c_nn_norm = torch.from_numpy(attr_norm)
        xc_nn_norm_high_freq = torch.cat(
            (
                xc_nn_norm_high_freq,
                c_nn_norm.unsqueeze(0).repeat(xc_nn_norm_high_freq.shape[0], 1, 1),
            ),
            dim=-1,
        )
        xc_nn_norm_low_freq = torch.cat(
            (
                xc_nn_norm_low_freq,
                c_nn_norm.unsqueeze(0).repeat(xc_nn_norm_low_freq.shape[0], 1, 1),
            ),
            dim=-1,
        )
        rc_nn_norm = torch.from_numpy(attr_norm_rout)

        elev_all = torch.from_numpy(attrs_ds['meanelevation'].values)
        ac_all = torch.from_numpy(attrs_ds['uparea'].values)
        areas = torch.from_numpy(attrs_ds['catchsize'].values)

        outlet_topo = torch.eye(n_units)
        # rc_nn_norm = torch.randn(int(outlet_topo.sum()), rs)
        data_dict = {
            'xc_nn_norm_low_freq': xc_nn_norm_low_freq,
            'xc_nn_norm_high_freq': xc_nn_norm_high_freq,
            'c_nn_norm': c_nn_norm,
            'rc_nn_norm': rc_nn_norm,
            'x_phy_low_freq': x_phy_low_freq,
            'x_phy_high_freq': x_phy_high_freq,
            'ac_all': ac_all,
            'elev_all': elev_all,
            'areas': areas,
            'outlet_topo': outlet_topo,
        }
        for key in data_dict:
            data_dict[key] = data_dict[key].to(device=device, dtype=torch.float32)

        output = model.dpl_model(data_dict)

        hourly_predict.append(
            output['Qs'][-current_window_size:, :, 0].detach().cpu().numpy()
        )

    hourly_predict = np.swapaxes(np.concatenate(hourly_predict, axis=0), 0, 1)
    predict_time = zTest_full_time[start_index:end_i]

    gauge_ids = STAIDs
    time_index = predict_time

    da = xr.DataArray(
        hourly_predict,
        dims=("gauge", "time"),
        coords={"gauge": gauge_ids, "time": time_index},
        name="Runoff",  # name of the variable
    )

    # Wrap in a Dataset (NetCDF likes Datasets)
    ds = xr.Dataset({"Simulation": da})

    # Save to NetCDF
    ds.to_netcdf(
        forward_save_path
        + f"hourly_simulation_{ichunk}_{g_start:05d}_{g_end - 1:05d}.nc",
        mode="w",
        format="NETCDF4",
    )


if __name__ == "__main__":
    import time
    import multiprocessing

    startTime = time.time()

    # GPU CONFIGURATION
    GPU_ids = [0, 1, 2, 3]
    num_gpus = len(GPU_ids)

    # DATA CHUNKING
    items = list(range(n_chunks))
    GPU_ids_list = [GPU_ids[x % len(GPU_ids)] for x in items]

    # BATCHING PROCESSES
    # Note: Creating a new pool inside a loop is inefficient,
    # but I will keep your logic structure to minimize changes.
    processeornumber = num_gpus * 8
    iS = np.arange(0, len(items), processeornumber)
    iE = np.append(iS[1:], len(items))

    for i in range(len(iS)):
        subGPU_ids_list = GPU_ids_list[iS[i] : iE[i]]
        subitem = items[iS[i] : iE[i]]

        print(f"Starting Batch {i}")
        print("zone idx ", subitem)
        print("GPUs ", subGPU_ids_list)

        # Create the pool
        pool = multiprocessing.Pool(processes=num_gpus * 8)

        # Prepare arguments generator
        args = ((subitem[idx], gpuid) for idx, gpuid in enumerate(subGPU_ids_list))

        # --- FIX IS HERE ---
        # We wrap the imap call in list() to force execution
        results = list(pool.imap(hourly_forward, args))

        pool.close()
        pool.join()

    print("Cost time: ", time.time() - startTime)
