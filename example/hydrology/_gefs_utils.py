"""Utilities for GEFS-based hydrology forecasting example.
> See './example/hydrology/example_dhbv_1_1p_gefs.ipynb'.

NOTE: may be formally adopted into dmg at a later time.

@ZhennanShi1
"""

import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.dmg.core.utils import Dates


def print_dataset_info(dataset):
    """Print dataset tensor shapes and descriptions."""
    print("\n\033[1mDataset Inputs\033[0m")
    # Header row
    print(f"{'Dataset key':<15}{'Shape':<20}{'Description'}")
    print("-" * 60)
    # Rows
    print(
        f"{'x_phy':<15}{str(tuple(dataset['x_phy'].size())):<20} # [time, basin, forcing_features]",
    )
    print(
        f"{'c_phy':<15}{str(tuple(dataset['c_phy'].size())):<20} # [basin, attr_features] (no physical attributes here)",
    )
    print(
        f"{'x_nn':<15}{str(tuple(dataset['x_nn'].size())):<20} # [time, basin, nn_forcing_features]",
    )
    print(
        f"{'c_nn':<15}{str(tuple(dataset['c_nn'].size())):<20} # [basin, nn_attr_features]",
    )
    print(
        f"{'xc_nn_norm':<15}{str(tuple(dataset['xc_nn_norm'].size())):<20} # [time, basin, combined_features]",
    )
    print(
        f"{'target':<15}{str(tuple(dataset['target'].size())):<20} # [time, basin, 1] observed streamflow",
    )
    print("\nSample of target tensor (first 5 timesteps, first basin):")
    print(f"{dataset['target'][:5, 0, 0]}")


def get_parameters_from_model(dpl_model, data, n_par, mu, device="cpu"):
    """Obtain parameters from a dPL model's neural network."""
    nn_model = dpl_model.nn_model
    print("nn model is", dpl_model.nn_model)
    print("xc_nn_norm shape before NN:", data["xc_nn_norm"].shape)
    xc_nn = data["xc_nn_norm"].to(device)  # make sure it is on GPU

    with torch.no_grad():
        out = nn_model(xc_nn)  # only time-series input
        if isinstance(out, (tuple, list)):
            out = out[0]

        T, B, F = (
            out.shape
        )  # zhennan comments: T: time dimension, B: # of basins, F: number of output features
        print(f"Output shape from NN: T={T}, B={B}, F={F}")  # 715, 1, 226

        if F == n_par * mu + 2:  # F = 14*16 + 2 -> 226
            flat = out[:, :, : n_par * mu]
            pars = flat.view(T, B, n_par, mu).to(device)
            rts = out[-1, :, n_par * mu : n_par * mu + 2].to(device)
        else:
            raise RuntimeError(
                f"Unexpected feature size {F}, expected {n_par}, {n_par * mu}, or {n_par * mu + 2}",
            )

        # sanity check
        print(f"pars shape: {pars.shape}")  # [715, 1, 14, 16]
        print(f"rts shape: {rts.shape}")  # [1, 2]

    return pars, rts  # pars: [T, B, n_par, mu], and rts: [B, 2]


def obtain_gage_name(GAGE_NAME_PATH, gage_id):
    """Obtain gage name from gage ID using the provided CSV file."""
    gage_names = pd.read_csv(GAGE_NAME_PATH, sep=";", dtype={"gauge_id": str})
    gage_names['gauge_id'] = gage_names['gauge_id'].str.lstrip('0')
    match = gage_names.loc[gage_names['gauge_id'] == str(gage_id), 'gauge_name']
    if match.empty:
        raise ValueError(f"GAGE_ID {gage_id} not found in {GAGE_NAME_PATH}")
    else:
        gage_name = match.values[0]
    return gage_name


def plot_ensemble_hydrograph(
    gage_path,
    gage_id,
    start_date,
    obs,
    sim,
    ens_preds,
    history_len,
    save_path=None,
):
    """Plot ensemble hydrograph with observations and simulation.

    Plots 2-month history + 15-day forecast:
      - Observed (black);
      - Simulation (red) for history;
      - 5 ensembles in distinct colors for forecast.
    """
    print("the length is here", len(sim), len(obs))
    ## Get the gage name first
    gage_name = obtain_gage_name(gage_path, gage_id)

    HORIZON = ens_preds.shape[1]
    start_date = pd.to_datetime(start_date)
    dates_obs = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + HORIZON,
    )
    dates_sim = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + 1,
    )

    # Pad ensembles for plotting (only forecast period valid)
    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates_obs, obs, "k-", lw=2, marker="*", label="Observed")
    plt.plot(dates_sim, sim, "r-", lw=1.5, marker="o", label="Simulation")

    # Ensembles in distinct colors
    colors = cm.tab10.colors  # 10 distinct colors
    for i in range(ens_preds.shape[0]):
        plt.plot(
            dates_obs,
            padded_ensembles[i],
            lw=1.5,
            color=colors[i % len(colors)],
            label=f"Ensemble {i + 1}",
        )

    # Shading for history vs forecast
    plt.axvspan(
        dates_obs[0],
        dates_obs[history_len],
        color="gray",
        alpha=0.1,
        label="Pre-GEFS Simulation",
    )
    plt.axvspan(
        dates_obs[history_len],
        dates_obs[-1],
        color="orange",
        alpha=0.1,
        label="GEFS Forecast",
    )

    plt.title(f"GEFS Forecast — Gage {gage_id} ({gage_name})")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (mm/day)")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=int(np.ceil(len(labels) / 4)))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def to_time_first(x_torch, device):
    """Convert tensor from [B,T,F] -> [T,B,F] and move to device."""
    return x_torch.permute(1, 0, 2).float().to(device)  # [T,B,F]


def safe_minmax(tensor):
    """Return min and max ignoring NaNs."""
    if tensor.numel() == 0:
        return np.nan, np.nan
    safe_min = torch.where(
        torch.isnan(tensor),
        torch.tensor(float("inf"), device=tensor.device),
        tensor,
    )
    min_val = float(torch.min(safe_min))
    safe_max = torch.where(
        torch.isnan(tensor),
        torch.tensor(float("-inf"), device=tensor.device),
        tensor,
    )
    max_val = float(torch.max(safe_max))
    return min_val, max_val


def checknans(warm_states):
    """Check NaNs in warm states and print min/max."""
    for name, st in zip(["sp", "mw", "sm", "suz", "slz"], warm_states):
        smin, smax = safe_minmax(st)
        print(name, "nan#", torch.isnan(st).sum().item(), "min", smin, "max", smax)


def GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date, showblock=False):
    """Basic error check for extracted GEFS forecast block."""
    if not idx_list:
        raise ValueError(f"Starting GEFS date {start_date.date()} not exist!")
    if len(fc_block) < horizon:
        raise ValueError("Extracted GEFS forcing not enough!")
    if showblock:
        print("\n========= GEFS Forecast Block (NO ERROR) =========")
        print(fc_block.to_string(index=False))


def selectbasins(rand, seed, basin_pool, n_basins, basin: int = 2046000):
    """Select basins for testing: either random selection or fixed basin."""
    if rand:
        random.seed(seed)
        selected_basins = random.sample(basin_pool, n_basins)
        print("Randomly selected basins:", selected_basins)
    else:
        print(f"Selecting a fixed basin {basin} for testing...")
        selected_basins = [basin]  #  [1547700]
        print("Fixed selected a basin:", selected_basins)
    return selected_basins


def startid_endid(start_date, forecast, config):
    """Get start and end indices for simulation based on start_date."""
    timesteps = Dates(
        config["simulation"],
        config["delta_model"]["rho"],
    ).batch_daily_time_range  # 730
    sidx = np.where(timesteps == start_date)[0][0]  # 715
    eidx = sidx + forecast  # 730
    return sidx, eidx, timesteps
