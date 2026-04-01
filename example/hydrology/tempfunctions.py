import numpy as np
import pandas as pd
import json
import torch 
import os 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.dmg.core.utils import Dates 
 
def plot_forecast_stacked(
    GAGE_NAME_PATH,
    gage_id,
    start_date,
    obs,                 # length = history_len + HORIZON
    sim,                 # length = history_len + 1
    ens_preds,           # shape = [n_ens, HORIZON]
    history_len,
    Q_all_np,             
    sidx,                # input index of forecast start in the dataset timeline
    FORECAST,
    WARMUPTIME,
    det_pred=None,       # length = HORIZON
    save_path=None,
):
    """
    Two stacked plots:
      TOP: Observed + history sim + ensembles + envelope + optional restart deterministic forecast line
      BOTTOM: Continuous single-run output curve + NSE/KGE on overlap with obs
    """
    gage_name = obtain_gage_name(GAGE_NAME_PATH, gage_id)

    ens_preds = np.asarray(ens_preds)
    if ens_preds.ndim != 2:
        raise ValueError(f"ens_preds must be [n_ens, HORIZON], got {ens_preds.shape}")

    HORIZON = ens_preds.shape[1]
    if HORIZON != FORECAST:
        print(f"WARNING: HORIZON={HORIZON} != FORECAST={FORECAST}")

    start_date = pd.to_datetime(start_date)

    # ---------- Top plot dates ----------
    dates_obs_top = pd.date_range(
    start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + HORIZON,
    )
    dates_sim_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len,
    )

    # Pad ensembles to full timeline
    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    ens_min = np.full(history_len + HORIZON, np.nan)
    ens_max = np.full(history_len + HORIZON, np.nan)
    if not np.all(np.isnan(ens_preds)):
        ens_min[history_len:] = np.nanmin(ens_preds, axis=0)
        ens_max[history_len:] = np.nanmax(ens_preds, axis=0)

    # ---------- Continuous run dates ----------
    dataset_day0 = start_date - pd.Timedelta(days=sidx)
    dates_cont = pd.date_range(
        start=dataset_day0 + pd.Timedelta(days=WARMUPTIME),
        periods=len(Q_all_np),
    )

    # ---------- Figure ----------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 9),
        sharex=True, sharey=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # ===================== TOP =====================
    ax1.plot(dates_obs_top, obs, "k-", lw=2, marker="*", label="Observed")
    ax1.plot(dates_sim_top, sim, "r-", lw=1.5, marker="o", label="Simulation (history)")

    ax1.fill_between(dates_obs_top, ens_min, ens_max, color="blue", alpha=0.1, label="Ensembles")
    for i in range(ens_preds.shape[0]):
        ax1.plot(dates_obs_top, padded_ensembles[i], lw=1.0, color="darkblue")

    if det_pred is not None:
        det_pred = np.asarray(det_pred).reshape(-1)
        if det_pred.shape[0] != HORIZON:
            raise ValueError(f"det_pred length {det_pred.shape[0]} != HORIZON {HORIZON}")
        padded_det = np.full(history_len + HORIZON, np.nan)
        padded_det[history_len:] = det_pred
        ax1.plot(
            dates_obs_top,
            padded_det,
            lw=2.5,
            color="red",
            label="HBV restart (deterministic)"
        )

    ax1.axvspan(dates_obs_top[0], dates_obs_top[history_len], color="gray", alpha=0.1, label="Pre-GEFS")
    ax1.axvspan(dates_obs_top[history_len], dates_obs_top[-1], color="orange", alpha=0.1, label="Forecast")

    ax1.set_title(f"Forecast — Gage {gage_id} ({gage_name})")
    ax1.set_ylabel("Streamflow (mm/day)")
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper center", ncol=max(1, int(np.ceil(len(l1) / 4))))

    # ---------- Same x-limits for both ----------
    pad_days = 3
    x0, x1 = dates_obs_top[0], dates_obs_top[-1]
    ax1.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))

    # ===================== BOTTOM =====================
    mask = (dates_cont >= x0) & (dates_cont <= x1)

    cont_dates_win = dates_cont[mask]
    cont_vals_win = np.asarray(Q_all_np)[mask]

    ax2.plot(
        cont_dates_win,
        cont_vals_win,
        "-o",
        color="green",
        lw=1.5,
        markersize=6,
        label="HBV continuous output"
    )

    # ---------- Align observed data with continuous run ----------
    obs_series = pd.Series(np.asarray(obs), index=dates_obs_top)
    cont_series = pd.Series(cont_vals_win, index=cont_dates_win)

    common_dates = obs_series.index.intersection(cont_series.index)
    obs_aligned = obs_series.loc[common_dates].values
    cont_aligned = cont_series.loc[common_dates].values

    # remove NaNs before metric calculation
    valid = np.isfinite(obs_aligned) & np.isfinite(cont_aligned)

    nse_val, kge_val = np.nan, np.nan
    if valid.sum() > 1:
        obs_valid = obs_aligned[valid]
        cont_valid = cont_aligned[valid]

        # optional: also plot observed values on bottom panel
        ax2.plot(
            common_dates[valid],
            obs_valid,
            "k-*",
            lw=1.5,
            markersize=6,
            label="Observed"
        )

        nse_val = nse(cont_valid, obs_valid)
        kge_val = kge(cont_valid, obs_valid)

        # add text box on plot
        ax2.text(
            0.02, 0.98,
            f"NSE = {nse_val:.3f}\nKGE = {kge_val:.3f}",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Streamflow (mm/day)")
    ax2.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
 
 
def nse(sim, obs):
    return 1 - np.sum((sim - obs)**2) / np.sum((obs - obs.mean())**2)

def kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta  = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def compute_bias_correction_from_dataset(
    daymet_tensor, gefs_df, timesteps, basin_idx, window=15, method="scalar"
):
    """
    Compute bias correction factors (scalar or CDF-based) for GEFS forcings
    using Daymet climatology in a moving DOY window.

    Parameters
    ----------
    daymet_tensor : torch.Tensor [time, basin, 3]
        Historical Daymet forcings from the dataset.
    gefs_df : pd.DataFrame
        GEFS historical forcings with columns: date, prcp, tmean, pet.
    timesteps : np.ndarray
        Daily time axis corresponding to Daymet.
    basin_idx : int
        Basin index to select from Daymet tensor.
    window : int
        ±days around each DOY to compute moving-window stats.
    method : str, {"scalar", "cdf"}
        Bias correction method:
        - "scalar": mean ratio (current method)
        - "cdf": quantile mapping / CDF matching

    Returns
    -------
    dict
        A dictionary of corrections per variable.
        For "scalar" → corrections[var][doy] = scalar value
        For "cdf" → corrections[var][doy] = (p_grid, mapped_values)
    """
    # Convert Daymet tensor → DataFrame
    df_daymet = pd.DataFrame({
        "date": pd.to_datetime(timesteps),
        "prcp": daymet_tensor[:, basin_idx, 0].detach().cpu().numpy(),
        "tmean": daymet_tensor[:, basin_idx, 1].detach().cpu().numpy(),
        "pet": daymet_tensor[:, basin_idx, 2].detach().cpu().numpy(),
    })
    df_daymet["doy"] = df_daymet["date"].dt.dayofyear

    gefs_df = gefs_df.copy()
    gefs_df["doy"] = gefs_df["date"].dt.dayofyear

    corrections = {v: {} for v in ["prcp","tmean","pet"]}

    for var in ["prcp","tmean","pet"]:
        for doy in range(1, 366):
            mask_d = (df_daymet["doy"] - doy).abs() <= window
            mask_g = (gefs_df["doy"] - doy).abs() <= window

            vals_d = df_daymet.loc[mask_d, var].dropna()
            vals_g = gefs_df.loc[mask_g, var].dropna()

            if len(vals_d) < 10 or len(vals_g) < 10:
                continue

            if method == "scalar":
                mean_d = vals_d.mean()
                mean_g = vals_g.mean()
                corrections[var][doy] = mean_d / mean_g if mean_g > 1e-6 else 1.0

            elif method == "cdf":
                # Create percentile grid (0–100)
                p_grid = np.linspace(0, 100, 101)
                q_daymet = np.percentile(vals_d, p_grid)
                q_gefs = np.percentile(vals_g, p_grid)
                corrections[var][doy] = (q_gefs, q_daymet)  # (x_in, x_out)
            else:
                raise ValueError(f"Unknown correction method '{method}'")

    return corrections


def pre_processing(df, corrections, method="scalar"):
    """
    Apply bias correction (scalar or CDF) to GEFS forecast DataFrame.
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear

    for var in ["prcp", "tmean", "pet"]:
        if method == "scalar":
            df[var] *= df["doy"].map(lambda d: corrections[var].get(d, 1.0))

        elif method == "cdf":
            corrected_vals = []
            for i, row in df.iterrows():
                doy = row["doy"]
                val = row[var]
                if doy not in corrections[var]:
                    corrected_vals.append(val)
                    continue
                q_in, q_out = corrections[var][doy]
                # interpolate within empirical CDFs
                corrected_vals.append(np.interp(val, q_in, q_out))
            df[var] = corrected_vals

        else:
            raise ValueError(f"Unknown correction method '{method}'")

    return df


def post_processing(sim_pre_GEFS, ens_preds): # post processing
    """
    Apply a constant offset correction so that the first forecasted streamflow
    connects smoothly with the last simulated (Daymet) streamflow.

    Parameters
    ----------
    sim_pre_GEFS : np.ndarray
        1D array of simulated streamflow before the forecast period (from HBV using Daymet).
    ens_preds : np.ndarray
        2D array of ensemble forecasts with shape (N_ENSEMBLES, FORECAST_LENGTH).

    Returns
    -------
    np.ndarray
        Offset-corrected ensemble forecasts of the same shape.
    """
    Q_sim_end = sim_pre_GEFS[-1]  # last simulated streamflow value
    ens_preds_corrected = []

    for ens in ens_preds:
        offset = Q_sim_end - ens[0]          # compute offset
        ens_corrected = np.maximum(ens + offset, 0.0)  # apply and ensure non-negative
        ens_preds_corrected.append(ens_corrected)

    return np.array(ens_preds_corrected)


def obtain_gage_name(GAGE_NAME_PATH, gage_id): 
    gage_names = pd.read_csv(
    GAGE_NAME_PATH,
    sep=";",
    dtype={"gauge_id": str}
    ) 
    gage_names['gauge_id'] = gage_names['gauge_id'].str.lstrip('0') 
    match = gage_names.loc[gage_names['gauge_id'] == str(gage_id), 'gauge_name']
    if match.empty:
        raise ValueError(f"GAGE_ID {gage_id} not found in {GAGE_NAME_PATH}")
    else:
        gage_name = match.values[0]
    return gage_name

  
def GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date, showblock=False):
    if not idx_list: raise ValueError(f"Starting GEFS date {start_date.date()} not exist!")
    if len(fc_block) < horizon: raise ValueError("Extracted GEFS forcing not enough!")
    if showblock:
        print("\n========= GEFS Forecast Block (NO ERROR) =========")
        print(fc_block.to_string(index=False))
         
def run_warm_forecasts_restart(
    model,
    state_path,
    gage_id,
    basin_idx,
    start_date,
    horizon,
    N_ENSEMBLES,
    GEFS_DIR,
    data_loader,
    timesteps,
    WINDOW,
    CORRECTION,
    device,
    verbose,
):
    ens_preds = []

    name = list(model.model_dict.keys())[0]
    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    # restart behavior
    phy_model.cache_states = True
    phy_model.warm_up = 0
    phy_model.warm_up_states = True

    for ens_id in range(N_ENSEMBLES): 
        # 1. Reset to saved stop-point states 
        model.load_states(path=state_path)
 
        # 2. Read GEFS forcings 
        f_path = os.path.join(GEFS_DIR, f"ens0{ens_id+1}", f"{gage_id:08d}.txt")
        if not os.path.exists(f_path):
            raise ValueError(f"Missing GEFS file: {f_path}")

        df = pd.read_csv(f_path, sep=r"\s+", header=0)
        df = df.rename(columns={
            "Year": "year", "Mnth": "month", "Day": "day",
            "prcp(mm/day)": "prcp", "tmean(C)": "tmean", "pet(mm/day)": "pet"
        })
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

        idx_list = df.index[df["date"] == start_date].to_list()
        if not idx_list:
            raise ValueError(f"Start date {start_date} not found in {f_path}")

        fc_block = df.iloc[idx_list[0]: idx_list[0] + horizon].copy()
        GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date)
 
        # 3. Bias correction 
        if CORRECTION:
            bias_corrections = compute_bias_correction_from_dataset(
                data_loader.dataset["x_phy"],
                df,
                timesteps,
                basin_idx,
                window=WINDOW,
                method=CORRECTION,
            )
            fc_block = pre_processing(fc_block, bias_corrections, method=CORRECTION)
 
        # 4. Build HBV forcing tensor 
        raw_np = fc_block[["prcp", "tmean", "pet"]].to_numpy().astype(np.float32)
        x_phy_fc = torch.tensor(raw_np[:, np.newaxis, :], dtype=torch.float32, device=device)
        x_phy_fc[torch.isnan(x_phy_fc)] = 0.0

        # ----------------------------------------
        # 5. Build NN input tensor
        # ----------------------------------------
        xc_nn_norm_fc = build_xc_nn_norm_forecast(
            data_loader=data_loader,
            basin_idx=basin_idx,
            fc_block=fc_block,
            device=device,
            verbose = verbose,
        )

        # ----------------------------------------
        # 6. Run LSTM -> HBV
        # ----------------------------------------
        with torch.no_grad():
            raw_nn_out = nn_model(xc_nn_norm_fc)
            fluxes = phy_model(
                x_dict={"x_phy": x_phy_fc},
                parameters=raw_nn_out
            )

        q_fc = fluxes["streamflow"][:, 0, 0].detach().cpu().numpy()
        ens_preds.append(q_fc)

    return np.array(ens_preds)

def build_xc_nn_norm_forecast(data_loader, basin_idx, fc_block, device, verbose):
    import numpy as np
    import torch

    if verbose:
        print("===== BUILD XC_NN_NORM FORECAST =====")

    # 1. Raw forecast forcings
    x_nn_fc = fc_block[data_loader.nn_forcings].to_numpy().astype(np.float32)   # [T, 3]
    if verbose:
        print("raw GEFS forcings shape:", x_nn_fc.shape)

    x_nn_fc = x_nn_fc[:, np.newaxis, :]                                          # [T, 1, 3]
    if verbose:
        print("after add basin dim:", x_nn_fc.shape)

    # 2. Static attributes
    c_nn_fc = data_loader.dataset["c_nn"][basin_idx:basin_idx+1].detach().cpu().numpy().astype(np.float32)  # [1, 35]
    if verbose:
        print("static attributes shape:", c_nn_fc.shape)

    # 3. Normalize forcings
    x_nn_norm = data_loader.to_norm(x_nn_fc, data_loader.nn_forcings)
    if verbose:
        print("normalized forcings shape:", x_nn_norm.shape)

    # 4. Normalize static attrs
    c_nn_norm = data_loader.to_norm(c_nn_fc, data_loader.nn_attributes)
    if verbose:
        print("normalized static attrs shape:", c_nn_norm.shape)

    # 5. Repeat static attrs over time
    c_nn_norm = np.repeat(
        np.expand_dims(c_nn_norm, 0),
        x_nn_norm.shape[0],
        axis=0
    )  # [T, 1, 35]
    if verbose:
        print("repeated static attrs shape:", c_nn_norm.shape)

    # 6. Concatenate
    xc_nn_norm_fc = np.concatenate((x_nn_norm, c_nn_norm), axis=2)  # [T, 1, 38]
    if verbose:
        print("final xc_nn_norm_fc shape:", xc_nn_norm_fc.shape)

    # 7. Convert to tensor
    xc_nn_norm_fc = torch.tensor(xc_nn_norm_fc, dtype=torch.float32, device=device)
    if verbose:
        print("final tensor shape:", xc_nn_norm_fc.shape)

    if verbose:
        print("=====================================")

    return xc_nn_norm_fc

 

def run_segment(
    model,
    dataset,
    basin_idx,
    device,
    name=None,
    start_idx=None,
    end_idx=None,
    warm_up=0,
    cache_states=False,
    warm_up_states=True,
    state_path=None,
):
    """
    Slice one basin/time segment from dataset and run NN + physical model.

    Parameters
    ----------
    model : ModelHandler
    dataset : dict
        Must contain 'xc_nn_norm' and 'x_phy'
    basin_idx : int
    device : torch.device
    name : str or None
        Model name. If None, uses the first key in model.model_dict.
    start_idx, end_idx : int or None
        Time slice for the segment.
    warm_up : int
    cache_states : bool
    warm_up_states : bool
    state_path : str or None
        If provided, load saved states before forward.

    Returns
    -------
    nn_out : torch.Tensor
    fluxes : dict
    phy_model : object
    data_dict : dict
        The sliced inputs used for this segment.
    """
    if name is None:
        name = list(model.model_dict.keys())[0]

    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    data_dict = {
        "xc_nn_norm": dataset["xc_nn_norm"][start_idx:end_idx, basin_idx:basin_idx + 1, :].clone(),
        "x_phy": dataset["x_phy"][start_idx:end_idx, basin_idx:basin_idx + 1, :].clone(),
    }

    phy_model.warm_up = warm_up
    phy_model.cache_states = cache_states
    phy_model.warm_up_states = warm_up_states

    model.eval()

    if state_path is not None:
        model.load_states(path=state_path)

    with torch.no_grad():
        xc = data_dict["xc_nn_norm"].to(device)
        x_phy = data_dict["x_phy"].to(device)

        nn_out = nn_model(xc)
        fluxes = phy_model(
            x_dict={"x_phy": x_phy},
            parameters=nn_out,
        )
    # return nn_out, fluxes, phy_model, data_dict
    return fluxes['streamflow'][:, 0, 0].detach().cpu().numpy()


def selectbasins():
    with open("./predownloaded/Best_two_in_region.json", "r") as f:
        basin_groups = json.load(f)
    basin_pool = [b for group in basin_groups.values() for b in group] 
    return basin_pool


def startid_endid(config, warm, ENSEMBLE_START_DATE, FORECAST):
    timesteps = Dates(config['sim'], config['model']['rho']).batch_daily_time_range # 730
    # print("check on config", config["sim"])
    sidx = np.where(timesteps == ENSEMBLE_START_DATE)[0][0] # 715
    eidx = sidx + FORECAST # 730 
    history_len = len(timesteps) - warm - FORECAST
    # print("sidx, eidx, timesteps, history_len are", sidx, eidx, timesteps, history_len)
    return sidx, eidx, timesteps, history_len 

def plot_forecast_separate(
    GAGE_NAME_PATH,
    gage_id,
    start_date,
    obs,                 # length = history_len + HORIZON
    sim,                 # length = history_len
    ens_preds,           # shape = [n_ens, HORIZON]
    history_len,
    Q_all_np,            # continuous output from long run
    sidx,                # input index of forecast start in the dataset timeline
    FORECAST,
    WARMUPTIME,
    det_pred=None,       # length = HORIZON
    save_dir=None,
    file_prefix=None,
):
    """
    Save two separate figures:
      1) Top forecast figure
      2) Bottom continuous-run figure
    """
    gage_name = obtain_gage_name(GAGE_NAME_PATH, gage_id)

    ens_preds = np.asarray(ens_preds)
    if ens_preds.ndim != 2:
        raise ValueError(f"ens_preds must be [n_ens, HORIZON], got {ens_preds.shape}")

    HORIZON = ens_preds.shape[1]
    if HORIZON != FORECAST:
        print(f"WARNING: HORIZON={HORIZON} != FORECAST={FORECAST}")

    start_date = pd.to_datetime(start_date)

    # ---------- Top plot dates ----------
    dates_obs_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + HORIZON,
    )
    dates_sim_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len,
    )

    # ---------- Pad ensembles ----------
    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    ens_min = np.full(history_len + HORIZON, np.nan)
    ens_max = np.full(history_len + HORIZON, np.nan)
    if not np.all(np.isnan(ens_preds)):
        ens_min[history_len:] = np.nanmin(ens_preds, axis=0)
        ens_max[history_len:] = np.nanmax(ens_preds, axis=0)

    # ---------- Continuous run dates ----------
    dataset_day0 = start_date - pd.Timedelta(days=sidx)
    dates_cont = pd.date_range(
        start=dataset_day0 + pd.Timedelta(days=WARMUPTIME),
        periods=len(Q_all_np),
    )

    # ---------- x-range shared by both ----------
    pad_days = 3
    x0, x1 = dates_obs_top[0], dates_obs_top[-1]

    # ==========================================================
    # FIGURE 1: TOP FORECAST
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))

    ax1.plot(dates_obs_top, obs, "k-", lw=2, label="Observed")
    ax1.plot(dates_sim_top, sim, "r-", lw=1.5, label="Simulation (history)")

    ax1.fill_between(dates_obs_top, ens_min, ens_max, color="green", alpha=0.1, label="Ensembles")
    for i in range(ens_preds.shape[0]):
        ax1.plot(dates_obs_top, padded_ensembles[i], lw=1.5, color="darkgreen")

    if det_pred is not None:
        det_pred = np.asarray(det_pred).reshape(-1)
        if det_pred.shape[0] != HORIZON:
            raise ValueError(f"det_pred length {det_pred.shape[0]} != HORIZON {HORIZON}")
        padded_det = np.full(history_len + HORIZON, np.nan)
        padded_det[history_len:] = det_pred
        ax1.plot(
            dates_obs_top,
            padded_det,
            lw=1.5,
            color="purple",
            label="HBV Restart"
        )
    # ---------- Metrics for top figure ----------
    obs = np.asarray(obs).reshape(-1)
    sim = np.asarray(sim).reshape(-1)
    ens_preds = np.asarray(ens_preds)

    # Split observed into history and forecast parts
    obs_hist = obs[:history_len]
    obs_fcst = obs[history_len:history_len + HORIZON]

    # --- Pre-forecast metrics: sim vs observed history ---
    pre_nse, pre_kge = np.nan, np.nan
    valid_pre = np.isfinite(sim) & np.isfinite(obs_hist)

    if valid_pre.sum() > 1:
        pre_nse = nse(sim[valid_pre], obs_hist[valid_pre])
        pre_kge = kge(sim[valid_pre], obs_hist[valid_pre])

    # --- Forecast ensemble metrics: each ensemble member vs observed forecast ---
    ens_nse_list = []
    ens_kge_list = []

    for i in range(ens_preds.shape[0]):
        pred_i = ens_preds[i]
        valid_i = np.isfinite(pred_i) & np.isfinite(obs_fcst)

        if valid_i.sum() > 1:
            ens_nse_i = nse(pred_i[valid_i], obs_fcst[valid_i])
            ens_kge_i = kge(pred_i[valid_i], obs_fcst[valid_i])
        else:
            ens_nse_i = np.nan
            ens_kge_i = np.nan

        ens_nse_list.append(ens_nse_i)
        ens_kge_list.append(ens_kge_i)

    ens_nse_arr = np.asarray(ens_nse_list)
    ens_kge_arr = np.asarray(ens_kge_list)
    metric_text = "GEFS Ensembles\n"
    for i in range(len(ens_nse_arr)):
        metric_text += f"E{i+1}: NSE={ens_nse_arr[i]:.3f}, KGE={ens_kge_arr[i]:.3f}\n"
            
    ax1.axvspan(dates_obs_top[0], dates_obs_top[history_len], color="gray", alpha=0.2, label="Pre-Forecast")
    ax1.axvspan(dates_obs_top[history_len], dates_obs_top[-1], color="orange", alpha=0.2, label="Forecast")
    
    metric_text = (
        f"Pre-Forecast\n"
        f"NSE={pre_nse:.3f}, KGE={pre_kge:.3f}\n\n"
        f"GEFS Ensembles\n"
    )
    for i in range(len(ens_nse_arr)):
        metric_text += f"E{i+1}: NSE={ens_nse_arr[i]:.3f}, KGE={ens_kge_arr[i]:.3f}\n"
        
    ax1.text(
        0.98, 0.98,
        metric_text,
        transform=ax1.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    
    ax1.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax1.set_title(f"GEFS Forecast — Gage {gage_id} ({gage_name})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Streamflow (mm/day)")
    ax1.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper center", ncol=max(1, int(np.ceil(len(l1) / 4))))
    fig1.tight_layout()

    # ==========================================================
    # FIGURE 2: BOTTOM CONTINUOUS RUN
    # ==========================================================
    fig2, ax2 = plt.subplots(figsize=(12, 4.8))

    mask = (dates_cont >= x0) & (dates_cont <= x1)
    cont_dates_win = dates_cont[mask]
    cont_vals_win = np.asarray(Q_all_np)[mask]
 
    # ---------- Align observed data with continuous run ----------
    obs_series = pd.Series(np.asarray(obs), index=dates_obs_top)
    cont_series = pd.Series(cont_vals_win, index=cont_dates_win)

    common_dates = obs_series.index.intersection(cont_series.index)
    obs_aligned = obs_series.loc[common_dates].values
    cont_aligned = cont_series.loc[common_dates].values

    valid = np.isfinite(obs_aligned) & np.isfinite(cont_aligned)

    nse_val, kge_val = np.nan, np.nan
    if valid.sum() > 1:
        obs_valid = obs_aligned[valid]
        cont_valid = cont_aligned[valid]

        ax2.plot(
            common_dates[valid],
            obs_valid,
            "k-",
            lw=1.5,
            markersize=6,
            label="Observed"
        )

        nse_val = nse(cont_valid, obs_valid)
        kge_val = kge(cont_valid, obs_valid)
    
    ax2.plot(
        cont_dates_win,
        cont_vals_win,
        "-",
        color="red",
        lw=1.5,
        markersize=6,
        label="Continuous Simulation"
    )
    ax2.text(
        0.98, 0.98, # position of these nse kge values
        f"NSE = {nse_val:.3f}\nKGE = {kge_val:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax2.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax2.set_title(f"Continuous Run — Gage {gage_id} ({gage_name})")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Streamflow (mm/day)")
    ax2.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.legend(loc="upper center")
    fig2.tight_layout()

    # ---------- Save ----------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if file_prefix is None:
            file_prefix = f"GAGE_{gage_id}"

        top_path = os.path.join(save_dir, f"{file_prefix}_forecast.png")
        bottom_path = os.path.join(save_dir, f"{file_prefix}_continuous.png")

        fig1.savefig(top_path, dpi=300, bbox_inches="tight")
        fig2.savefig(bottom_path, dpi=300, bbox_inches="tight")
 
        print(f"Saved bottom figure to: {bottom_path}")

    plt.show()
    print(f"Saved plot for basin {gage_id} → {save_dir}")
    plt.close(fig1)
    plt.close(fig2)
    
 
 
def evaluate_one_basin(model, dataset, basin_idx, device, nmul, staind, tdRep, routing, dydrop):
    #  
    x_phy = torch.nan_to_num(
        dataset["x_phy"][:, basin_idx:basin_idx+1, :3],
        nan=0.0
    ).to(device)     # [T, 1, 3]

    # ---------------- Observed discharge ----------------
    obs = dataset["target"][:, basin_idx, 0].cpu().numpy()   # [T]

    # ---------------- NN → HBV parameters ----------------
    hist_dict = {
        "xc_nn_norm": dataset["xc_nn_norm"][:, basin_idx:basin_idx+1].clone()
        # [T, 1, 38]
    }

    pars, rtwts = get_parameters_from_model(
        model.model_dict["Hbv_1_1p"],
        hist_dict,
        n_par=14,
        mu=nmul,
        device=device,
    )
    
    hbv = h1pp.HBV().to(device) 
    # ---------------- Run YOUR HBV1_1p ----------------
    with torch.no_grad():
        Qs = hbv(
            x=x_phy,          # [T, 1, 3]
            parameters=pars,  # [T, 1, 14, 16]
            staind=staind,
            tdlst=tdRep,
            mu=nmul,
            muwts=None,
            rtwts=rtwts,      # [1,2]
            bufftime=0,
            instate=False,
            outstate=False,
            routOpt=routing,
            dydrop=dydrop,
        )

    sim = Qs[:, 0, 0].detach().cpu().numpy()

    nse_val = nse(sim, obs)
    kge_val = kge(sim, obs)
    return nse_val, kge_val

def print_selected_basin_metrics_from_json(metrics, selected_basins, basin_pool):
    sumnse, sumkge = 0.0, 0.0
    count = 0

    for GAGE_ID in selected_basins:
        print(f"For basin {GAGE_ID} ===")
        basin_idx = basin_pool.index(GAGE_ID)

        nse_val = metrics['nse'][basin_idx]
        kge_val = metrics['kge'][basin_idx]
        print(f"NSE={nse_val}, KGE={kge_val}")

        if nse_val is None or kge_val is None:
            print(f"Skip basin {GAGE_ID} because metric is None")
            continue

        sumnse += float(nse_val)
        sumkge += float(kge_val)
        count += 1

    if count > 0:
        print(f"\nAverage over {count} valid selected basins: NSE={sumnse/count}, KGE={sumkge/count}")
    else:
        print("\nNo valid basins found.")
        

def print_and_average_selected_basin_metrics(
    model,
    dataset,
    selected_basins,
    basin_pool,
    device,
    nmul,
    staind,
    tdRep,
    routing,
    dydrop,
):
    sumnse, sumkge = 0.0, 0.0
    count = 0

    for GAGE_ID in selected_basins:
        print(f"For basin {GAGE_ID} ===")
        basin_idx = basin_pool.index(GAGE_ID)

        nse_val, kge_val = evaluate_one_basin(
            model, dataset, basin_idx, device, nmul, staind, tdRep, routing, dydrop
        )
        print(f"NSE={nse_val}, KGE={kge_val}")

        if (
            nse_val is None or kge_val is None
            or np.isnan(nse_val)
            or np.isnan(kge_val)
        ):
            print(f"Skip basin {GAGE_ID} because metric is None or NaN")
            continue
        sumnse += float(nse_val)
        sumkge += float(kge_val)
        count += 1

    if count > 0:
        avg_nse = sumnse / count
        avg_kge = sumkge / count
        print(f"\nAverage over {count} valid basins: NSE={avg_nse}, KGE={avg_kge}")
        return avg_nse, avg_kge
    else:
        print("\nNo valid basins found.")
        return None, None

def cleannans(metrics, METRIC):

    metric_vals = metrics[METRIC]

    clean_metric_vals = [
        float(x) for x in metric_vals
        if x is not None and not np.isnan(float(x))
    ]

    metrics_clean = {METRIC: clean_metric_vals}
    return metrics_clean


 