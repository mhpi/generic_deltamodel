import numpy as np
import pandas as pd
import json
import torch 
import os 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.dmg.core.utils import Dates
from hydrodl2.models.hbv import HBV1_1p as h1pp

def plot_ensemble_forecast(
    gage_id,
    start_date,
    obs,
    sim,
    ens_preds,
    history_len=60,
    save_path=None,
):
    """
    Plot 2 months history + 15-day forecast:
      - Observed (black)
      - Simulation (red) for history
      - 5 ensembles in distinct colors for forecast
    """
    print("history_len:", history_len)
    print("forecast_len:", forecast_len)
    print("dates len:", len(dates))
    print("obs len:", len(obs))

    forecast_len = ens_preds.shape[1]
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + forecast_len,
    )

    # Squeeze arrays
    obs = np.squeeze(obs)
    sim = np.squeeze(sim)

    # Build full series for sim (only history is valid)
    sim_full = np.full(history_len + forecast_len, np.nan)
    sim_full[:history_len] = sim[-history_len:]  # last 60 days

    # Pad ensembles for plotting (only forecast period valid)
    padded_ensembles = np.full((ens_preds.shape[0], history_len + forecast_len), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    # Plot
    plt.figure(figsize=(12, 6))

    # Observed
    plt.plot(dates, obs, "k-", lw=2, label="Observed")

    # Simulation (history only)
    plt.plot(dates, sim_full, "r-", lw=1.5, label="Simulation (history)")

    # Ensembles in distinct colors
    colors = cm.tab10.colors  # 10 distinct colors
    for i in range(ens_preds.shape[0]):
        plt.plot(
            dates,
            padded_ensembles[i],
            lw=1.5,
            color=colors[i % len(colors)],
            label=f"Ensemble {i+1}",
        )

    # Shading for history vs forecast
    plt.axvspan(dates[0], dates[history_len - 1], color="gray", alpha=0.1, label="History")
    plt.axvspan(dates[history_len], dates[-1], color="orange", alpha=0.1, label="Forecast")

    plt.title(f"GEFS Forecast — Gage {gage_id}")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (mm/day)")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_forecast_stacked(
    GAGE_NAME_PATH,
    gage_id,
    start_date,
    obs,                 # length = history_len + HORIZON
    sim,                 # length = history_len + 1
    ens_preds,           # shape = [n_ens, HORIZON]
    history_len,
    Q_all_np,            # continuous output from long run (length = T_total - BUFFTIME)
    sidx,                # input index of forecast start in the dataset timeline
    FORECAST,
    BUFFTIME,
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
        start=dataset_day0 + pd.Timedelta(days=BUFFTIME),
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
 

 # To call this function
    # plot_forecast_stacked(
    #     GAGE_NAME_PATH=GAGE_NAME_PATH,
    #     gage_id=GAGE_ID,
    #     start_date=ENSEMBLE_START_DATE,
    #     obs=obs_full_window,
    #     sim=sim_pre_GEFS,
    #     ens_preds=ens_preds,
    #     history_len=history_len,
    #     Q_all_np=Q_all_np,          # continuous curve
    #     sidx=sidx,
    #     FORECAST=FORECAST,
    #     BUFFTIME=BUFFTIME,
    #     det_pred=Qf_det,            # restart deterministic
    #     save_path=save_path,
    # )
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


def get_parameters_from_model(dpl_model, data, n_par, mu, device="cpu"):
    nn_model = dpl_model.nn_model
    # print("nn model is", dpl_model.nn_model)
    # print("xc_nn_norm shape before NN:", data["xc_nn_norm"].shape)
    xc_nn = data["xc_nn_norm"].to(device) # make sure it is on GPU

    with torch.no_grad():
        out = nn_model(xc_nn)   # only time-series input
        if isinstance(out, (tuple, list)):
            out = out[0]
 
        T, B, F = out.shape # zhennan comments: T: time dimension, B: # of basins, F: number of output features
        # print(f"Output shape from NN: T={T}, B={B}, F={F}") # 715, 1, 226
        
        if F == n_par * mu + 2: # F = 14*16 + 2 -> 226
            flat = out[:, :, : n_par * mu]
            pars = flat.view(T, B, n_par, mu).to(device)
            rts  = out[-1, :, n_par * mu : n_par * mu + 2].to(device)  
        else:
            raise RuntimeError(
                f"Unexpected feature size {F}, expected {n_par}, {n_par*mu}, or {n_par*mu+2}"
            )
            
        # sanity check
        # print(f"pars shape: {pars.shape}") # [715, 1, 14, 16]
        # print(f"rts shape: {rts.shape}") # [1, 2]
                 
    return pars, rts # pars: [T, B, n_par, mu], and rts: [B, 2]

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

def to_time_first(x_torch, device):
    return x_torch.permute(1, 0, 2).float().to(device)  # [T,B,F]

def safe_minmax(tensor):
    """Return min and max ignoring NaNs."""
    if tensor.numel() == 0:
        return np.nan, np.nan
    safe_min = torch.where(torch.isnan(tensor),
                           torch.tensor(float("inf"), device=tensor.device),
                           tensor)
    min_val = float(torch.min(safe_min))
    safe_max = torch.where(torch.isnan(tensor),
                           torch.tensor(float("-inf"), device=tensor.device),
                           tensor)
    max_val = float(torch.max(safe_max))
    return min_val, max_val

def checknans(warm_states): 
    for name, st in zip(["sp","mw","sm","suz","slz"], warm_states):
        smin, smax = safe_minmax(st)
        print(name, "nan#", torch.isnan(st).sum().item(),
              "min", smin, "max", smax)
        
def GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date, showblock=False):
    if not idx_list: raise ValueError(f"Starting GEFS date {start_date.date()} not exist!")
    if len(fc_block) < horizon: raise ValueError("Extracted GEFS forcing not enough!")
    if showblock:
        print("\n========= GEFS Forecast Block (NO ERROR) =========")
        print(fc_block.to_string(index=False))
         

def run_warm_forecasts(
    hbv, pars_last, rtwts_hist, warm_states,
    gage_id, start_date, horizon, varF,
    N_ENSEMBLES, GEFS_DIR, timesteps, WIDNOW, CORRECTION, staind, tdRep, nmul, routing, dydrop, device
):
    ens_preds = []
      # will store per-basin correction once

    for ens_id in range(N_ENSEMBLES):
        # ----- Load GEFS forecast for this ensemble -----
        f_path = os.path.join(GEFS_DIR, f"ens0{ens_id+1}", f"{gage_id:08d}.txt")
        if not os.path.exists(f_path):
            raise ValueError(f"Missing GEFS file: {f_path}")

        df = pd.read_csv(f_path, sep=r"\s+", header=0)
        df = df.rename(columns={
            "Year": "year", "Mnth": "month", "Day": "day",
            "prcp(mm/day)": "prcp", "tmean(C)": "tmean", "pet(mm/day)": "pet"
        })
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

        # ----- Extract forecast block -----
        idx_list = df.index[df["date"] == start_date].to_list()
        if not idx_list:
            raise ValueError(f"Start date {start_date} not found in {f_path}")
        fc_block = df.iloc[idx_list[0] : idx_list[0] + horizon]
        GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date)

        # ----- Compute bias correction once (for the first ensemble) -----
        if not CORRECTION:
            bias_corrections = compute_bias_correction_from_dataset(
                dataset["x_phy"], df, timesteps, basin_idx, window=WINDOW, method=CORRECTION  # or "scalar"
            )
            print(f"Bias correction computed for basin {gage_id}") 
            # ----- Apply bias correction -----
            fc_block = pre_processing(fc_block, bias_corrections, method=CORRECTION)

        # ----- Convert forcings to tensor -----
        raw_np = fc_block[varF].to_numpy().astype(np.float32)
        forc_raw = torch.tensor(raw_np[np.newaxis, :, :], dtype=torch.float32, device=device)
        forc_raw[torch.isnan(forc_raw)] = 0.0

        # ----- Run HBV forecast with warm states -----
        with torch.no_grad():
            Qs_fc = hbv(
                x=to_time_first(forc_raw, device),
                parameters=pars_last,
                staind=staind, tdlst=tdRep,
                mu=nmul,
                muwts=None, rtwts=rtwts_hist,
                bufftime=0,
                outstate=False, instate=True,
                init_states=warm_states,
                routOpt=routing, dydrop=dydrop,
            )

        # ----- Sanity check and store -----
        ens_fc = Qs_fc[:, 0, 0].detach().cpu().numpy()
        if np.isnan(ens_fc).all():
            raise ValueError(f"ens_fc are all NaNs for basin {gage_id}, ensemble {ens_id}!")

        ens_preds.append(ens_fc)

    return np.array(ens_preds)  # shape: (N_ENSEMBLES, horizon)

def selectbasins(RANDOM, SEED):
    with open("./predownloaded/Best_two_in_region.json", "r") as f:
        basin_groups = json.load(f)

    # flatten list
    basin_pool = [b for group in basin_groups.values() for b in group]
    
    #print(basin_pool)
    if RANDOM:
        random.seed(SEED)
        selected_basins = random.sample(basin_pool, N_BASINS)
        # print("Randomly selected basins:", selected_basins)
    else:
        selected_basins = basin_pool
        # print("Selected basins from JSON:", selected_basins)

    return selected_basins

def startid_endid(config, ENSEMBLE_START_DATE, FORECAST):
    timesteps = Dates(config["simulation"], config["delta_model"]["rho"]).batch_daily_time_range # 730
    print("check on config", config["simulation"])
    sidx = np.where(timesteps == ENSEMBLE_START_DATE)[0][0] # 715
    eidx = sidx + FORECAST # 730 
    history_len = len(timesteps) - config['delta_model']['phy_model']['warm_up'] - FORECAST
    print("sidx, eidx, timesteps, history_len are", sidx, eidx, timesteps, history_len)
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
    BUFFTIME,
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
        start=dataset_day0 + pd.Timedelta(days=BUFFTIME),
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

    ax1.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax1.set_title(f"Forecast — Gage {gage_id} ({gage_name})")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Streamflow (mm/day)")
    ax1.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

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

    ax2.plot(
        cont_dates_win,
        cont_vals_win,
        "-",
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

    ax2.text(
        0.98, 0.98,
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
    ax2.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    ax2.legend(loc="upper left")
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

        print(f"Saved top figure to: {top_path}")
        print(f"Saved bottom figure to: {bottom_path}")

    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    

def plot_ensemble_forecast(GAGE_NAME_PATH, gage_id, start_date, obs, sim, ens_preds, history_len, save_path=None):
    """
    Plot 2 months history + 15-day forecast:
      - Observed (black)
      - Simulation (red) for history
      - Ensembles (all black, single label "Ensembles")
      - Grey shaded envelope for ensembles
    """ 
    print("the length is here", len(sim), len(obs))
    ## Get the gage name first
    gage_name = obtain_gage_name(GAGE_NAME_PATH, gage_id)
    
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

    # ======== Ensemble Envelope (Gray Shaded Area) ========
    ens_min = np.nanmin(padded_ensembles, axis=0)
    ens_max = np.nanmax(padded_ensembles, axis=0)

    # Plot 
    plt.figure(figsize=(12, 6))

    # Observed
    plt.plot(dates_obs, obs, "k-", lw=2, label="Observed")

    # Simulation (history only)
    plt.plot(dates_sim, sim, "r-", lw=2, label="Simulation")

    # Gray shading for ensemble envelope
    plt.fill_between(
        dates_obs,
        ens_min,
        ens_max,
        color="blue",
        alpha=0.1,
        label="Ensembles"   # SINGLE legend entry
    )

    # All ensemble curves (black, no labels)
    for i in range(ens_preds.shape[0]):
        plt.plot(
            dates_obs,
            padded_ensembles[i],
            lw=1.0,
            color="darkblue"
        )

    # Shading for history vs forecast
    plt.axvspan(dates_obs[0], dates_obs[history_len], color="gray", alpha=0.1, label="Pre-GEFS Simulation")
    plt.axvspan(dates_obs[history_len], dates_obs[-1], color="orange", alpha=0.1, label="GEFS Forecast")

    plt.title(f"GEFS Forecast — Gage {gage_id} ({gage_name})")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (mm/day)")
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

    # Legend (automatically includes only unique labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=int(np.ceil(len(labels)/4)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    

# ====================================================
# Evaluate ONE basin (HBV1_1p manual forward)
# ====================================================
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
        # print(f"For basin {GAGE_ID} ===")
        basin_idx = basin_pool.index(GAGE_ID)

        nse_val = metrics['nse'][basin_idx]
        kge_val = metrics['kge'][basin_idx]
        # print(f"NSE={nse_val}, KGE={kge_val}")

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



# # If you want to plot a figure based on dmg hydrodl HBV1.1 of the last gage id you selected previously
# obs_full_window = (
#         dataset["target"][sidx - history_len : eidx, basin_idx, 0]
#         .detach().cpu().numpy()
#     )
# sim_pre_GEFS = (
#         output['Hbv_1_1p']['streamflow'][sidx - history_len - BUFFTIME : sidx - BUFFTIME + 1, basin_idx]
#         .detach().cpu().numpy()
#     )

# save_path = f"figs/GEFS_{GAGE_ID}_dmgHBV.png"
# plot_ensemble_forecast(
#     GAGE_NAME_PATH=GAGE_NAME_PATH,
#     gage_id=GAGE_ID,
#     start_date=ENSEMBLE_START_DATE,
#     obs=obs_full_window,
#     sim=sim_pre_GEFS,
#     ens_preds=ens_preds,
#     history_len=history_len,
#     save_path=save_path,
# )
# print(f"Saved plot for basin {GAGE_ID} → {save_path}")