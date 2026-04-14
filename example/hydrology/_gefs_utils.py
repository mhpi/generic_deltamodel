"""Utilities for GEFS-based hydrology forecasting example.

See ./example/hydrology/example_dhbv_1_1p_gefs.ipynb.

NOTE: may be formally adopted into dmg at a later time.

@ZhennanShi1
"""

import os
import random
from typing import Any, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dmg.core.utils import Dates

### Metrics


def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


### Dataset helpers


def print_dataset_info(dataset: dict[str, torch.Tensor]) -> None:
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


def get_parameters_from_model(
    dpl_model: Any,
    data: dict[str, torch.Tensor],
    n_par: int,
    mu: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Obtain parameters from a dPL model's neural network.

    Handles three possible output shapes from the NN:

    - F == n_par: no ensemble, no routing weights.
    - F == n_par * mu: ensemble parameters, no routing weights.
    - F == n_par * mu + 2: ensemble parameters + routing weights.

    Parameters
    ----------
    dpl_model
        Instantiated DplModel containing a nn_model attribute.
    data
        Dataset dict containing xc_nn_norm.
    n_par
        Number of HBV parameters.
    mu
        Number of ensemble parameter sets (nmul).
    device
        Device to move output tensors to.

    Returns
    -------
    tuple[torch.Tensor, Optional[torch.Tensor]]
        (pars, rts) where pars has shape [T, B, n_par, mu]
        and rts has shape [B, 2] or is None if not present.
    """
    nn_model = dpl_model.nn_model
    xc_nn = data["xc_nn_norm"].to(device)

    with torch.no_grad():
        out = nn_model(xc_nn)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.dim() != 3:
            raise RuntimeError(f"Unexpected output rank {out.dim()} (expected 3)")

        T, B, F = out.shape

        if F == n_par:
            pars = out.unsqueeze(-1).to(device)
            rts = None
        elif F == n_par * mu:
            pars = out.view(T, B, n_par, mu).to(device)
            rts = None
        elif F == n_par * mu + 2:
            flat = out[:, :, : n_par * mu]
            pars = flat.view(T, B, n_par, mu).to(device)
            rts = out[-1, :, n_par * mu : n_par * mu + 2].to(device)
        else:
            raise RuntimeError(
                f"Unexpected feature size {F}, expected {n_par}, "
                f"{n_par * mu}, or {n_par * mu + 2}",
            )

    return pars, rts  # pars: [T, B, n_par, mu], rts: [B, 2] or None


### Gage metadata


def obtain_gage_name(GAGE_NAME_PATH: str, gage_id: int) -> str:
    """Obtain gage name from gage ID using the provided CSV file.

    Parameters
    ----------
    GAGE_NAME_PATH
        Path to a semicolon-delimited CSV with columns gauge_id and
        gauge_name.
    gage_id
        USGS gage ID to look up.

    Returns
    -------
    str
        Human-readable gage name.
    """
    gage_names = pd.read_csv(GAGE_NAME_PATH, sep=";", dtype={"gauge_id": str})
    gage_names['gauge_id'] = gage_names['gauge_id'].str.lstrip('0')
    match = gage_names.loc[gage_names['gauge_id'] == str(gage_id), 'gauge_name']
    if match.empty:
        raise ValueError(f"GAGE_ID {gage_id} not found in {GAGE_NAME_PATH}")
    return match.values[0]


### Bias correction


def compute_bias_correction_from_dataset(
    daymet_tensor: torch.Tensor,
    gefs_df: pd.DataFrame,
    timesteps: np.ndarray,
    basin_idx: int,
    window: int = 15,
    method: str = "scalar",
) -> dict[str, dict[int, Any]]:
    """Compute bias correction factors for GEFS forcings using Daymet climatology.

    Uses a moving day-of-year window to compute either scalar ratios or CDF
    quantile mappings between GEFS and Daymet climatologies.

    Parameters
    ----------
    daymet_tensor
        Historical Daymet forcings of shape [time, basin, 3]
        with channels (prcp, tmean, pet).
    gefs_df
        GEFS historical forcings with columns: date, prcp, tmean,
        pet.
    timesteps
        Daily time axis corresponding to the Daymet tensor.
    basin_idx
        Basin index to select from the Daymet tensor.
    window
        +/-days around each DOY used to compute moving-window statistics.
    method
        Bias correction method: "scalar" for mean ratio (Daymet / GEFS),
        or "cdf" for quantile mapping from GEFS to Daymet.

    Returns
    -------
    dict[str, dict[int, Any]]
        corrections[var][doy] is a scalar ratio when method="scalar",
        or a (q_gefs, q_daymet) tuple when method="cdf".
    """
    df_daymet = pd.DataFrame(
        {
            "date": pd.to_datetime(timesteps),
            "prcp": daymet_tensor[:, basin_idx, 0].detach().cpu().numpy(),
            "tmean": daymet_tensor[:, basin_idx, 1].detach().cpu().numpy(),
            "pet": daymet_tensor[:, basin_idx, 2].detach().cpu().numpy(),
        }
    )
    df_daymet["doy"] = df_daymet["date"].dt.dayofyear

    gefs_df = gefs_df.copy()
    gefs_df["doy"] = gefs_df["date"].dt.dayofyear

    corrections = {v: {} for v in ["prcp", "tmean", "pet"]}

    for var in ["prcp", "tmean", "pet"]:
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
                p_grid = np.linspace(0, 100, 101)
                q_daymet = np.percentile(vals_d, p_grid)
                q_gefs = np.percentile(vals_g, p_grid)
                corrections[var][doy] = (q_gefs, q_daymet)
            else:
                raise ValueError(f"Unknown correction method '{method}'")

    return corrections


def pre_processing(
    df: pd.DataFrame,
    corrections: dict[str, dict[int, Any]],
    method: str = "scalar",
) -> pd.DataFrame:
    """Apply bias corrections to a GEFS forecast DataFrame.

    Parameters
    ----------
    df
        GEFS forecast DataFrame with columns date, prcp, tmean,
        pet.
    corrections
        Correction factors as returned by
        :func:`compute_bias_correction_from_dataset`.
    method
        Bias correction method: "scalar" or "cdf".

    Returns
    -------
    pd.DataFrame
        Copy of df with bias-corrected forcing columns.
    """
    df = df.copy()
    df["doy"] = df["date"].dt.dayofyear

    for var in ["prcp", "tmean", "pet"]:
        if method == "scalar":
            df[var] *= df["doy"].map(lambda d, v=var: corrections[v].get(d, 1.0))
        elif method == "cdf":
            corrected_vals = []
            for _, row in df.iterrows():
                doy = row["doy"]
                val = row[var]
                if doy not in corrections[var]:
                    corrected_vals.append(val)
                    continue
                q_in, q_out = corrections[var][doy]
                corrected_vals.append(np.interp(val, q_in, q_out))
            df[var] = corrected_vals
        else:
            raise ValueError(f"Unknown correction method '{method}'")

    return df


def post_processing(
    sim_pre_GEFS: np.ndarray,
    ens_preds: np.ndarray,
) -> np.ndarray:
    """Offset-correct ensemble forecasts to connect smoothly with prior simulation.

    Applies a per-ensemble constant offset so that the first forecast value
    matches the last simulated (Daymet) streamflow value, then clips at zero.

    Parameters
    ----------
    sim_pre_GEFS
        Simulated streamflow before the forecast window (Daymet-forced HBV),
        shape [T].
    ens_preds
        Raw ensemble forecast values, shape [N_ENSEMBLES, FORECAST].

    Returns
    -------
    np.ndarray
        Offset-corrected ensemble forecasts, shape [N_ENSEMBLES, FORECAST].
    """
    Q_sim_end = sim_pre_GEFS[-1]
    ens_preds_corrected = []
    for ens in ens_preds:
        offset = Q_sim_end - ens[0]
        ens_preds_corrected.append(np.maximum(ens + offset, 0.0))
    return np.array(ens_preds_corrected)


### Plotting


def plot_ensemble_hydrograph(
    gage_path: str,
    gage_id: int,
    start_date: str,
    obs: np.ndarray,
    sim: np.ndarray,
    ens_preds: np.ndarray,
    history_len: int,
    save_path: Optional[str] = None,
) -> None:
    """Plot ensemble hydrograph: history window + forecast period.

    Shows observed (black), simulation (red) for history, and each GEFS
    ensemble member in a distinct color for the forecast.

    Parameters
    ----------
    gage_path
        Path to the gage names CSV file.
    gage_id
        USGS gage ID.
    start_date
        Forecast start date (parseable by :func:`pandas.to_datetime`).
    obs
        Observed streamflow, shape [history_len + FORECAST].
    sim
        Simulated streamflow for the history window, shape [history_len + 1].
    ens_preds
        Ensemble forecast values, shape [N_ENSEMBLES, FORECAST].
    history_len
        Number of historical days to display.
    save_path
        File path to save the figure. If None, figure is only displayed.
    """
    print("the length is here", len(sim), len(obs))
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

    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    plt.figure(figsize=(12, 6))
    plt.plot(dates_obs, obs, "k-", lw=2, marker="*", label="Observed")
    plt.plot(dates_sim, sim, "r-", lw=1.5, marker="o", label="Simulation")

    colors = cm.tab10.colors
    for i in range(ens_preds.shape[0]):
        plt.plot(
            dates_obs,
            padded_ensembles[i],
            lw=1.5,
            color=colors[i % len(colors)],
            label=f"Ensemble {i + 1}",
        )

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

    plt.title(f"GEFS Forecast - Gage {gage_id} ({gage_name})")
    plt.xlabel("Date")
    plt.ylabel("Streamflow (mm/day)")
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper center", ncol=int(np.ceil(len(labels) / 4)))
    plt.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_forecast_separate(
    GAGE_NAME_PATH: str,
    gage_id: int,
    start_date: str,
    obs: np.ndarray,
    sim: np.ndarray,
    ens_preds: np.ndarray,
    history_len: int,
    sidx: int,
    FORECAST: int,
    WARMUPTIME: int,
    Q_all_np: Optional[np.ndarray] = None,
    det_pred: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    file_prefix: Optional[str] = None,
    plot_history_days: int = 60,
) -> None:
    """Save three separate forecast figures for a single basin.

    - **Figure 1**: Full history + forecast with ensemble spread and metrics.
    - **Figure 2**: Continuous Daymet-forced run vs observed (skipped if
      Q_all_np is None).
    - **Figure 3**: Zoomed-in forecast (plot_history_days of history +
      FORECAST days).

    Parameters
    ----------
    GAGE_NAME_PATH
        Path to the gage names CSV file.
    gage_id
        USGS gage ID.
    start_date
        Forecast start date (parseable by :func:`pandas.to_datetime`).
    obs
        Observed streamflow, shape [history_len + FORECAST].
    sim
        Simulated streamflow for the history window (Daymet-forced),
        shape [history_len].
    ens_preds
        Ensemble forecast values, shape [N_ENSEMBLES, FORECAST].
    history_len
        Number of historical days shown in the figure.
    sidx
        Index of the forecast start in the full dataset time axis.
    FORECAST
        Forecast horizon in days.
    WARMUPTIME
        Model warm-up length in days.
    Q_all_np
        Full continuous-run output. If None, Figure 2 is skipped.
    det_pred
        Optional deterministic restart prediction to overlay, shape
        [FORECAST].
    save_dir
        Directory to save figures. If None, figures are only displayed.
    file_prefix
        Filename prefix. Defaults to "GAGE_{gage_id}".
    plot_history_days
        Number of history days shown in Figure 3.
    """
    gage_name = obtain_gage_name(GAGE_NAME_PATH, gage_id)

    ens_preds = np.asarray(ens_preds)
    if ens_preds.ndim != 2:
        raise ValueError(f"ens_preds must be [n_ens, FORECAST], got {ens_preds.shape}")

    HORIZON = ens_preds.shape[1]
    start_date = pd.to_datetime(start_date)

    dates_obs_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len + HORIZON,
    )
    dates_sim_top = pd.date_range(
        start=start_date - pd.Timedelta(days=history_len),
        periods=history_len,
    )

    padded_ensembles = np.full((ens_preds.shape[0], history_len + HORIZON), np.nan)
    padded_ensembles[:, history_len:] = ens_preds

    ens_min = np.full(history_len + HORIZON, np.nan)
    ens_max = np.full(history_len + HORIZON, np.nan)
    if not np.all(np.isnan(ens_preds)):
        ens_min[history_len:] = np.nanmin(ens_preds, axis=0)
        ens_max[history_len:] = np.nanmax(ens_preds, axis=0)

    obs = np.asarray(obs).reshape(-1)
    sim = np.asarray(sim).reshape(-1)
    obs_hist = obs[:history_len]
    obs_fcst = obs[history_len : history_len + HORIZON]

    # Pre-forecast metrics
    pre_nse, pre_kge = np.nan, np.nan
    valid_pre = np.isfinite(sim) & np.isfinite(obs_hist)
    if valid_pre.sum() > 1:
        pre_nse = nse(sim[valid_pre], obs_hist[valid_pre])
        pre_kge = kge(sim[valid_pre], obs_hist[valid_pre])

    # Ensemble forecast metrics
    ens_nse_list, ens_kge_list = [], []
    for i in range(ens_preds.shape[0]):
        valid_i = np.isfinite(ens_preds[i]) & np.isfinite(obs_fcst)
        if valid_i.sum() > 1:
            ens_nse_list.append(nse(ens_preds[i][valid_i], obs_fcst[valid_i]))
            ens_kge_list.append(kge(ens_preds[i][valid_i], obs_fcst[valid_i]))
        else:
            ens_nse_list.append(np.nan)
            ens_kge_list.append(np.nan)

    metric_text = (
        f"Pre-Forecast\nNSE={pre_nse:.3f}, KGE={pre_kge:.3f}\n\nGEFS Ensembles\n"
    )
    for i in range(len(ens_nse_list)):
        metric_text += (
            f"E{i + 1}: NSE={ens_nse_list[i]:.3f}, KGE={ens_kge_list[i]:.3f}\n"
        )

    pad_days = 3
    x0, x1 = dates_obs_top[0], dates_obs_top[-1]
    plot_history_days = min(plot_history_days, history_len)
    x0_zoom = start_date - pd.Timedelta(days=plot_history_days)
    x1_zoom = start_date + pd.Timedelta(days=FORECAST - 1)

    has_det = det_pred is not None
    if has_det:
        det_pred = np.asarray(det_pred).reshape(-1)
        restart_dates = pd.date_range(start=start_date, periods=HORIZON)
        connector_dates = [dates_sim_top[-1], restart_dates[0]]
        connector_vals = [sim[-1], det_pred[0]]

    def _draw_forecast_axes(ax):
        ax.plot(dates_obs_top, obs, "k-", lw=2, label="Observed")
        ax.plot(dates_sim_top, sim, "r-", lw=1.5, label="Simulation (history)")
        ax.fill_between(
            dates_obs_top,
            ens_min,
            ens_max,
            color="green",
            alpha=0.1,
            label="Ensemble spread",
        )
        for i in range(ens_preds.shape[0]):
            ax.plot(dates_obs_top, padded_ensembles[i], lw=1.5, color="darkgreen")
        if has_det:
            ax.plot(connector_dates, connector_vals, color="purple", lw=1.5)
            ax.plot(
                restart_dates, det_pred, color="purple", lw=1.5, label="HBV Restart"
            )
        ax.text(
            0.98,
            0.98,
            metric_text,
            transform=ax.transAxes,
            va="top",
            ha="right",
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Streamflow (mm/day)")
        ax.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)

    # Figure 1: Full history + forecast
    fig1, ax1 = plt.subplots(figsize=(12, 5.5))
    _draw_forecast_axes(ax1)
    ax1.axvspan(
        dates_obs_top[0],
        dates_obs_top[history_len],
        color="gray",
        alpha=0.2,
        label="Pre-Forecast",
    )
    ax1.axvspan(
        dates_obs_top[history_len],
        dates_obs_top[-1],
        color="orange",
        alpha=0.2,
        label="Forecast",
    )
    ax1.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
    ax1.set_title(f"GEFS Forecast - Gage {gage_id} ({gage_name})")
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="upper center", ncol=max(1, int(np.ceil(len(l1) / 4))))
    fig1.tight_layout()

    # Figure 2: Continuous Daymet run (optional)
    fig2 = None
    if Q_all_np is not None:
        dataset_day0 = start_date - pd.Timedelta(days=sidx)
        dates_cont = pd.date_range(
            start=dataset_day0 + pd.Timedelta(days=WARMUPTIME),
            periods=len(Q_all_np),
        )
        fig2, ax2 = plt.subplots(figsize=(12, 4.8))
        mask = (dates_cont >= x0) & (dates_cont <= x1)
        cont_dates_win = dates_cont[mask]
        cont_vals_win = np.asarray(Q_all_np)[mask]

        obs_series = pd.Series(obs, index=dates_obs_top)
        cont_series = pd.Series(cont_vals_win, index=cont_dates_win)
        common_dates = obs_series.index.intersection(cont_series.index)
        obs_aligned = obs_series.loc[common_dates].values
        cont_aligned = cont_series.loc[common_dates].values
        valid = np.isfinite(obs_aligned) & np.isfinite(cont_aligned)

        nse_val, kge_val = np.nan, np.nan
        if valid.sum() > 1:
            nse_val = nse(cont_aligned[valid], obs_aligned[valid])
            kge_val = kge(cont_aligned[valid], obs_aligned[valid])
            ax2.plot(
                common_dates[valid], obs_aligned[valid], "k-", lw=1.5, label="Observed"
            )

        ax2.plot(
            cont_dates_win,
            cont_vals_win,
            "-",
            color="red",
            lw=1.5,
            label="Continuous Simulation",
        )
        if has_det:
            ax2.plot(connector_dates, connector_vals, color="purple", lw=1.5)
            ax2.plot(
                restart_dates, det_pred, color="purple", lw=1.5, label="HBV Restart"
            )
        ax2.text(
            0.98,
            0.98,
            f"NSE = {nse_val:.3f}\nKGE = {kge_val:.3f}",
            transform=ax2.transAxes,
            va="top",
            ha="right",
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
        )
        ax2.set_xlim(x0 - pd.Timedelta(days=pad_days), x1 + pd.Timedelta(days=pad_days))
        ax2.set_title(f"Continuous Run - Gage {gage_id} ({gage_name})")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Streamflow (mm/day)")
        ax2.grid(True, linestyle="--", linewidth=1.5, alpha=0.7)
        ax2.legend(loc="upper center")
        fig2.tight_layout()

    # Figure 3: Zoomed forecast
    fig3, ax3 = plt.subplots(figsize=(12, 5.5))
    _draw_forecast_axes(ax3)
    ax3.axvspan(x0_zoom, start_date, color="gray", alpha=0.2, label="Pre-Forecast")
    ax3.axvspan(start_date, x1_zoom, color="orange", alpha=0.2, label="Forecast")
    ax3.set_xlim(
        x0_zoom - pd.Timedelta(days=pad_days), x1_zoom + pd.Timedelta(days=pad_days)
    )

    # Auto y-range for zoomed window
    y_candidates = []
    mask_obs_zoom = (dates_obs_top >= x0_zoom) & (dates_obs_top <= x1_zoom)
    mask_sim_zoom = (dates_sim_top >= x0_zoom) & (dates_sim_top <= x1_zoom)
    if np.any(mask_obs_zoom):
        y_candidates.append(obs[mask_obs_zoom])
    if np.any(mask_sim_zoom):
        y_candidates.append(sim[mask_sim_zoom])
    for i in range(ens_preds.shape[0]):
        if np.any(mask_obs_zoom):
            y_candidates.append(padded_ensembles[i][mask_obs_zoom])
    if has_det:
        mask_det_zoom = (restart_dates >= x0_zoom) & (restart_dates <= x1_zoom)
        if np.any(mask_det_zoom):
            y_candidates.append(det_pred[mask_det_zoom])
        y_candidates.append(np.asarray(connector_vals))
    if y_candidates:
        y_all = np.concatenate([np.asarray(y).ravel() for y in y_candidates])
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size > 0:
            y_min, y_max = np.min(y_all), np.max(y_all)
            y_pad = (
                0.08 * (y_max - y_min) if y_max > y_min else max(0.1, 0.08 * abs(y_max))
            )
            ax3.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

    ax3.set_title(
        f"GEFS Forecast (Zoomed {plot_history_days}-Day History) - "
        f"Gage {gage_id} ({gage_name})"
    )
    h3, l3 = ax3.get_legend_handles_labels()
    ax3.legend(h3, l3, loc="upper center", ncol=max(1, int(np.ceil(len(l3) / 4))))
    fig3.tight_layout()

    # Save
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if file_prefix is None:
            file_prefix = f"GAGE_{gage_id}"
        fig1.savefig(
            os.path.join(save_dir, f"{file_prefix}_forecast.png"),
            dpi=300,
            bbox_inches="tight",
        )
        if fig2 is not None:
            fig2.savefig(
                os.path.join(save_dir, f"{file_prefix}_continuous.png"),
                dpi=300,
                bbox_inches="tight",
            )
        fig3.savefig(
            os.path.join(save_dir, f"{file_prefix}_forecast_zoom.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"Saved figures for basin {gage_id} -> {save_dir}")

    plt.show()
    plt.close("all")


### Tensor utilities


def to_time_first(x_torch: torch.Tensor, device: str) -> torch.Tensor:
    """Convert tensor from [B, T, F] to [T, B, F] and move to device.

    Parameters
    ----------
    x_torch
        Input tensor of shape [B, T, F].
    device
        Target device string (e.g., "cpu", "cuda").

    Returns
    -------
    torch.Tensor
        Permuted tensor of shape [T, B, F] on the target device.
    """
    return x_torch.permute(1, 0, 2).float().to(device)


def safe_minmax(tensor: torch.Tensor) -> tuple[float, float]:
    """Return (min, max) of a tensor, ignoring NaNs.

    Parameters
    ----------
    tensor
        Input tensor (any shape).

    Returns
    -------
    tuple[float, float]
        (min_value, max_value), or (nan, nan) for empty tensors.
    """
    if tensor.numel() == 0:
        return np.nan, np.nan
    safe_min = torch.where(
        torch.isnan(tensor),
        torch.tensor(float("inf"), device=tensor.device),
        tensor,
    )
    safe_max = torch.where(
        torch.isnan(tensor),
        torch.tensor(float("-inf"), device=tensor.device),
        tensor,
    )
    return float(torch.min(safe_min)), float(torch.max(safe_max))


### Diagnostic utilities


def checknans(warm_states: tuple[torch.Tensor, ...]) -> None:
    """Print NaN count and min/max for each HBV warm state.

    Parameters
    ----------
    warm_states
        Tuple of HBV state tensors (sp, mw, sm, suz, slz).
    """
    for name, st in zip(["sp", "mw", "sm", "suz", "slz"], warm_states):
        smin, smax = safe_minmax(st)
        print(name, "nan#", torch.isnan(st).sum().item(), "min", smin, "max", smax)


def GEFSdataErrorCheck(
    idx_list: list[int],
    fc_block: pd.DataFrame,
    horizon: int,
    start_date: str,
    showblock: bool = False,
) -> None:
    """Validate an extracted GEFS forecast block.

    Parameters
    ----------
    idx_list
        Row indices of start_date in the GEFS DataFrame.
    fc_block
        Extracted GEFS rows for the forecast window.
    horizon
        Required number of forecast days.
    start_date
        Forecast start date (used in error messages).
    showblock
        If True, print the full forecast block when validation passes.
    """
    if not idx_list:
        raise ValueError(f"Starting GEFS date {start_date.date()} not exist!")
    if len(fc_block) < horizon:
        raise ValueError("Extracted GEFS forcing not enough!")
    if showblock:
        print("\n========= GEFS Forecast Block (NO ERROR) =========")
        print(fc_block.to_string(index=False))


def cleannans(metrics: dict[str, list], METRIC: str) -> dict[str, list[float]]:
    """Return a copy of metrics with NaN and None values removed for one metric.

    Parameters
    ----------
    metrics
        Per-basin metrics dict (e.g., as loaded from metrics.json).
    METRIC
        Name of the metric to filter (e.g., "nse").

    Returns
    -------
    dict[str, list[float]]
        Dict with a single key METRIC mapping to the cleaned value list.
    """
    metric_vals = metrics[METRIC]
    clean = [float(x) for x in metric_vals if x is not None and not np.isnan(float(x))]
    return {METRIC: clean}


### Basin selection & timing


def selectbasins(
    rand: bool,
    seed: int,
    basin_pool: list[int],
    n_basins: int,
    basin: int = 2046000,
) -> list[int]:
    """Select basins for testing: random sample or a single fixed basin.

    Parameters
    ----------
    rand
        If True, randomly sample n_basins from basin_pool.
    seed
        Random seed for reproducibility.
    basin_pool
        Full list of available basin IDs.
    n_basins
        Number of basins to select (used when rand=True).
    basin
        Fixed basin ID used when rand=False.

    Returns
    -------
    list[int]
        Selected basin IDs.
    """
    if rand:
        random.seed(seed)
        selected_basins = random.sample(basin_pool, n_basins)
        print("Randomly selected basins:", selected_basins)
    else:
        print(f"Selecting a fixed basin {basin} for testing...")
        selected_basins = [basin]
        print("Fixed selected a basin:", selected_basins)
    return selected_basins


def startid_endid(
    start_date: pd.Timestamp,
    forecast: int,
    config: dict[str, Any],
    warmup: int = 0,
) -> tuple[int, int, np.ndarray, int]:
    """Get start/end time indices and history length for a forecast run.

    Parameters
    ----------
    start_date
        Forecast start date.
    forecast
        Forecast horizon in days.
    config
        Model configuration dict (must contain 'sim' and
        'model.rho' keys).
    warmup
        Model warm-up length in days (used to compute history_len).

    Returns
    -------
    tuple[int, int, np.ndarray, int]
        (sidx, eidx, timesteps, history_len) where sidx is the index
        of start_date in the simulation time axis, eidx = sidx +
        forecast, timesteps is the full daily time axis, and
        history_len = len(timesteps) - warmup - forecast.
    """
    timesteps = Dates(
        config["sim"],
        config["model"]["rho"],
    ).batch_daily_time_range
    sidx = np.where(timesteps == start_date)[0][0]
    eidx = sidx + forecast
    history_len = len(timesteps) - warmup - forecast
    return sidx, eidx, timesteps, history_len


### Forecasting


def run_warm_forecasts(
    hbv: Any,
    pars_last: torch.Tensor,
    rtwts_hist: torch.Tensor,
    warm_states: tuple[torch.Tensor, ...],
    gage_id: int,
    basin_idx: int,
    dataset: dict[str, torch.Tensor],
    start_date: pd.Timestamp,
    horizon: int,
    varF: list[str],
    N_ENSEMBLES: int,
    GEFS_DIR: str,
    timesteps: np.ndarray,
    WINDOW: int,
    CORRECTION: Optional[str],
    device: str,
    staind: int = -1,
    tdRep: Optional[list[int]] = None,
    nmul: int = 16,
    routing: bool = True,
    dydrop: float = 0.0,
) -> np.ndarray:
    """Run warm-started HBV ensemble forecasts using GEFS forcings.

    For each ensemble member, loads the corresponding GEFS forcing file,
    optionally applies bias correction, then runs HBV forward from the
    provided warm states.

    Parameters
    ----------
    hbv
        Instantiated HBV physics model (nn.Module).
    pars_last
        NN-generated HBV parameters repeated over the forecast horizon,
        shape [FORECAST, 1, n_par, mu].
    rtwts_hist
        Routing weights from the NN, shape [1, 2].
    warm_states
        HBV state tensors (sp, mw, sm, suz, slz) from end of history run.
    gage_id
        USGS gage ID (used to locate GEFS forcing files).
    basin_idx
        Basin index in the dataset.
    dataset
        Full model dataset (used for bias correction climatology).
    start_date
        Forecast start date.
    horizon
        Forecast horizon in days.
    varF
        Forcing variable names expected by HBV (e.g., ['prcp', 'tmean', 'pet']).
    N_ENSEMBLES
        Number of GEFS ensemble members to run.
    GEFS_DIR
        Root directory containing GEFS ensemble subdirectories.
    timesteps
        Full simulation time axis (for bias correction DOY alignment).
    WINDOW
        +/-day window for DOY-based bias correction.
    CORRECTION
        Bias correction method ("scalar", "cdf") or None to skip.
    device
        Target device string.
    staind
        HBV state index option.
    tdRep
        HBV time-delay representation list. Defaults to [1, 3, 13].
    nmul
        Number of ensemble parameter sets (mu).
    routing
        Whether to apply routing in HBV.
    dydrop
        HBV dynamic parameter dropout rate.

    Returns
    -------
    np.ndarray
        Ensemble forecast array, shape [N_ENSEMBLES, horizon].
    """
    if tdRep is None:
        tdRep = [1, 3, 13]

    ens_preds = []

    for ens_id in range(N_ENSEMBLES):
        f_path = os.path.join(GEFS_DIR, f"ens0{ens_id + 1}", f"{gage_id:08d}.txt")
        if not os.path.exists(f_path):
            raise ValueError(f"Missing GEFS file: {f_path}")

        df = pd.read_csv(f_path, sep=r"\s+", header=0)
        df = df.rename(
            columns={
                "Year": "year",
                "Mnth": "month",
                "Day": "day",
                "prcp(mm/day)": "prcp",
                "tmean(C)": "tmean",
                "pet(mm/day)": "pet",
            }
        )
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

        idx_list = df.index[df["date"] == start_date].to_list()
        if not idx_list:
            raise ValueError(f"Start date {start_date} not found in {f_path}")
        fc_block = df.iloc[idx_list[0] : idx_list[0] + horizon]
        GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date)

        if CORRECTION:
            bias_corrections = compute_bias_correction_from_dataset(
                dataset["x_phy"],
                df,
                timesteps,
                basin_idx,
                window=WINDOW,
                method=CORRECTION,
            )
            fc_block = pre_processing(fc_block, bias_corrections, method=CORRECTION)

        raw_np = fc_block[varF].to_numpy().astype(np.float32)
        forc_raw = torch.tensor(
            raw_np[np.newaxis, :, :],
            dtype=torch.float32,
            device=device,
        )
        forc_raw[torch.isnan(forc_raw)] = 0.0

        with torch.no_grad():
            Qs_fc = hbv(
                x=to_time_first(forc_raw, device),
                parameters=pars_last,
                staind=staind,
                tdlst=tdRep,
                mu=nmul,
                muwts=None,
                rtwts=rtwts_hist,
                bufftime=0,
                outstate=False,
                instate=True,
                init_states=warm_states,
                routOpt=routing,
                dydrop=dydrop,
            )

        ens_fc = Qs_fc[:, 0, 0].detach().cpu().numpy()
        if np.isnan(ens_fc).all():
            raise ValueError(
                f"ens_fc are all NaNs for basin {gage_id}, ensemble {ens_id}!"
            )
        ens_preds.append(ens_fc)

    return np.array(ens_preds)  # [N_ENSEMBLES, horizon]


def build_xc_nn_norm_forecast(
    data_loader: Any,
    basin_idx: int,
    fc_block: pd.DataFrame,
    device: str,
    verbose: bool = False,
) -> torch.Tensor:
    """Build normalized NN input tensor for a GEFS forecast block.

    Normalizes GEFS forcings and concatenates with static basin attributes,
    matching the format of the historical xc_nn_norm tensor.

    Parameters
    ----------
    data_loader
        DataLoader with nn_forcings, nn_attributes, to_norm(),
        and dataset attributes.
    basin_idx
        Basin index to extract static attributes for.
    fc_block
        GEFS forecast block DataFrame with forcing columns.
    device
        Target device string.
    verbose
        If True, print intermediate shapes for debugging.

    Returns
    -------
    torch.Tensor
        Normalized input tensor, shape [T, 1, n_features].
    """
    if verbose:
        print("===== BUILD XC_NN_NORM FORECAST =====")

    x_nn_fc = fc_block[data_loader.nn_forcings].to_numpy().astype(np.float32)  # [T, 3]
    x_nn_fc = x_nn_fc[:, np.newaxis, :]  # [T, 1, 3]
    if verbose:
        print("raw GEFS forcings shape:", x_nn_fc.shape)

    c_nn_fc = (
        data_loader.dataset["c_nn"][basin_idx : basin_idx + 1]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )  # [1, n_attr]
    if verbose:
        print("static attributes shape:", c_nn_fc.shape)

    x_nn_norm = data_loader.to_norm(x_nn_fc, data_loader.nn_forcings)
    c_nn_norm = data_loader.to_norm(c_nn_fc, data_loader.nn_attributes)
    if verbose:
        print("normalized forcings shape:", x_nn_norm.shape)
        print("normalized static attrs shape:", c_nn_norm.shape)

    c_nn_norm = np.repeat(
        np.expand_dims(c_nn_norm, 0), x_nn_norm.shape[0], axis=0
    )  # [T, 1, n_attr]
    xc_nn_norm_fc = np.concatenate((x_nn_norm, c_nn_norm), axis=2)  # [T, 1, n_features]
    if verbose:
        print("final xc_nn_norm_fc shape:", xc_nn_norm_fc.shape)

    return torch.tensor(xc_nn_norm_fc, dtype=torch.float32, device=device)


def run_warm_forecasts_restart(
    model: Any,
    state_path: str,
    gage_id: int,
    basin_idx: int,
    start_date: pd.Timestamp,
    horizon: int,
    N_ENSEMBLES: int,
    GEFS_DIR: str,
    data_loader: Any,
    timesteps: np.ndarray,
    WINDOW: int,
    CORRECTION: Optional[str],
    device: str,
    verbose: bool = False,
) -> np.ndarray:
    """Run GEFS ensemble forecasts by reloading saved model states for each member.

    Unlike :func:`run_warm_forecasts` (which uses pre-computed HBV states and
    parameters), this function reloads the full model checkpoint (LSTM + HBV
    states) saved by model.save_states() and runs the complete LSTM -> HBV
    pipeline for each ensemble member with GEFS forcings.

    Parameters
    ----------
    model
        Initialized ModelHandler instance.
    state_path
        Path to the saved model state file (.pt).
    gage_id
        USGS gage ID (used to locate GEFS forcing files).
    basin_idx
        Basin index in the dataset.
    start_date
        Forecast start date.
    horizon
        Forecast horizon in days.
    N_ENSEMBLES
        Number of GEFS ensemble members to run.
    GEFS_DIR
        Root directory containing GEFS ensemble subdirectories.
    data_loader
        DataLoader with dataset, nn_forcings, nn_attributes,
        and to_norm() attributes.
    timesteps
        Full simulation time axis (for bias correction DOY alignment).
    WINDOW
        +/-day window for DOY-based bias correction.
    CORRECTION
        Bias correction method ("scalar", "cdf") or None to skip.
    device
        Target device string.
    verbose
        If True, print intermediate shapes for debugging.

    Returns
    -------
    np.ndarray
        Ensemble forecast array, shape [N_ENSEMBLES, horizon].
    """
    ens_preds = []
    name = list(model.model_dict.keys())[0]
    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    phy_model.cache_states = True
    phy_model.warmup = 0
    phy_model.warmup_states = True

    for ens_id in range(N_ENSEMBLES):
        model.load_states(path=state_path)

        f_path = os.path.join(GEFS_DIR, f"ens0{ens_id + 1}", f"{gage_id:08d}.txt")
        if not os.path.exists(f_path):
            raise ValueError(f"Missing GEFS file: {f_path}")

        df = pd.read_csv(f_path, sep=r"\s+", header=0)
        df = df.rename(
            columns={
                "Year": "year",
                "Mnth": "month",
                "Day": "day",
                "prcp(mm/day)": "prcp",
                "tmean(C)": "tmean",
                "pet(mm/day)": "pet",
            }
        )
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])

        idx_list = df.index[df["date"] == start_date].to_list()
        if not idx_list:
            raise ValueError(f"Start date {start_date} not found in {f_path}")
        fc_block = df.iloc[idx_list[0] : idx_list[0] + horizon].copy()
        GEFSdataErrorCheck(idx_list, fc_block, horizon, start_date)

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

        # HBV physics forcing
        raw_np = fc_block[["prcp", "tmean", "pet"]].to_numpy().astype(np.float32)
        x_phy_fc = torch.tensor(
            raw_np[:, np.newaxis, :],
            dtype=torch.float32,
            device=device,
        )
        x_phy_fc[torch.isnan(x_phy_fc)] = 0.0

        # NN input (LSTM)
        xc_nn_norm_fc = build_xc_nn_norm_forecast(
            data_loader=data_loader,
            basin_idx=basin_idx,
            fc_block=fc_block,
            device=device,
            verbose=verbose,
        )

        with torch.no_grad():
            raw_nn_out = nn_model(xc_nn_norm_fc)
            fluxes = phy_model(
                x_dict={"x_phy": x_phy_fc},
                parameters=raw_nn_out,
            )

        ens_preds.append(fluxes["streamflow"][:, 0, 0].detach().cpu().numpy())

    return np.array(ens_preds)  # [N_ENSEMBLES, horizon]


def run_segment(
    model: Any,
    dataset: dict[str, torch.Tensor],
    basin_idx: int,
    device: str,
    name: Optional[str] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    warmup: int = 0,
    cache_states: bool = False,
    warmup_states: bool = True,
    state_path: Optional[str] = None,
) -> np.ndarray:
    """Run the LSTM -> HBV pipeline on a single basin time segment.

    Slices the dataset for one basin and time window, sets physics model
    options, optionally loads saved states, then runs a forward pass.

    Parameters
    ----------
    model
        Initialized ModelHandler instance.
    dataset
        Must contain 'xc_nn_norm' of shape [T, B, F] and
        'x_phy' of shape [T, B, F].
    basin_idx
        Basin index to slice from the dataset.
    device
        Target device string.
    name
        Key in model.model_dict. Defaults to the first key.
    start_idx
        Start of the time slice (Python slice semantics).
    end_idx
        End of the time slice (Python slice semantics).
    warmup
        Warm-up days to pass to the physics model.
    cache_states
        Whether the physics model should cache its end states.
    warmup_states
        Whether to use warm-up states during the forward pass.
    state_path
        If given, load saved model states before the forward pass.

    Returns
    -------
    np.ndarray
        Simulated streamflow for the requested segment (after warm-up),
        shape [T_out].
    """
    if name is None:
        name = list(model.model_dict.keys())[0]

    nn_model = model.model_dict[name].nn_model
    phy_model = model.model_dict[name].phy_model

    data_dict = {
        "xc_nn_norm": dataset["xc_nn_norm"][
            start_idx:end_idx, basin_idx : basin_idx + 1, :
        ].clone(),
        "x_phy": dataset["x_phy"][
            start_idx:end_idx, basin_idx : basin_idx + 1, :
        ].clone(),
    }

    phy_model.warmup = warmup
    phy_model.cache_states = cache_states
    phy_model.warmup_states = warmup_states

    model.eval()

    if state_path is not None:
        model.load_states(path=state_path)

    with torch.no_grad():
        xc = data_dict["xc_nn_norm"].to(device)
        x_phy = data_dict["x_phy"].to(device)
        nn_out = nn_model(xc)
        fluxes = phy_model(x_dict={"x_phy": x_phy}, parameters=nn_out)

    return fluxes["streamflow"][:, 0, 0].detach().cpu().numpy()


### Evaluation output


def print_selected_basin_metrics_from_json(
    metrics: dict[str, list],
    selected_basins: list[int],
    basin_pool: list[int],
) -> None:
    """Print NSE and KGE for a list of basins from a per-basin metrics dict.

    Parameters
    ----------
    metrics
        Per-basin metrics dict (e.g., loaded from metrics.json), with
        keys "nse" and "kge" mapping to per-basin value lists.
    selected_basins
        Basin IDs to print metrics for.
    basin_pool
        Ordered list of all basin IDs (used to resolve index from ID).
    """
    sumnse, sumkge = 0.0, 0.0
    count = 0

    for gage_id in selected_basins:
        print(f"Basin {gage_id}:")
        basin_idx = basin_pool.index(gage_id)
        nse_val = metrics["nse"][basin_idx]
        kge_val = metrics["kge"][basin_idx]
        print(f"  NSE={nse_val}, KGE={kge_val}")

        if nse_val is None or kge_val is None:
            print("  Skipped (metric is None)")
            continue

        sumnse += float(nse_val)
        sumkge += float(kge_val)
        count += 1

    if count > 0:
        print(
            f"\nMean over {count} valid basins: NSE={sumnse / count:.4f}, KGE={sumkge / count:.4f}"
        )
    else:
        print("\nNo valid basins found.")
