import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

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
