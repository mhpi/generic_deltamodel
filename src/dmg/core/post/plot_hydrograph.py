from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import r2_score

from dmg.core.data.data import timestep_resample
from dmg.core.utils.utils import format_resample_interval


def plot_hydrograph(
    timesteps: pd.DatetimeIndex,
    predictions: Union[NDArray[np.float32], torch.Tensor],
    obs: Union[NDArray[np.float32], torch.Tensor] = None,
    resample: Literal['D', 'W', 'M', 'Y'] = 'D',
    title=None,
    ylabel: str = 'Streamflow (mm/day)',
    minor_ticks: bool = False,
    figsize: tuple = (12, 6),
    fontsize: int = 12,
    dpi: int = 100,
) -> None:
    """Plot the hydrograph of model predictions and observations (if specified).

    Parameters
    ----------
    timesteps
        The timesteps of the predictions.
    predictions
        The model predictions.
    obs
        The observed streamflow values.
    resample
        The resampling interval for the data.
    title
        The title of the plot.
    ylabel
        The y-axis label.
    minor_ticks
        Whether to show minor ticks on the plot.
    figsize
        The figure size.
    fontsize
        The font size of the plot.
    dpi
        The resolution of the plot.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    if obs is None:
        obs = np.zeros_like(predictions)

    # Resample the data to the specified temporal resolution.
    data = pd.DataFrame(
        {
            'time': timesteps,
            'pred': predictions,
            'obs': obs,
        }
    )
    data = timestep_resample(data, resolution=resample, method='mean')

    plt.rcParams.update({'font.size': fontsize})

    # Create the figure.
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(
        data['time'],
        list(data['pred']),
        label='Prediction',
        marker='o',
        color='r',
    )

    if obs.mean() != 0:
        plt.plot(
            data['time'],
            list(data['obs']),
            label='Observation',
            marker='o',
            color='b',
        )

    plt.title(title)
    # plt.xlabel('Time')
    plt.xlabel(f"Time ({format_resample_interval(resample)})")
    plt.ylabel(ylabel)

    # plt.annotate(
    #     f"Prediction Interval: {format_resample_interval(resample)}",
    #     xy=(0.03, 0.9),
    #     xycoords='axes fraction',
    #     color='black',
    #     bbox=dict(
    #         boxstyle='round,pad=0.3',
    #         edgecolor='gray',
    #         facecolor='lightgray',
    #         alpha=0.5
    #     ),
    # )

    if obs.mean() != 0:
        plt.legend(
            loc='upper right',
            frameon=True,
        )

    plt.xticks(rotation=45)

    ax = plt.gca()  # Get the current axis

    if minor_ticks:
        ax.minorticks_on()

    # Align minor ticks with major ticks
    # ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # One minor tick between major ticks

    # Optionally adjust major tick locator based on resampling interval
    # from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    # from matplotlib.dates import DayLocator, WeekdayLocator, MonthLocator, YearLocator

    # len_data = len(data)
    # if 'D' in resample:
    #     ax.xaxis.set_major_locator(DayLocator(interval=len_data//5))
    # elif 'W' in resample:
    #     ax.xaxis.set_major_locator(WeekdayLocator(interval=len_data//10))
    # elif 'M' in resample:
    #     ax.xaxis.set_major_locator(MonthLocator(interval=1))
    # elif 'Y' in resample:
    #     ax.xaxis.set_major_locator(YearLocator(interval=1))

    # Add grid lines
    ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)

    plt.show()


def plot_hydrograph_grid(
    timesteps: pd.DatetimeIndex,
    predictions: Union[NDArray[np.float32], torch.Tensor],
    obs: Union[NDArray[np.float32], torch.Tensor] = None,
    resample: Literal['D', 'W', 'M', 'Y'] = 'D',
    titles: list = None,
    suptitle: str = None,
    ylabel: Union[str, list] = None,
    line_labels: tuple = ('Prediction', 'Observation'),  # Changed default
    colors: tuple = ('r', 'orange'),  # NEW: Added colors
    fill_obs: bool = True,  # NEW: Toggle for fill
    minor_ticks: bool = False,
    figsize: tuple = (12, 10),
    fontsize: int = 12,
    dpi: int = 100,
    save_path: str = None,
) -> None:
    """Plot hydrographs for multiple target variables in a grid layout."""
    # --- 1. Data Preparation ---
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    obs_available = obs is not None

    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    elif obs is None:
        obs = np.zeros_like(predictions)  # Create dummy data if None

    num_vars = predictions.shape[1]  # Number of target variables

    # --- 2. Grid Setup ---
    rows = (num_vars + 1) // 2
    if num_vars == 1:
        cols = 1
    else:
        cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes.flatten()  # Always flatten for easy indexing

    # --- 3. Plotting Loop ---
    for i in range(num_vars):
        data = pd.DataFrame(
            {
                'time': timesteps,
                'pred': predictions[:, i],
                'obs': obs[:, i],
            }
        )
        data = timestep_resample(data, resolution=resample, method='mean')

        ax = axes[i]

        # Plot Prediction
        ax.plot(
            data['time'], data['pred'], label=line_labels[0], color=colors[0], zorder=3
        )

        # Plot Observation (if available)
        if obs_available:
            ax.plot(
                data['time'],
                data['obs'],
                label=line_labels[1],
                color=colors[1],
                zorder=2,
            )

            # Fill area under observation curve
            if fill_obs:
                ax.fill_between(
                    data['time'], data['obs'], color=colors[1], alpha=0.3, zorder=1
                )

            # Add R2 score only if obs are available
            r2 = r2_score(data['obs'], data['pred'])
            ax.text(
                0.05,
                0.95,
                f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='top',
            )

            # Add legend only if obs are available
            ax.legend(loc='upper right', frameon=True)

        # --- 4. Styling ---
        ax.set_title(titles[i] if titles and i < len(titles) else f'Target {i + 1}')
        ax.set_xlabel(f"Time [{format_resample_interval(resample)}]")

        if isinstance(ylabel, list):
            ax.set_ylabel(ylabel[i] if i < len(ylabel) else None)
        else:
            ax.set_ylabel(ylabel)

        ax.grid(which='major', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.6)
        if minor_ticks:
            ax.minorticks_on()

        ax.tick_params(axis='x', rotation=45)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize + 2)

    # Hide unused subplots if num_vars is odd
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
