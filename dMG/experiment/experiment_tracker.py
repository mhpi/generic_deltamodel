"""
Track and record metrics during an experiment.

From Tadd Bindas (dMC) 2024.
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
import torch
import torch.nn as nn
from conf.config import Config
from core.calc.metrics import Metrics
from torch.utils.tensorboard.writer import SummaryWriter

log = logging.getLogger(__name__)



class Tracker(ABC):
    @abstractmethod
    def flush(self):
        raise NotImplementedError

    @abstractmethod
    def plot_all_time_series(
        self,
        pred,
        obs,
        gage_dict,
        time_range,
        gage_indices,
        mode,
        epoch: Optional[int] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def plot_box_fig(
        self,
        data: List[np.ndarray],
        label1: Optional[List[str]] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def plot_box_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        start_time: str,
        end_time: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def plot_cdf_fig(
        self,
        x_list,
        ax=None,
        title=None,
        legendlist=None,
        figsize=(8, 6),
        ref="None",
        clist=None,
        xlabel=None,
        ylabel=None,
        showDiff=None,
        xlimit=None,
        linespec=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def plot_cdf(
        self,
        metric: Dict[str, np.ndarray],
        zones: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def plot_parameter_distribution(self, x_data, y_data, x_label, y_label):
        raise NotImplementedError

    @abstractmethod
    def plot_time_series(
        self, prediction, observation, time_range, gage_id, name, mode, metrics
    ):
        raise NotImplementedError

    @abstractmethod
    def save_state(
        self, epoch: int, mini_batch: int, mlp: nn.Module, optimizer: nn.Module
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_metrics(
        self, pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def weight_histograms(self, model, step):
        raise NotImplementedError

    @abstractmethod
    def weight_histograms_linear(self, step, weights, layer_number):
        raise NotImplementedError

    @abstractmethod
    def write_metrics(self, zones: Optional[str] = None) -> None:
        raise NotImplementedError

    # @abstractmethod
    # def write_to_zarr(
    #     self,
    #     pred: np.ndarray,
    #     gage_ids: np.ndarray,
    #     time_range: pd.DatetimeIndex,
    #     start_time: str,
    #     end_time: str,
    # ) -> None:
    #     raise NotImplementedError


class ExperimentTracker(Tracker):
    def __init__(self, **kwargs):
        self.cfg: Config = kwargs["cfg"]
        self.tensorboard_save_path: Path = Path(self.cfg.params.save_path)

        # self.plot_path = self.tensorboard_save_path / "plots"
        # self.plot_path.mkdir(parents=True, exist_ok=True)

        # self.saved_model_path = self.tensorboard_save_path / "saved_models"
        # self.saved_model_path.mkdir(parents=True, exist_ok=True)

        # self.zarr_data_path = self.tensorboard_save_path / "zarr_data"
        # self.zarr_data_path.mkdir(parents=True, exist_ok=True)

        self.metrics = None
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_save_path))

    def flush(self):
        self.writer.flush()

    def plot_all_time_series(
        self,
        pred,
        obs,
        gage_dict,
        time_range,
        gage_indices,
        mode,
        epoch: Optional[int] = None,
    ):
        for i in range(pred.shape[0]):
            gage_idx = gage_indices[i]
            metrics = {"nse": self.metrics.nse[i]}
            fig = (
                self.plot_time_series(
                    pred[i],
                    obs[i],
                    time_range,
                    gage_dict["STAID"][gage_idx],
                    gage_dict["STANAME"][gage_idx],
                    mode,
                    metrics=metrics,
                ),
            )
            gage_id = gage_dict["STAID"][gage_idx]
            if epoch is not None:
                save_path = (
                    self.plot_path
                    / f"gage_{gage_id}_{mode}_epoch_{epoch}_hydrograph.png"
                )
            else:
                save_path = self.plot_path / f"gage_{gage_id}_{mode}_hydrograph.png"
            fig[0].savefig(save_path)
            plt.close(fig[0])
            self.flush()

    def plot_box_fig(
        self,
        data,
        label1=None,
    ):
        nrows = 1
        ncols = len(data)

        # Create a figure and a set of subplots
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(2 * ncols, 6), constrained_layout=True
        )

        # Check if axes is a single Axes object or an array of Axes, this is important when ncols or nrows is 1
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Iterate over each dataset and corresponding axis to plot the boxplot
        for ax, data, label in zip(axes.flat, data, label1):
            bp = ax.boxplot(data, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightsteelblue")
            ax.set_title(label)

        plt.tight_layout()
        return fig

    def plot_box_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        start_time: str,
        end_time: str,
    ):
        x_label = []
        data_box = []
        for k, v in metrics.items():
            data = v
            x_label.append(k)
            if data.size > 0:  # Check if data is not empty
                data = data[~np.isnan(data)]  # Remove NaNs
                if k == "NSE" or k == "KGE":
                    data = np.clip(
                        data, -1, None
                    )  # Clipping the lower bound to -1 for NSE and KGE
                data_box.append(data)
        fig = self.plot_box_fig(
            data_box,
            x_label,
        )
        fig.patch.set_facecolor("white")
        save_path = self.plot_path / f"model_{self.cfg.mode}_metrics_box_plot.png"
        fig.savefig(save_path)
        self.writer.add_figure(
            f"Model {self.cfg.mode} {start_time} - {end_time} Boxplot", fig
        )
        self.flush()

    def plot_cdf_fig(
        self,
        x_list,
        ax=None,
        title=None,
        legendlist=None,
        figsize=(8, 6),
        ref=None,
        clist=None,
        xlabel=None,
        ylabel=None,
        showDiff=None,
        xlimit=None,
        linespec=None,
    ):
        """
        Plot Cumulative Distribution Functions (CDFs) for a list of datasets.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        if clist is None:
            cmap = plt.get_cmap("jet")
            clist = cmap(np.linspace(0, 1, len(x_list)))

        for k, x in enumerate(x_list):
            xSort = np.sort(x[~np.isnan(x)])
            yRank = np.arange(len(xSort)) / float(len(xSort) - 1)

            medianValue = np.median(xSort)
            legStr = (
                f"{legendlist[k]} | Median NSE: {medianValue:.4f}"
                if legendlist is not None
                else f"Median: {medianValue:.4f}"
            )

            ax.plot(xSort, yRank, color=clist[k], label=legStr)

        ax.grid(True)
        if title is not None:
            ax.set_title(title, loc="center")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlimit is not None:
            ax.set_xlim(xlimit)

        if ref == "121":
            ax.plot([0, 1], [0, 1], "k", label="y=x")
        elif ref == "norm":
            xNorm = np.linspace(-5, 5, 1000)
            normCdf = stats.norm.cdf(xNorm, 0, 1)
            ax.plot(xNorm, normCdf, "k", label="Gaussian")

        ax.legend(loc="best", frameon=False)

        return fig, ax

    def plot_cdf(
        self,
        metric: Dict[str, np.ndarray],
        zones: str,
    ):
        metric_list = []
        x_label = []
        for k, v in metric.items():
            data = v
            if k == "NSE" or k == "KGE":
                data = np.clip(
                    data, -1, None
                )  # Clipping the lower bound to -1 for NSE and KGE
            metric_list.append(data)
            x_label.append(k)
        legendlist = [self.cfg.observations.name]
        # Step 3: Define optional parameters for customization
        boxPlotName = f"Model {self.cfg.mode} Performance ({self.cfg.test.start_time}) - {self.cfg.test.end_time})"
        ylabel = "Cumulative Frequency"
        figsize = (14, 8)  # Optional: Customize figure size
        fig, ax = self.plot_cdf_fig(
            x_list=metric_list,
            title=boxPlotName,
            clist=["orange", "darkblue", "red", "black"],
            legendlist=legendlist,
            figsize=figsize,
            xlabel=x_label,
            ylabel=ylabel,
            ref=None,
        )
        save_path = self.plot_path / f"model_{self.cfg.mode}_{x_label[0]}_cdf.png"
        fig.savefig(save_path)
        self.writer.add_figure(
            f"Model {self.cfg.mode} {x_label[0]} CDF | Zones {zones}", fig
        )
        self.flush()

    def plot_parameter_distribution(self, x_data, y_data, x_label, y_label):
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(x_data, y_data, color="blue", label="River Graph Edges")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.title(f"{x_label} vs {y_label}")
        return fig

    def plot_time_series(
        self, prediction, observation, time_range, gage_id, name, mode, metrics
    ):
        warmup = self.cfg.params.warmup
        fig = plt.figure(figsize=(10, 5))
        prediction_to_plot = prediction[warmup:]
        observation_to_plot = observation[warmup:]
        plt.plot(time_range[warmup:], observation_to_plot, label="Observation")
        plt.plot(time_range[warmup:], prediction_to_plot, label="Routed Streamflow")
        nse = metrics["nse"]
        if self.cfg.observations.name.lower() != "grdc":
            gage_id = str(gage_id).zfill(8)
        plt.title(
            f"{mode} time period Hydrograph - " f"GAGE ID: {gage_id} - Name: {name}"
        )
        plt.xlabel("Time (hours)")
        plt.ylabel(r"Discharge $m^3/s$")
        plt.legend(title=f"NSE: {nse:.4f}")

        return fig

    def save_state(
        self, epoch: int, mini_batch: int, mlp: nn.Module, optimizer: nn.Module
    ) -> None:
        mlp_state_dict = {key: value.cpu() for key, value in mlp.state_dict().items()}

        cpu_optimizer_state_dict = {}
        for key, value in optimizer.state_dict().items():
            if key == "state":
                cpu_optimizer_state_dict[key] = {}
                for param_key, param_value in value.items():
                    cpu_optimizer_state_dict[key][param_key] = {}
                    for sub_key, sub_value in param_value.items():
                        if torch.is_tensor(sub_value):
                            cpu_optimizer_state_dict[key][param_key][
                                sub_key
                            ] = sub_value.cpu()
                        else:
                            cpu_optimizer_state_dict[key][param_key][
                                sub_key
                            ] = sub_value
            elif key == "param_groups":
                cpu_optimizer_state_dict[key] = []
                for param_group in value:
                    cpu_param_group = {}
                    for param_key, param_value in param_group.items():
                        cpu_param_group[param_key] = param_value
                    cpu_optimizer_state_dict[key].append(cpu_param_group)
            else:
                cpu_optimizer_state_dict[key] = value

        state = {
            "model_state_dict": mlp_state_dict,
            "optimizer_state_dict": cpu_optimizer_state_dict,
            "rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        if mini_batch == -1:
            state["epoch"] = epoch + 1
            state["mini_batch"] = 0
        else:
            state["epoch"] = epoch
            state["mini_batch"] = mini_batch

        torch.save(
            state,
            self.saved_model_path / f"{self.cfg.name}"
            f"_epoch_{state['epoch']}"
            f"_mb_{state['mini_batch']}.pt",
        )

    def set_metrics(
        self, pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> None:
        self.metrics = Metrics(pred=pred, target=target)

    def weight_histograms(self, model, step):
        log.debug("Visualizing model weights...")
        for layer_number in range(len(model.layers)):
            layer = model.layers[layer_number]
            if isinstance(layer, nn.Linear):
                weights = layer.weight
                self.weight_histograms_linear(step, weights, layer_number)

    def weight_histograms_linear(self, step, weights, layer_number):
        try:
            flattened_weights = weights.flatten()
            tag = f"layer_{layer_number}"
            self.writer.add_histogram(
                tag, flattened_weights, global_step=step, bins="tensorflow"
            )
        except ValueError as e:
            log.error("Weights contain NaN values")
            raise e

    def write_metrics(self, zones: Optional[str] = None) -> None:
        if zones is not None:
            save_path = self.tensorboard_save_path / f"model_metrics_{zones}.json"
        else:
            save_path = self.tensorboard_save_path / "model_metrics.json"
        json_cfg = self.metrics.model_dump_json(indent=4)
        with save_path.open("w") as f:
            json.dump(json_cfg, f)

    # def write_to_zarr(
    #     self,
    #     pred: np.ndarray,
    #     gage_ids: np.ndarray,
    #     time_range: pd.DatetimeIndex,
    #     start_time: str,
    #     end_time: str,
    # ) -> None:
    #     pred_da = xr.DataArray(
    #         data=pred,
    #         dims=["gage_ids", "time"],
    #         coords={"gage_ids": gage_ids, "time": time_range},
    #     )
    #     ds = xr.Dataset(
    #         data_vars={"predictions": pred_da},
    #         attrs={
    #             "description": f"Predictions and obs for time period "
    #             f"{start_time} -"
    #             f" {end_time}"
    #         },
    #     )
    #     ds.to_zarr(
    #         self.zarr_data_path / f"{start_time}_{end_time}_validation_predictions",
    #         mode="w",
    #     )
