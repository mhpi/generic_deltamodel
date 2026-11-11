import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from dmg.core.data.data import extract_temporal_features, split_dataset_by_basin
from dmg.core.data.loader_utils import (
    flow_conversion,
    load_nn_data,
    load_norm_stats,
    normalize_data,
    to_tensor,
)
from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.load_nc import NetCDFDataset

try:
    from dmg.core.data.loaders.zarr_image_loader import ZarrStationImageLoader
    _ZARR_AVAILABLE = True
except ImportError:
    _ZARR_AVAILABLE = False

log = logging.getLogger(__name__)


class NnDualLoader(BaseLoader):
    """
    NnDirectLoader + an additional pretrained NN input stream loaded from
    config['pretrained_path'].

    Keeps ALL original keys (x_phy, c_phy, x_nn, c_nn, xc_nn_norm, target, temporal_features).
    Adds:
      - xc_pretrained_norm
    """

    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nc_tool = NetCDFDataset()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite

        # task vars
        self.nn_attributes = config["model"]["nn"].get("attributes", [])
        self.nn_forcings = config["model"]["nn"].get("forcings", [])
        self.target = config["train"]["target"]

        # pretrained vars
        nn_cfg = config["model"]["nn"]
        self.pretrained_ts_vars = nn_cfg.get("pretrained_time_series_vars", [])
        self.pretrained_static_vars = nn_cfg.get("pretrained_static_vars", [])

        # paths
        self.task_nc_path = config["data_path"]
        self.pretrained_nc_path = config["pretrained_path"]

        # norms
        self.log_norm_vars = config["model"].get("use_log_norm", []) or []
        out_base = config.get("out_path", "results")
        self.task_norm_path = os.path.join(out_base, "normalization_statistics.json")
        self.pre_norm_path = os.path.join(
            out_base, "normalization_statistics_pretrained.json"
        )

        self.device = config["device"]
        self.dtype = torch.float32

        # spatial testing config
        self.test = config.get("test", {})
        self.is_spatial_test = self.test and self.test.get("type") == "spatial"
        if holdout_index is not None:
            self.holdout_index = holdout_index
        elif self.is_spatial_test and "current_holdout_index" in self.test:
            self.holdout_index = self.test["current_holdout_index"]
        elif self.is_spatial_test and self.test.get("holdout_indexs"):
            self.holdout_index = self.test["holdout_indexs"][0]
        else:
            self.holdout_index = None

        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        self.pre_norm_stats = None

        # Optional per-station image loader (static images indexed by station row).
        image_zarr_path = config.get("image_zarr_path", None)
        if image_zarr_path:
            if not _ZARR_AVAILABLE:
                log.warning("[NnDualLoader] image_zarr_path set but ZarrStationImageLoader not available; skipping.")
                self.image_loader = None
            else:
                self.image_loader = ZarrStationImageLoader(
                    zarr_path=image_zarr_path,
                    image_array_name=config.get("image_array_name", "images"),
                    station_id_name=config.get("image_station_id_name", "station_id"),
                    mode=config.get("image_mode", "concat_inputs"),
                    scale_uint8=config.get("image_scale_uint8", True),
                )
                log.info(f"[NnDualLoader] Image loader ready: {image_zarr_path}")
        else:
            self.image_loader = None

        self.load_dataset()

    def load_dataset(self) -> None:
        train_range = {
            "start": self.config["train"]["start_time"],
            "end": self.config["train"]["end_time"],
        }
        test_range = {
            "start": self.config["test"]["start_time"],
            "end": self.config["test"]["end_time"],
        }

        if self.is_spatial_test:
            train_data = self._preprocess_data("train", train_range)
            test_data = self._preprocess_data("test", test_range)

            self.train_dataset, _ = split_dataset_by_basin(
                train_data, self.config, self.holdout_index
            )
            _, self.eval_dataset = split_dataset_by_basin(
                test_data, self.config, self.holdout_index
            )
        else:
            if self.test_split:
                self.train_dataset = self._preprocess_data("train", train_range)
                self.eval_dataset = self._preprocess_data("test", test_range)
            else:
                full_range = {
                    "start": self.config["train"]["start_time"],
                    "end": self.config["test"]["end_time"],
                }
                self.dataset = self._preprocess_data("all", full_range)

    def _preprocess_data(
        self, scope: str, t_range: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        # --- task data ---
        nn_data = load_nn_data(
            self._cfg_with_data_path(self.task_nc_path),
            scope,
            t_range,
            self.nn_forcings,
            self.nn_attributes,
            self.target,
            self.device,
            self.nc_tool,
        )

        target = nn_data["target"]

        # Filter fill/sentinel values to NaN
        target[target < -10] = np.nan

        x_nn = (
            nn_data["x_nn"].cpu().numpy()
            if torch.is_tensor(nn_data["x_nn"])
            else nn_data["x_nn"]
        )
        c_nn = (
            nn_data["c_nn"].cpu().numpy()
            if torch.is_tensor(nn_data["c_nn"])
            else nn_data["c_nn"]
        )

        # Build HBV physics forcings [prcp, tmean, pet] from raw nc variables.
        # The nc file has P, Tmax, Tmin, and a shortwave radiation series as
        # dynamic vars; tmean and pet are derived here rather than loaded
        # directly. The radiation variable's name differs across task
        # datasets (e.g. "SWd" for Caravan/global, "srad_daymet" for
        # CAMELS-531), so it's configurable via model.phy.raw_forcings.
        #
        # raw_forcing_vars is normally 4 elements [P, Tmax, Tmin, radiation],
        # with pet Hargreaves-Samani-derived from the last 3. If a 5th
        # element is given, it's used as a MEASURED pet directly instead of
        # deriving it (e.g. a dataset's own FAO Penman-Monteith estimate,
        # which is more accurate than the Hargreaves-Samani approximation
        # when a real pet variable is actually available) -- tmean is still
        # derived from Tmax/Tmin either way.
        phy_cfg = self.config["model"].get("phy") or {}
        phy_forcings = phy_cfg.get("forcings", [])
        raw_forcing_vars = phy_cfg.get("raw_forcings", ["P", "Tmax", "Tmin", "SWd"])
        if phy_forcings:
            try:
                raw_phy = load_nn_data(
                    self._cfg_with_data_path(self.task_nc_path),
                    scope,
                    t_range,
                    raw_forcing_vars,
                    [],
                    [],
                    self.device,
                    self.nc_tool,
                )
                raw = raw_phy["x_nn"]
                raw = raw.cpu().numpy() if torch.is_tensor(raw) else raw
                # raw: [T, N, 4 or 5], ordered per raw_forcing_vars ->
                # [P, Tmax, Tmin, radiation, (measured_pet)]
                prcp = raw[..., 0]
                tmax = raw[..., 1]
                tmin = raw[..., 2]
                tmean = (tmax + tmin) / 2.0
                if raw.shape[-1] >= 5:
                    pet = np.maximum(raw[..., 4], 0.0)
                else:
                    swd = raw[..., 3]
                    # Hargreaves-Samani PET using measured shortwave radiation:
                    # radiation (W/m²) -> mm/day via latent heat of vaporisation (λ ≈ 2.45 MJ/kg)
                    rs_mm = swd * 0.0864 * 0.408
                    td = np.maximum(tmax - tmin, 0.0)
                    pet = np.maximum(0.0135 * (tmean + 17.8) * np.sqrt(td) * rs_mm, 0.0)
                x_phy = np.stack([prcp, tmean, pet], axis=-1).astype(np.float32)
            except Exception as _e:
                log.warning(
                    f"[NnDualLoader] Could not build HBV physics forcings: {_e}. Using zeros."
                )
                x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
        else:
            x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
        c_phy = np.zeros((c_nn.shape[0], 0), dtype=np.float32)

        # temporal features
        start_date = pd.to_datetime(t_range["start"].replace("/", "-"))
        end_date = pd.to_datetime(t_range["end"].replace("/", "-"))
        warmup_days = self.config["model"]["warmup"]
        date_range = pd.date_range(
            start_date - pd.Timedelta(days=warmup_days), end_date, freq="D"
        )
        temporal_features = extract_temporal_features(date_range)

        # task normalization
        self.norm_stats = load_norm_stats(
            self.task_norm_path,
            self.overwrite,
            x_nn,
            c_nn,
            target,
            self.nn_forcings,
            self.nn_attributes,
            self.target,
            self.log_norm_vars,
            self.config,
        )
        xc_nn_norm = normalize_data(
            x_nn,
            c_nn,
            self.nn_forcings,
            self.nn_attributes,
            self.norm_stats,
            self.log_norm_vars,
        )

        # --- pretrained data (only when pretrained vars are configured) ---
        n_basins = x_nn.shape[0]
        if self.pretrained_ts_vars or self.pretrained_static_vars:
            pre_nn = load_nn_data(
                self._cfg_with_data_path(self.pretrained_nc_path),
                scope,
                t_range,
                self.pretrained_ts_vars,
                self.pretrained_static_vars,
                [],
                self.device,
                self.nc_tool,
            )
            x_pre = (
                pre_nn["x_nn"].cpu().numpy()
                if torch.is_tensor(pre_nn["x_nn"])
                else pre_nn["x_nn"]
            )
            c_pre = (
                pre_nn["c_nn"].cpu().numpy()
                if torch.is_tensor(pre_nn["c_nn"])
                else pre_nn["c_nn"]
            )

            # pretrained normalization (separate stats file, no log norms)
            # Strip area_name so get_basin_area doesn't search pretrained_static_vars
            # for a CAMELS attribute it will never find.
            cfg_pre = dict(self.config)
            if "observations" in cfg_pre:
                obs = dict(cfg_pre["observations"])
                obs.pop("area_name", None)
                cfg_pre["observations"] = obs
            dummy_target = np.zeros((x_pre.shape[0], x_pre.shape[1], 1), dtype=np.float32)
            self.pre_norm_stats = load_norm_stats(
                self.pre_norm_path,
                self.overwrite,
                x_pre,
                c_pre,
                dummy_target,
                self.pretrained_ts_vars,
                self.pretrained_static_vars,
                ["_dummy_"],
                [],
                cfg_pre,
            )
            xc_pretrained_norm = normalize_data(
                x_pre,
                c_pre,
                self.pretrained_ts_vars,
                self.pretrained_static_vars,
                self.pre_norm_stats,
                [],
            )

            # Align pretrained data to the task's station ordering AND calendar.
            # The two NC files may contain different station subsets and different
            # time coverage (e.g. the pretrained file may start later or end earlier).
            # Without aligning both axes by real station id / calendar date,
            # xc_pretrained_norm silently ends up with a different basin axis than
            # x_nn/target, or a time axis that's off by a few days relative to it
            # (same index != same date), which either corrupts which pretrained
            # timestep gets paired with which task timestep, or crashes the sampler
            # when it slices past the shorter array's end.
            #
            # station_ids/date_range come straight back from load_nn_data, already
            # filtered by the same basin-subset step applied to x_nn/x_pre, so they
            # match those arrays' row order/length exactly (unlike re-opening the raw
            # NC files here, which would be un-subsetted and a different length).
            try:
                task_ids = nn_data.get("station_ids")
                pre_ids = pre_nn.get("station_ids")
                task_dates = nn_data.get("date_range")
                pre_dates = pre_nn.get("date_range")
                if (
                    task_ids is None
                    or pre_ids is None
                    or task_dates is None
                    or pre_dates is None
                ):
                    raise ValueError(
                        "station_ids/date_range not available from load_nn_data"
                    )
                task_dates = pd.DatetimeIndex(task_dates)
                pre_dates = pd.DatetimeIndex(pre_dates)

                station_match = len(task_ids) == xc_pretrained_norm.shape[1] and np.array_equal(
                    task_ids, pre_ids
                )
                time_match = len(task_dates) == xc_pretrained_norm.shape[0] and pre_dates.equals(
                    task_dates
                )

                if not (station_match and time_match):
                    old_shape = xc_pretrained_norm.shape

                    pre_id_to_idx = {str(sid): i for i, sid in enumerate(pre_ids)}
                    station_dst, station_src = [], []
                    for task_i, sid in enumerate(task_ids):
                        pre_i = pre_id_to_idx.get(str(sid))
                        if pre_i is not None:
                            station_dst.append(task_i)
                            station_src.append(pre_i)

                    pre_date_to_idx = {d: i for i, d in enumerate(pre_dates)}
                    time_dst, time_src = [], []
                    for task_t, d in enumerate(task_dates):
                        pre_t = pre_date_to_idx.get(d)
                        if pre_t is not None:
                            time_dst.append(task_t)
                            time_src.append(pre_t)

                    n_f = old_shape[2]
                    aligned = np.zeros((len(task_dates), len(task_ids), n_f), dtype=np.float32)
                    if station_src and time_src:
                        aligned[np.ix_(time_dst, station_dst)] = xc_pretrained_norm[
                            np.ix_(time_src, station_src)
                        ]
                    xc_pretrained_norm = aligned
                    log.warning(
                        f"[NnDualLoader] Aligned xc_pretrained_norm to task grid: "
                        f"{len(station_src)}/{len(task_ids)} stations matched, "
                        f"{len(time_src)}/{len(task_dates)} timesteps matched "
                        f"(was {old_shape}); unmatched positions filled with zeros."
                    )
            except Exception as _e:
                log.warning(
                    f"[NnDualLoader] Could not align pretrained data to task grid: {_e}. "
                    "xc_pretrained_norm may not match task station/time axes."
                )
        else:
            # No pretrained vars configured (e.g. CudnnLstmModel baseline)
            n_t = x_nn.shape[1]
            xc_pretrained_norm = np.zeros((n_t, n_basins, 0), dtype=np.float32)

        # Store large tensors on CPU to avoid GPU OOM with large datasets.
        return {
            "x_phy": to_tensor(x_phy, "cpu", self.dtype),
            "c_phy": to_tensor(c_phy, "cpu", self.dtype),
            "x_nn": to_tensor(x_nn, "cpu", self.dtype),
            "c_nn": to_tensor(c_nn, "cpu", self.dtype),
            "xc_nn_norm": to_tensor(xc_nn_norm, "cpu", self.dtype),
            "temporal_features": to_tensor(temporal_features, "cpu", self.dtype),
            "target": to_tensor(target, "cpu", self.dtype),
            "xc_pretrained_norm": to_tensor(xc_pretrained_norm, "cpu", self.dtype),
            "image_loader": self.image_loader,
        }

    def _cfg_with_data_path(self, data_path: str) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg["data_path"] = data_path
        return cfg
