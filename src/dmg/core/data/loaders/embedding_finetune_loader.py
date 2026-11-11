import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

from dmg.core.data.data import extract_temporal_features, split_dataset_by_basin
from dmg.core.data.loader_utils import (
    load_nn_data,
    load_norm_stats,
    normalize_data,
    to_tensor,
)
from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.load_nc import NetCDFDataset

log = logging.getLogger(__name__)


class EmbeddingFinetuneLoader(BaseLoader):
    """
    Task data (same as NnDualLoader) plus a PRECOMPUTED embedding stream read
    straight from a NetCDF file (config['embedding_path']).

    Unlike NnDualLoader, this loader does not load raw pretrained-format
    time-series/static variables for on-the-fly encoding through a frozen
    foundation model. It loads embeddings that a foundation model already
    produced and that were saved to disk -- the format such a model's
    embeddings are expected to ship in -- so no encoder forward pass is
    needed at train/eval time.

    Expected embedding file layout: a variable (default name "embedding",
    configurable via model.nn.embedding_var_name) with dims
    (station_ids, time, embed_dim) for a per-timestep embedding, or
    (station_ids, embed_dim) for a single static embedding per basin
    (broadcast across all timesteps). The embedding width (embed_dim) is
    read from the file at load time -- 256, 1024, whatever the release
    actually contains -- and is never hardcoded here or in
    EmbeddingFinetuneing.

    Keeps ALL NnDirectLoader keys (x_phy, c_phy, x_nn, c_nn, xc_nn_norm,
    target, temporal_features). Adds:
      - xc_pretrained_norm: [time, basins, embed_dim] precomputed embeddings.
        Reuses NnDualLoader's key name so the existing samplers (HydroSampler,
        ModHydroSampler), which already know how to slice/pass through
        'xc_pretrained_norm', work unchanged.
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

        nn_cfg = config["model"]["nn"]
        self.nn_attributes = nn_cfg.get("attributes", [])
        self.nn_forcings = nn_cfg.get("forcings", [])
        self.target = config["train"]["target"]

        self.embedding_var_name = nn_cfg.get("embedding_var_name", "embedding")

        # paths
        self.task_nc_path = config["data_path"]
        self.embedding_nc_path = config["embedding_path"]

        # norms (task side only -- embeddings are used as released, unnormalized)
        self.log_norm_vars = config["model"].get("use_log_norm", []) or []
        out_base = config.get("out_path", "results")
        self.task_norm_path = os.path.join(out_base, "normalization_statistics.json")

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
        self.embedding_dim = None

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

        # Build HBV physics forcings [prcp, tmean, pet] from raw nc variables,
        # same derivation NnDualLoader uses, kept here so this loader remains
        # a drop-in swap for delta-model (physics-parameterization) configs
        # that always expect x_phy/c_phy in the dataset dict.
        #
        # raw_forcing_vars is normally 4 elements [P, Tmax, Tmin, radiation],
        # with pet Hargreaves-Samani-derived from the last 3. If a 5th
        # element is given, it's used as a MEASURED pet directly instead of
        # deriving it (e.g. a dataset's own FAO Penman-Monteith estimate,
        # more accurate than the Hargreaves-Samani approximation when a real
        # pet variable is actually available) -- tmean is still derived from
        # Tmax/Tmin either way.
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
                    f"[EmbeddingFinetuneLoader] Could not build HBV physics forcings: {_e}. Using zeros."
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

        # --- precomputed embedding stream ---
        # x_nn is time-major [T, N, F] (see load_nn_data); c_nn is basin-first
        # [N, S], so it -- not x_nn.shape[0] -- gives the basin count.
        xc_embedding = self._load_embedding_stream(nn_data, c_nn.shape[0])

        # Store large tensors on CPU to avoid GPU OOM with large datasets.
        return {
            "x_phy": to_tensor(x_phy, "cpu", self.dtype),
            "c_phy": to_tensor(c_phy, "cpu", self.dtype),
            "x_nn": to_tensor(x_nn, "cpu", self.dtype),
            "c_nn": to_tensor(c_nn, "cpu", self.dtype),
            "xc_nn_norm": to_tensor(xc_nn_norm, "cpu", self.dtype),
            "temporal_features": to_tensor(temporal_features, "cpu", self.dtype),
            "target": to_tensor(target, "cpu", self.dtype),
            "xc_pretrained_norm": to_tensor(xc_embedding, "cpu", self.dtype),
        }

    def _load_embedding_stream(
        self, nn_data: Dict[str, Any], n_basins: int
    ) -> np.ndarray:
        """Load precomputed embeddings and align them to the task's station
        ordering AND calendar, exactly as NnDualLoader aligns its pretrained
        stream: the embedding file may cover a different station subset or
        time span than the task data, so both axes are matched by real
        station id / calendar date rather than raw index.
        """
        task_ids = nn_data.get("station_ids")
        task_dates = pd.DatetimeIndex(nn_data.get("date_range"))
        n_time_task = len(task_dates)

        ds = xr.open_dataset(self.embedding_nc_path)
        try:
            if self.embedding_var_name not in ds:
                raise ValueError(
                    f"Embedding variable '{self.embedding_var_name}' not found in "
                    f"{self.embedding_nc_path}. Available variables: "
                    f"{list(ds.data_vars)}"
                )

            da = ds[self.embedding_var_name]
            has_time = "time" in da.dims

            if has_time:
                da = da.transpose("station_ids", "time", ...)
                emb_dates = pd.DatetimeIndex(ds["time"].values)
            else:
                da = da.transpose("station_ids", ...)
                emb_dates = None

            emb_array = np.asarray(da.values, dtype=np.float32)
            emb_ids = (
                ds["station_ids"].values
                if "station_ids" in ds
                else np.arange(emb_array.shape[0])
            )
        finally:
            ds.close()

        embedding_dim = emb_array.shape[-1]
        self.embedding_dim = embedding_dim

        id_to_idx = {str(sid): i for i, sid in enumerate(emb_ids)}
        station_dst, station_src = [], []
        for task_i, sid in enumerate(task_ids):
            emb_i = id_to_idx.get(str(sid))
            if emb_i is not None:
                station_dst.append(task_i)
                station_src.append(emb_i)

        if has_time:
            # As-of (forward-fill) match: for each task day, use the embedding
            # at the latest available emb_date <= that day. For a daily
            # embedding file this reduces to an exact match; for a coarser
            # file (monthly/seasonal/annual period-start timestamps, as
            # produced by scripts/generate_embeddings.py) this holds each
            # period's embedding constant across every day inside it, which
            # is exactly how those space-saving aggregates are meant to be
            # consumed -- no separate code path needed per resolution.
            sort_order = np.argsort(emb_dates.values)
            emb_dates_sorted = emb_dates.values[sort_order]

            pos = np.searchsorted(emb_dates_sorted, task_dates.values, side='right') - 1
            valid_time = pos >= 0
            time_dst = np.nonzero(valid_time)[0]
            time_src = sort_order[pos[valid_time]]

            aligned = np.zeros(
                (n_time_task, n_basins, embedding_dim), dtype=np.float32
            )
            if station_src and len(time_src):
                # emb_array is [n_emb_station, n_emb_time, D] -> [time, station, D]
                emb_tsd = np.transpose(emb_array, (1, 0, 2))
                aligned[np.ix_(time_dst, station_dst)] = emb_tsd[
                    np.ix_(time_src, station_src)
                ]

            if len(station_src) < len(task_ids) or len(time_src) < n_time_task:
                log.warning(
                    f"[EmbeddingFinetuneLoader] Aligned embeddings to task grid: "
                    f"{len(station_src)}/{len(task_ids)} stations matched, "
                    f"{len(time_src)}/{n_time_task} timesteps matched via as-of "
                    "lookup; task days before the embedding file's earliest "
                    "date (or unmatched stations) are filled with zeros."
                )
        else:
            static_aligned = np.zeros((n_basins, embedding_dim), dtype=np.float32)
            if station_src:
                static_aligned[station_dst] = emb_array[station_src]
                if len(station_src) < len(task_ids):
                    log.warning(
                        f"[EmbeddingFinetuneLoader] Aligned static embeddings to "
                        f"task grid: {len(station_src)}/{len(task_ids)} stations "
                        "matched; unmatched basins filled with zeros."
                    )
            # Broadcast the per-basin embedding across every task timestep.
            aligned = np.repeat(static_aligned[np.newaxis, :, :], n_time_task, axis=0)

        return aligned

    def _cfg_with_data_path(self, data_path: str) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg["data_path"] = data_path
        return cfg
