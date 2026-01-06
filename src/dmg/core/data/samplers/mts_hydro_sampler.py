# from dmg.core.data.loaders.mts_hydro_loader import DistributedDataSchema

# import torch
# from torch.utils.data import Sampler
# from typing import Optional
# from pydantic import BaseModel, ConfigDict
# from torchtyping import TensorType

# from dmg.core.data.samplers.base import BaseSampler


# class MtsHydroSampler(BaseSampler):
#     """Multi-timescale hydrological data sampler for distributed datasets."""

#     def __init__(
#         self,
#         config: dict,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.time_window_loader: Optional[TimeWindowLoader] = None

#     def get_samples(self, dataset: DistributedDataSchema, mode: str = 'train'):
#         """
#         :param mode: train, valid, test, simulation.
#         :return:
#         """
#         config = self.config
#         window_size = config['delta_model']['phy_model']['high_freq_model'][
#             'window_size'
#         ]
#         stride = config['train']['stride']
#         warmup_daily = config['delta_model']['phy_model']['low_freq_model'][
#             'window_size'
#         ]
#         warmup_hourly = config['delta_model']['phy_model']['high_freq_model'][
#             'train_warmup'
#         ]
#         agg_timescale_dim = tuple(
#             config['delta_model']['phy_model']['high_freq_model']['agg_timescale_dim']
#         )
#         train_batch_size = config['train']['batch_size']
#         valid_batch_size = config['valid']['batch_size']
#         test_batch_size = config['test']['batch_size']
#         if mode == 'train':
#             batch_size = train_batch_size
#             shuffle = True
#             is_seamless = False
#             k = window_size
#         elif mode == 'valid':
#             batch_size = valid_batch_size
#             shuffle = True
#             is_seamless = False
#             k = window_size
#         elif mode == 'test':
#             batch_size = test_batch_size
#             shuffle = False
#             is_seamless = False
#             k = window_size
#         elif mode == 'simulation':
#             batch_size = 999999  # all at once
#             shuffle = False
#             is_seamless = True
#             k = len(dataset.time) - warmup_daily * 24
#         else:
#             raise ValueError(f"Unknown mode: {mode} for MtsHydroSampler.get_samples")

#         self.time_window_loader = TimeWindowLoader(
#             data=dataset,
#             k=k,
#             stride=stride,
#             warmup_daily=warmup_daily,
#             warmup_hourly=warmup_hourly,
#             agg_timescale_dim=agg_timescale_dim,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             is_seamless=is_seamless,
#         )
#         for batch in self.time_window_loader:
#             data_dict = {
#                 'target': batch.target,
#                 'xc_nn_norm_high_freq': batch.xc_nn_norm_hourly,
#                 'x_phy_high_freq': batch.x_phy_hourly,
#                 'xc_nn_norm_low_freq': batch.xc_nn_norm_daily,
#                 'x_phy_low_freq': batch.x_phy_daily,
#                 'c_nn_norm': batch.c_nn_norm,
#                 'rc_nn_norm': batch.rout_c_nn,
#                 'ac_all': batch.ac_all,
#                 'elev_all': batch.elev_all,
#                 'areas': batch.areas,
#                 'outlet_topo': batch.outlet_topo,
#                 'batch_sample': batch.gauge_index,
#                 'time': batch.time,
#             }
#             yield data_dict

#     def load_data(self):
#         """Custom implementation for loading data."""
#         print("Loading data...")

#     def preprocess_data(self):
#         """Custom implementation for preprocessing data."""
#         print("Preprocessing data...")

#     def cleanup_memory(self):
#         """Custom implementation for cleaning up memory."""
#         self.time_window_loader = None


# class DistributedSampleSchema(BaseModel):
#     """Schema for a batch of distributed samples."""

#     unit_index: TensorType["n_units"]
#     gauge_index: TensorType["n_gages"]
#     time: int

#     model_config = ConfigDict(arbitrary_types_allowed=True)


# class DistributedPhySampleSchema(DistributedSampleSchema):
#     """Schema for a batch of distributed physical samples."""

#     target: TensorType["n_gages", "window_size", "1"]
#     x_phy_daily: TensorType["daily_size", "n_units", "d"]
#     xc_nn_norm_daily: TensorType["daily_size", "n_units", "ds"]
#     x_phy_hourly: TensorType["window_size", "n_units", "d"]
#     xc_nn_norm_hourly: TensorType["window_size", "n_units", "ds"]
#     c_nn_norm: TensorType["n_units", "s"]
#     rout_c_nn: TensorType["n_pairs", "rs"]  # where topo == 1

#     ac_all: TensorType["n_units"]
#     elev_all: TensorType["n_units"]
#     areas: TensorType["n_units"]
#     outlet_topo: TensorType["n_gages", "n_units"]

#     unit_index: TensorType["n_units"]
#     gauge_index: TensorType["n_gages"]  # idx for gage-wise normalized loss
#     time: int

#     model_config = ConfigDict(arbitrary_types_allowed=True)


# def get_gauge_subset(
#     dataset: DistributedDataSchema, gauges: list[str]
# ) -> DistributedDataSchema:
#     """Get a subset of the dataset for specified gauges and their upstream units."""
#     gauge_indexes = [dataset.gauge.index(gauge) for gauge in gauges]
#     unit_indexes = torch.nonzero(dataset.topo[gauge_indexes, :].sum(axis=0)).squeeze(-1)
#     subset = DistributedDataSchema(
#         target=dataset.target[gauge_indexes, :],
#         dyn_input=dataset.dyn_input[unit_indexes, :, :],
#         static_input=dataset.static_input[unit_indexes, :],
#         rout_static_input=dataset.rout_static_input[gauge_indexes, :, :][
#             :, unit_indexes, :
#         ],
#         ac_all=dataset.ac_all[unit_indexes],
#         elev_all=dataset.elev_all[unit_indexes],
#         time=dataset.time,
#         topo=dataset.topo[gauge_indexes, :][:, unit_indexes],
#         areas=dataset.areas[unit_indexes],
#         gauge=[dataset.gauge[i] for i in gauge_indexes],
#         gauge_index=torch.tensor(gauge_indexes),
#         unit=[dataset.unit[i] for i in unit_indexes],
#     )
#     if dataset.scaled_target is not None:
#         subset.scaled_target = dataset.scaled_target[gauge_indexes, :]
#         subset.scaled_dyn_input = dataset.scaled_dyn_input[unit_indexes, :, :]
#         subset.scaled_static_input = dataset.scaled_static_input[unit_indexes, :]
#         subset.scaled_rout_static_input = dataset.scaled_rout_static_input[
#             gauge_indexes, :, :
#         ][:, unit_indexes, :]
#     return subset


# def get_time_subset(
#     dataset: DistributedDataSchema,
#     t0: int,
#     t1: int,
#     warmup_daily: int = 365,
#     warmup_hourly: int = 24 * 7,
# ) -> DistributedDataSchema:
#     """Get a subset of the dataset within a specified time range, including warmup periods."""
#     t0_all = t0 - warmup_hourly * 3600 - warmup_daily * 24 * 3600
#     time_mask = (dataset.time >= t0_all) & (dataset.time <= t1)
#     subset = DistributedDataSchema(
#         target=dataset.target[:, time_mask],
#         dyn_input=dataset.dyn_input[:, time_mask, :],
#         static_input=dataset.static_input,
#         rout_static_input=dataset.rout_static_input,
#         ac_all=dataset.ac_all,
#         elev_all=dataset.elev_all,
#         time=dataset.time[time_mask],
#         topo=dataset.topo,
#         areas=dataset.areas,
#         gauge=dataset.gauge,
#         gauge_index=dataset.gauge_index,
#         unit=dataset.unit,
#     )
#     if dataset.scaled_target is not None:
#         subset.scaled_target = dataset.scaled_target[:, time_mask]
#         subset.scaled_dyn_input = dataset.scaled_dyn_input[:, time_mask, :]
#         subset.scaled_static_input = dataset.scaled_static_input
#     return subset


# class TimeWindowBatchSampler(Sampler):
#     """Batch sampler that samples time windows with warmup periods."""

#     def __init__(
#         self,
#         batch_size: int,
#         Y: torch.Tensor,
#         stride: int,
#         k: int,
#         warmup_daily: int = 365,
#         warmup_hourly: int = 24 * 7,
#         is_seamless: bool = False,
#         shuffle=True,
#     ):
#         super().__init__(None)
#         self.windows_by_t = self._build_windows_by_t(
#             Y,
#             stride=stride,
#             k=k,
#             warmup_daily=warmup_daily,
#             warmup_hourly=warmup_hourly,
#             is_seamless=is_seamless,
#         )
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     # def __iter__(self):
#     #     time_keys = list(self.windows_by_t.keys())
#     #     if self.shuffle:
#     #         time_keys = [time_keys[i] for i in torch.randperm(len(time_keys)).tolist()]
#     #
#     #     for t in time_keys:
#     #         spatial_idxs = self.windows_by_t[t]
#     #         if self.shuffle:
#     #             spatial_idxs = [spatial_idxs[i] for i in torch.randperm(len(spatial_idxs)).tolist()]
#     #
#     #         for i in range(0, len(spatial_idxs), self.batch_size):
#     #             yield (t, [sid for sid in spatial_idxs[i:i + self.batch_size]])

#     def __iter__(self):
#         """Globally random at the batch level (so each yield is a random t paired with a random slice of spatial_idxs)."""
#         items = []
#         time_keys = list(self.windows_by_t.keys())
#         if self.shuffle:
#             time_keys = [time_keys[i] for i in torch.randperm(len(time_keys)).tolist()]

#         for t in time_keys:
#             spatial_idxs = self.windows_by_t[t]
#             if self.shuffle:
#                 spatial_idxs = [
#                     spatial_idxs[i] for i in torch.randperm(len(spatial_idxs)).tolist()
#                 ]
#             for i in range(0, len(spatial_idxs), self.batch_size):
#                 items.append((t, spatial_idxs[i : i + self.batch_size]))

#         if self.shuffle and len(items) > 1:
#             perm = torch.randperm(len(items)).tolist()
#             items = [items[i] for i in perm]

#         for t, batch in items:
#             yield (t, batch)

#     def __len__(self):
#         return sum(
#             (len(v) + self.batch_size - 1) // self.batch_size
#             for v in self.windows_by_t.values()
#         )

#     @staticmethod
#     def _build_windows_by_t(
#         Y: torch.Tensor,
#         stride: int,
#         k: int,
#         warmup_daily: int = 365,
#         warmup_hourly: int = 24 * 7,
#         is_seamless: bool = False,
#     ) -> dict[int, list[int]]:
#         n_s, n_t = Y.shape
#         mask = ~torch.isnan(Y)  # True where target exists

#         warmup_daily_len = 24 * warmup_daily
#         windows_by_t = {}
#         t_first = warmup_daily_len
#         t_last_inclusive = n_t - k

#         for t in range(t_first, t_last_inclusive + 1, stride):
#             if is_seamless:
#                 idxs = list(range(n_s))
#             else:
#                 idxs = [
#                     i for i in range(n_s) if mask[i, t + warmup_hourly : t + k].any()
#                 ]

#             if idxs:
#                 windows_by_t[t + warmup_hourly] = idxs

#         return windows_by_t


# class TimeWindowLoader:
#     """Data loader that samples time windows with warmup periods."""

#     def __init__(
#         self,
#         data: DistributedDataSchema,
#         k: int,
#         stride: int,
#         warmup_daily: int,
#         warmup_hourly: int,
#         agg_timescale_dim: tuple,
#         batch_size: int,
#         shuffle: bool = False,
#         is_seamless: bool = False,
#     ):
#         """
#         :param data: dataset.
#         :param k: hourly window size.
#         :param warmup: daily warmup size.
#         :param stride: sample time stride.
#         :param agg_timescale_dim: dims of dyn_input to aggregate (sum) when converting from hourly to daily.
#         :param is_seamless: inference mode, sample all stations at each time step.
#         """
#         self.data = data
#         self.k = int(k)
#         self.warmup_daily = int(warmup_daily)
#         self.warmup_hourly = int(warmup_hourly)
#         self.agg_timescale_dim = agg_timescale_dim
#         self.sampler = TimeWindowBatchSampler(
#             batch_size=batch_size,
#             Y=data.target,
#             stride=stride,
#             k=k,
#             warmup_daily=warmup_daily,
#             warmup_hourly=warmup_hourly,
#             is_seamless=is_seamless,
#             shuffle=shuffle,
#         )

#     def _build_batch(self, t: int, sids: list[int]) -> DistributedPhySampleSchema:
#         t = t - self.warmup_hourly  # hourly window start
#         t0 = t - self.warmup_daily * 24  # daily warmup start
#         t1 = t + self.k  # exclusive, hourly window end
#         data = self.data
#         agg_timescale_dim = self.agg_timescale_dim

#         local_gauge_indexes = torch.tensor(sorted(sids))
#         global_gauge_indexes = torch.tensor(
#             [data.gauge_index[i] for i in local_gauge_indexes]
#         )
#         unit_indexes = torch.nonzero(
#             data.topo[local_gauge_indexes, :].sum(axis=0)
#         ).squeeze(-1)
#         target = data.target[local_gauge_indexes, t:t1]
#         c_nn_norm = data.scaled_static_input[unit_indexes, :]
#         x_phy_daily = data.dyn_input[unit_indexes, t0:t, :]
#         x_phy_daily = (
#             x_phy_daily.contiguous()
#             .view(x_phy_daily.shape[0], -1, 24, x_phy_daily.shape[-1])
#             .mean(dim=2)
#         )
#         x_phy_daily[:, :, agg_timescale_dim] = x_phy_daily[:, :, agg_timescale_dim] * 24
#         x_phy_daily = x_phy_daily.swapaxes(0, 1).contiguous()
#         xc_nn_norm_daily = data.scaled_dyn_input[unit_indexes, t0:t, :]
#         xc_nn_norm_daily = (
#             xc_nn_norm_daily.contiguous()
#             .view(xc_nn_norm_daily.shape[0], -1, 24, xc_nn_norm_daily.shape[-1])
#             .mean(dim=2)
#         )
#         xc_nn_norm_daily = xc_nn_norm_daily.swapaxes(0, 1).contiguous()
#         xc_nn_norm_daily = torch.concat(
#             [
#                 xc_nn_norm_daily,
#                 c_nn_norm.unsqueeze(0).repeat(xc_nn_norm_daily.shape[0], 1, 1),
#             ],
#             dim=2,
#         )
#         x_phy_hourly = data.dyn_input[unit_indexes, t:t1, :]
#         x_phy_hourly = x_phy_hourly.swapaxes(0, 1).contiguous()
#         xc_nn_norm_hourly = data.scaled_dyn_input[unit_indexes, t:t1, :]
#         xc_nn_norm_hourly = xc_nn_norm_hourly.swapaxes(0, 1).contiguous()
#         xc_nn_norm_hourly = torch.concat(
#             [
#                 xc_nn_norm_hourly,
#                 c_nn_norm.unsqueeze(0).repeat(xc_nn_norm_hourly.shape[0], 1, 1),
#             ],
#             dim=2,
#         )

#         ac_all = data.ac_all[unit_indexes]
#         elev_all = data.elev_all[unit_indexes]
#         areas = data.areas[unit_indexes]
#         outlet_topo = data.topo[local_gauge_indexes, :][:, unit_indexes]

#         rout_c_nn = data.scaled_rout_static_input[local_gauge_indexes, :, :][
#             :, unit_indexes, :
#         ]
#         rout_c_nn = rout_c_nn.reshape(-1, rout_c_nn.shape[-1])[
#             outlet_topo.flatten().bool()
#         ]

#         batch = DistributedPhySampleSchema(
#             target=target.unsqueeze(-1),
#             x_phy_daily=x_phy_daily,
#             xc_nn_norm_daily=xc_nn_norm_daily,
#             x_phy_hourly=x_phy_hourly,
#             xc_nn_norm_hourly=xc_nn_norm_hourly,
#             c_nn_norm=c_nn_norm,
#             rout_c_nn=rout_c_nn,
#             ac_all=ac_all,
#             elev_all=elev_all,
#             areas=areas,
#             outlet_topo=outlet_topo,
#             unit_index=unit_indexes,
#             gauge_index=global_gauge_indexes,
#             time=data.time[t],  # start before warmup
#         )
#         return batch

#     def __iter__(self):
#         """Get batch indexes from sampler, then fetch data from data."""
#         for batch_indexes in self.sampler:
#             yield self._build_batch(*batch_indexes)
