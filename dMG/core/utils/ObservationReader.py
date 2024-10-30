# TODO: Adapt to PMI; from dMCdev @Tadd Bindas.
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from conf.config import Config
# from dMC.dataset_modules.utils import pad_gage_id
from core.utils.Dates import Dates
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

dask.config.set({"dataframe.query-planning": True})  # noqa: [E402]
import dask.dataframe as dd

log = logging.getLogger(__name__)



class ObservationReader(BaseModel, ABC):
    """
    Define an interface for reading data from different sources.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    dates: Dates
    files_path: Path
    gage_dict: Dict[str, List[Any]]

    def __init__(self, **kwargs):
        super(ObservationReader, self).__init__(
            dates=kwargs["dates"],
            files_path=kwargs["cfg"].observations.observations_path,
            gage_dict=kwargs["gage_dict"],
        )

    @abstractmethod
    def read_data(self) -> xr.Dataset:
        raise NotImplementedError("Missing read_data function")

    @abstractmethod
    def read_gage(
        self, gage_id: int, time_range: Union[pd.DatetimeIndex, np.ndarray]
    ) -> np.ndarray:
        raise NotImplementedError("Missing read_gage function")

    @staticmethod
    def convert_ft3_s_to_m3_s(flow_rates_ft3_s: np.ndarray) -> np.ndarray:
        """
        Convert a 2D tensor of flow rates
        from cubic feet per second (ft³/s) to cubic meters per second (m³/s).
        """
        conversion_factor = 0.0283168
        return flow_rates_ft3_s * conversion_factor


class GlobalReader(ObservationReader):
    """
    Reads the data from the following class:
    @article{beck2020global,
      title={Global fully distributed parameter regionalization based on observed streamflow from 4,229 headwater catchments},
      author={Beck, Hylke E and Pan, Ming and Lin, Peirong and Seibert, Jan and van Dijk, Albert IJM and Wood, Eric F},
      journal={Journal of Geophysical Research: Atmospheres},
      volume={125},
      number={17},
      pages={e2019JD031485},
      year={2020},
      publisher={Wiley Online Library}
    }
    """

    def __init__(self, **kwargs):
        super(GlobalReader, self).__init__(**kwargs)

    def read_data(self) -> xr.Dataset:
        raise NotImplementedError("This function is still a work in progress")

    def read_gage(self, gage_id: int, time_range: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError("This function is still a work in progress")


class GRDCReader(ObservationReader):
    def __init__(self, **kwargs):
        super(GRDCReader, self).__init__(**kwargs)

    def read_data(self) -> xr.Dataset:
        gage_idx = self.gage_dict["STAID"]
        y = np.zeros([len(gage_idx), len(self.dates.daily_time_range)])
        for idx, grdc_id in enumerate(
            tqdm(
                gage_idx,
                desc="\rReading GRDC observations",
                ncols=140,
                ascii=True,
            )
        ):
            data_obs = self.read_gage(grdc_id, self.dates.daily_time_range)
            y[idx, :] = data_obs
        ds = xr.Dataset(
            {"streamflow": (["gage_id", "time"], y)},
            coords={"gage_id": gage_idx, "time": self.dates.daily_time_range},
        )
        return ds

    def read_gage(self, gage_id, time_range) -> np.ndarray:
        file_path = Path(self.files_path) / f"{gage_id}.nc"
        ds = xr.open_dataset(file_path)
        filtered_ds = ds.sel(time=time_range)
        return filtered_ds.runoff_mean.values.squeeze()


class SusquehannaReader(ObservationReader):
    def __init__(self, **kwargs):
        super(SusquehannaReader, self).__init__(**kwargs)

    def read_data(self) -> xr.Dataset:
        padded_gage_idx = [
            pad_gage_id(gage_idx) for gage_idx in self.gage_dict["STAID"]
        ]
        y = np.zeros([len(padded_gage_idx), len(self.dates.daily_time_range)])
        for idx, srb_id in enumerate(
            tqdm(
                padded_gage_idx,
                desc="\rReading Susquehanna observations",
                ncols=140,
                ascii=True,
            )
        ):
            data_obs = self.read_gage(srb_id, self.dates.daily_time_range)
            y[idx, :] = data_obs
        ds = xr.Dataset(
            {"streamflow": (["gage_id", "time"], y)},
            coords={
                "gage_id": padded_gage_idx,
                "time": self.dates.daily_time_range,
            },
        )
        return ds

    def read_gage(self, gage_id, time_range) -> np.ndarray:
        file_path = Path(self.files_path)
        pattern = f"*-{gage_id}.csv"
        files = list(file_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files found for gage_id: {gage_id}.")
        ddf = dd.read_csv(files[0])
        ddf["dates"] = dd.to_datetime(ddf["dates"], format="%Y%m%d")
        mask = ddf["dates"].map_partitions(lambda part: part.isin(time_range))
        filtered_ddf = ddf[mask]
        filtered_df = filtered_ddf.compute()
        return filtered_df["v"].values


class ZarrUSGSReader(ObservationReader):
    def __init__(self, **kwargs):
        super(ZarrUSGSReader, self).__init__(**kwargs)

    def read_data(self) -> xr.Dataset:
        padded_gage_idx = [
            pad_gage_id(gage_idx) for gage_idx in self.gage_dict["STAID"]
        ]
        y = np.zeros([len(padded_gage_idx), len(self.dates.daily_time_range)])
        root = zarr.open_group(Path(self.files_path), mode="r")
        for idx, gage_id in enumerate(
            tqdm(
                padded_gage_idx,
                desc="\rReading Zarr USGS observations",
                ncols=140,
                ascii=True,
            )
        ):
            try:
                data_obs = root[gage_id]
                y[idx, :] = data_obs[self.dates.numerical_time_range]
            except KeyError as e:
                log.error(f"Cannot find zarr store: {e}")
        _observations = self.convert_ft3_s_to_m3_s(y)
        ds = xr.Dataset(
            {"streamflow": (["gage_id", "time"], _observations)},
            coords={
                "gage_id": padded_gage_idx,
                "time": self.dates.daily_time_range,
            },
        )
        return ds

    def read_gage(self, gage_id, time_range) -> np.ndarray:
        pass


# class USGSReader(ObservationReader):
#     def __init__(self, **kwargs):
#         super(USGSReader, self).__init__(**kwargs)
#
#     def read_data(self) -> xr.Dataset:
#         padded_gage_idx = [
#             pad_gage_id(gage_idx) for gage_idx in self.gage_dict["STAID"]
#         ]
#         y = np.zeros([len(padded_gage_idx), len(self.dates.daily_time_range)])
#         for idx, usgs_id in enumerate(
#             tqdm(
#                 padded_gage_idx,
#                 desc="Reading USGS observations",
#                 ncols=140,
#                 ascii=True,
#             )
#         ):
#             data_obs = self.read_gage(usgs_id, self.dates.daily_time_range)
#             y[idx, :] = data_obs
#         _observations = self.convert_ft3_s_to_m3_s(y)
#         ds = xr.Dataset(
#             {"streamflow": (["gage_id", "time"], _observations)},
#             coords={
#                 "gage_id": padded_gage_idx,
#                 "time": self.dates.daily_time_range,
#             },
#         )
#         return ds
#
#     def read_gage(self, gage_id, time_range) -> np.ndarray:
#         usgs_short = str(int(gage_id))
#         idx = (
#             np.where(self.gage_dict["STAID"] == usgs_short)[0][0]
#             if np.any(self.gage_dict["STAID"] == usgs_short)
#             else None
#         )
#         huc = str(self.gage_dict["HUC02"][idx])
#
#         file_path = (
#             Path(self.files_path) / huc.zfill(2) / f"{gage_id}.txt"
#         )
#         df_flow = pd.read_csv(
#             file_path, comment="#", sep="\t", dtype={"site_no": str}
#         ).iloc[1:, :]
#         columns_names = df_flow.columns.tolist()
#         columns_flow = []
#         columns_flow_cd = []
#         for column_name in columns_names:
#             if "_00060_00003" in column_name and "_00060_00003_cd" not in column_name:
#                 columns_flow.append(column_name)
#         for column_name in columns_names:
#             if "_00060_00003_cd" in column_name:
#                 columns_flow_cd.append(column_name)
#         if len(columns_flow) > 1:
#             log.debug("there are some columns for flow, choose one\n")
#             df_date_temp = df_flow["datetime"]
#             date_temp = pd.to_datetime(df_date_temp).values.astype("datetime64[D]")
#             c_temp, ind1_temp, ind2_temp = np.intersect1d(
#                 date_temp, time_range, return_indices=True
#             )
#             num_nan_lst = []
#             for i in range(len(columns_flow)):
#                 out_temp = np.full([len(time_range)], np.nan)
#                 df_flow_temp = df_flow[columns_flow[i]].copy()
#                 df_flow_temp.loc[df_flow_temp == "Rat"] = np.nan
#                 df_flow_temp.loc[df_flow_temp == "Dis"] = np.nan
#                 df_flow_temp.loc[df_flow_temp == "Ice"] = np.nan
#                 df_flow_temp.loc[df_flow_temp == "Ssn"] = np.nan
#                 out_temp[ind2_temp] = df_flow_temp.iloc[ind1_temp]
#                 num_nan = np.isnan(out_temp).sum()
#                 num_nan_lst.append(num_nan)
#             num_nan_np = np.array(num_nan_lst)
#             index_flow_num = np.argmin(num_nan_np)
#             df_flow.rename(columns={columns_flow[index_flow_num]: "flow"}, inplace=True)
#             df_flow.rename(
#                 columns={columns_flow_cd[index_flow_num]: "mode"}, inplace=True
#             )
#         else:
#             for column_name in columns_names:
#                 if (
#                     "_00060_00003" in column_name
#                     and "_00060_00003_cd" not in column_name
#                 ):
#                     df_flow.rename(columns={column_name: "flow"}, inplace=True)
#                     break
#             for column_name in columns_names:
#                 if "_00060_00003_cd" in column_name:
#                     df_flow.rename(columns={column_name: "mode"}, inplace=True)
#                     break
#
#         columns = ["agency_cd", "site_no", "datetime", "flow", "mode"]
#         if df_flow.empty:
#             df_flow = pd.DataFrame(columns=columns)
#         flow_check = "flow" in df_flow.columns.intersection(columns)
#         if not flow_check:
#             data_temp = df_flow.loc[:, df_flow.columns.intersection(columns)]
#             # add nan column to data_temp
#             data_temp = pd.concat([data_temp, pd.DataFrame(columns=["flow", "mode"])])
#         else:
#             data_temp = df_flow.loc[:, columns]
#         # fix flow which is not numeric data
#         data_temp.loc[data_temp["flow"] == "Ice", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Ssn", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Tst", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Eqp", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Rat", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Dis", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Bkw", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "***", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "Mnt", "flow"] = np.nan
#         data_temp.loc[data_temp["flow"] == "ZFL", "flow"] = np.nan
#         # set negative value -- nan
#         obs = data_temp["flow"].astype("float").values
#         obs[obs < 0] = np.nan
#         # time range intersection. set points without data nan values
#         nt = len(time_range)
#         out = np.full([nt], np.nan)
#         # date in df is str，so transform them to datetime
#         df_date = data_temp["datetime"]
#         date = pd.to_datetime(df_date).values.astype("datetime64[D]")
#         c, ind1, ind2 = np.intersect1d(date, time_range, return_indices=True)
#         out[ind2] = obs[ind1]
#         return out


def get_observation_reader(cfg: Config) -> Callable:
    source = cfg.observations.name.lower()
    readers = {
        "dev": ZarrUSGSReader,
        "global": GlobalReader,
        "grdc": GRDCReader,
        "susquehanna": SusquehannaReader,
        # "usgs": USGSReader,
        "gages_3000": ZarrUSGSReader,
    }
    try:
        return readers[source]
    except KeyError:
        raise KeyError(
            "No defined data source. Use either: "
            "usgs, grdc, susquehanna, or gages_3000"
        )
