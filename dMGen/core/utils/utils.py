# TODO: Go through these functions from `__init__.py` in dMCdev to see what is needed for PMI.
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
import xarray as xr
import zarr
from conf.config import Config
from core.utils.Dates import Dates
# from data.utils.Network import FullZoneNetwork, Network
from tqdm import tqdm

log = logging.getLogger(__name__)



def find_shared_keys(*dicts) -> list:
    """
    Find keys shared between multiple dictionaries.

    Args:
        *dicts: Variable number of dictionaries.

    Returns:
        shared_keys: A list of keys shared between input dictionaries.
    """
    if len(dicts) == 1:
        return list()

    # Start with the keys of the first dictionary
    shared_keys = set(dicts[0].keys())

    # Intersect with the keys of all other dictionaries
    for d in dicts[1:]:
        shared_keys.intersection_update(d.keys())

    return list(shared_keys)


def pad_gage_id(number: Union[int, str]) -> str:
    """
    Left pads a number with '0' if it has 7 digits to make it 8 digits long.

    Parameters:
        number (int or str): The number to be left-padded.

    Returns:
        str: The left-padded number as a string.
    """
    return str(number).zfill(8)


def read_gage_info(gage_info_path: Path) -> Dict[str, List[str]]:
    """
    Read gage information from a specified file.

    Parameters:
        gage_info_path (Path): The file path for the gage information CSV.

    Returns:
        dict: A dictionary with gage information.

    Raises:
        FileNotFoundError: If the specified file path is not found.
        KeyError: If the CSV file is missing any of the expected column headers.
    """
    expected_column_names = [
        "STAID",
        "STANAME",
        "HUC02",
        "DRAIN_SQKM",
        "LAT_GAGE",
        "LNG_GAGE",
        "COMID",
        "edge_intersection",
        "zone_edge_id",
        "zone_edge_uparea",
        "zone_edge_vs_gage_area_difference",
        "drainage_area_percent_error",
    ]

    try:
        df = pd.read_csv(gage_info_path, delimiter=",")

        if not set(expected_column_names).issubset(set(df.columns)):
            missing_headers = set(expected_column_names) - set(df.columns)
            raise KeyError(
                f"The CSV file is missing the following headers: " f"{missing_headers}"
            )

        df["STAID"] = df["STAID"].astype(str)

        out = {
            field: df[field].tolist()
            if field == "STANAME"
            else df[field].values.tolist()
            for field in expected_column_names
            if field in df.columns
        }
        return out
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {gage_info_path}")


def get_attribute(attributes: zarr.Group, attribute: str) -> torch.Tensor:
    """
    Get the attribute data from the zarr group.

    Parameters:
        attributes (zarr.Group): The zarr group containing attribute data.
        attribute (str): The name of the attribute to retrieve.

    Returns:
        torch.Tensor: The attribute data as a torch tensor.

    """
    all_attr = []
    for k, v in attributes.items():
        try:
            attr = torch.tensor(v[attribute][:], dtype=torch.float64)
            all_attr.append(attr)
        except KeyError:
            msg = f"{attribute} not found in zone {k}."
            log.exception(msg)
            raise KeyError
    data = torch.cat(all_attr, dim=0)
    return data


def get_attributes(attributes: zarr.Group, cfg: Config, zone: str, idx) -> torch.Tensor:
    """
    Get attributes from a zarr group for a given zone and index.

    Parameters:
    - attributes (zarr.Group): The zarr group containing the attributes.
    - cfg (Config): The configuration object.
    - zone (str): The zone for which to retrieve attributes.
    - idx: The index of the attributes to retrieve.

    Returns:
    - torch.Tensor: A tensor containing the attributes for the given zone and index.

    """
    merit_zone = attributes[str(zone)]
    _attributes = torch.zeros(
        (len(idx), len(cfg.params.attributes)), dtype=torch.float64
    )
    for _idx, attribute in enumerate(cfg.params.attributes):
        _attributes[:, _idx] = torch.tensor(
            merit_zone[attribute][idx], dtype=torch.float64
        )
    return _attributes


# def create_hydrofabric_attributes(
#     cfg: Config,
#     attributes: zarr.Group,
#     network: Union[FullZoneNetwork, Network],
#     names: List[str] = ["all_attributes"],
# ) -> torch.Tensor:
#     """
#     Create hydrofabric attributes.

#     This function takes in a configuration object, a zarr group containing attribute data, and a network object (either a FullZoneNetwork or Network). It returns a torch tensor containing the hydrofabric attributes.

#     Parameters:
#         cfg (Config): The configuration object.
#         attributes (zarr.Group): The zarr group containing attribute data.
#         network (Union[FullZoneNetwork, Network]): The network object.
#         names (List[str]): THE names of the attributes to retrieve. Defaults to ["all_attributes"].

#     Returns:
#         torch.Tensor: The hydrofabric attributes.

#     """
#     all_attr = []
#     global_indices = network.edge_order
#     for idx, attribute in enumerate(
#         tqdm(
#             cfg.params.attributes,
#             desc="\rReading attribute data",
#             ncols=140,
#             ascii=True,
#         )
#     ):
#         # TODO make this check clearer. A little hacky
#         if attribute in names or names == ["all_attributes"]:
#             attr = get_attribute(attributes, attribute)
#             nan_free_attr = filter_nan(attr, idx, cfg)
#             all_attr.append(nan_free_attr)
#     data = torch.stack(all_attr, dim=1)
#     subset_data = data[global_indices]
#     attributes_ = subset_data
    return attributes_


def filter_nan(attr: torch.Tensor, idx: int, cfg: Config):
    nan_idx = torch.nonzero(input=torch.isnan(attr), as_tuple=True)
    if nan_idx[0].shape[0] > 0:
        # Masking NaN values with the default value from config.py
        try:
            default = cfg.params.attribute_defaults[idx]
        except IndexError:
            msg = (
                "Index of attribute defaults is out of range. Check your Config.py file"
            )
            log.exception(msg)
            raise IndexError(msg)
        attr[nan_idx] = default
    return attr


# def create_hydrofabric_observations(
#     dates: Dates,
#     gage_dict: Dict[str, Any],
#     network: Union[FullZoneNetwork, Network],
#     observations: xr.Dataset,
# ) -> xr.Dataset:
#     """
#     Create hydrofabric observations.

#     Parameters:
#         dates (Dates): An instance of the Dates class containing the batch daily and hourly time ranges.
#         gage_dict (Dict[str, Any]): A dictionary containing gage information.
#         network (Union[FullZoneNetwork, Network]): An instance of the Network class.
#         observations (xr.Dataset): An xarray dataset containing the observations.

#     Returns:
#         xr.Dataset,: An xarray dataset containing the interpolated streamflow values.

#     """
#     log.info("Reading Observations")
#     if network.cfg.observations.name != "grdc":
#         gage_ids = [
#             pad_gage_id(gage_dict["STAID"][x])
#             for x in network.gage_information["gage_dict_idx"]
#         ]
#     else:
#         gage_ids = [
#             gage_dict["STAID"][x] for x in network.gage_information["gage_dict_idx"]
#         ]
#     ds = observations.sel(time=dates.batch_daily_time_range, gage_id=gage_ids)
#     ds_interpolated = ds.interp(time=dates.batch_hourly_time_range, method="linear")
#     return ds_interpolated


def determine_proc_zone(cfg: Config, data: List[Tuple[int, str, str]]) -> Config:
    """
    Setting the Zone that each rank is responsible for testing on.

    Parameters:
        cfg (Config): The configuration object containing the path to the data sources.
        data (List[Tuple[int, str, str]]): The formatted data for the torch Dataset.

    Returns:
        Config: The configuration object containing the zone that each rank is responsible for testing on.
    """
    rank = cfg.local_rank
    world_size = cfg.world_size

    zone_counts = defaultdict(int)
    for _, _, zone in data:
        zone_counts[int(zone)] += 1
    zones = np.array(list(zone_counts.keys()))
    gage_counts = np.array(list(zone_counts.values()))

    sorted_idx = np.argsort(gage_counts)
    sorted_zones = np.flip(zones[sorted_idx])
    if rank < (world_size - 1):
        rank_zones = sorted_zones[rank].tolist()
    else:
        rank_zones = sorted_zones[rank:].tolist()

    if isinstance(rank_zones, int):
        rank_zones = [rank_zones]
    cfg.test.zone = rank_zones
    return cfg


def format_gage_data(
    cfg: Config, gage_dict: Dict[str, Any]
) -> List[Tuple[int, str, str]]:
    """
    Formatting data from the gage dictionary into usable input for the torch Dataset

    Parameters:
        cfg (Config): The configuration object containing the path to the data sources.
        gage_dict (Dict[str, Any]): The dictionary containing gage information.

    Returns:
        List[Tuple[int, str, str]]: The formatted data for the torch Dataset. The reason
        for the tuple is to have the index, the gage id, and the zone for each gage in a
        format similar to how the torch Dataset generates it in GeneralDataset.

    """
    if cfg.observations.name.lower() != "grdc":
        padded_gage_ids = [pad_gage_id(_id) for _id in gage_dict["STAID"]]
    else:
        padded_gage_ids = [_id for _id in gage_dict["STAID"]]
    zones = [str(comid)[:2] for comid in gage_dict["COMID"]]
    idx = range(len(padded_gage_ids))
    data = [(idx[i], padded_gage_ids[i], zones[i]) for i in idx]
    return data


def set_attributes(cfg: Config) -> zarr.Group:
    """
    Set the attributes of a zarr group.

    Parameters:
        cfg (Config): The configuration object containing the path to the data sources.

    Returns:
        zarr.Group: The zarr group containing the attributes for all zones.

    """
    attributes = zarr.open_group(Path(cfg.data_sources.edges), mode="r")
    return attributes


def set_global_indexing(
    attributes: zarr.Group,
) -> Tuple[Dict[int, Tuple[Any, int]], Dict[Tuple[Any, int], int]]:
    """
    Set global indexing for attributes.

    This function takes a zarr.Group object containing attributes and creates
    global indexing for each attribute zone. It returns two dictionaries:
        global_to_zone_mapping and zone_to_global_mapping.

    Parameters:
    - attributes (zarr.Group): The zarr.Group object containing attributes.

    Returns:
    - Tuple[Dict[int, Tuple[Any, int]], Dict[Tuple[Any, int], int]]:
        A tuple of two dictionaries. The first dictionary maps global indices to zone
        and local indices, while the second dictionary maps zone and local indices to
        global indices.

    Example Usage:
    ```python
    import zarr

    # Create a zarr.Group object
    attributes = zarr.Group()

    # Call the set_global_indexing function
    global_to_zone_mapping, zone_to_global_mapping = set_global_indexing(attributes)
    """
    count = 0
    global_to_zone_mapping = {}
    zone_to_global_mapping = {}

    for zone, attr in tqdm(
        attributes.items(),
        desc="\rCreating global edge indexing",
        ncols=140,
        ascii=True,
    ):
        uparea = attr.uparea[:]
        for local_idx, value in enumerate(uparea):
            global_to_zone_mapping[count] = (zone, local_idx)
            zone_to_global_mapping[(zone, local_idx)] = count
            count += 1
    return global_to_zone_mapping, zone_to_global_mapping


# def set_min_max_scaler(
#     cfg: Config, attributes: zarr.Group
# ) -> preprocessing.MinMaxScaler:
#     """
#     Set up and return a MinMaxScaler object for scaling attributes.

#     Parameters:
#         cfg (Config): The configuration object containing parameters.
#         attributes (zarr.Group): The zarr group containing attribute data.

#     Returns:
#         preprocessing.MinMaxScaler: The fitted MinMaxScaler object.

#     """
#     scaler = preprocessing.MinMaxScaler()
#     all_attr = []
#     for idx, attribute in enumerate(
#         tqdm(cfg.params.attributes, desc="\rFitting scaler", ncols=140, ascii=True)
#     ):
#         attr = get_attribute(attributes, attribute)
#         nan_free_attr = filter_nan(attr, idx, cfg)
#         all_attr.append(nan_free_attr)
#     try:
#         data = torch.stack(all_attr, dim=1).numpy()
#     except RuntimeError as e:
#         msg = f"Ragged array: {e}"
#         log.exception(msg)
#         raise RuntimeError(msg)
#     scaler.fit(data)
#     return scaler


def set_min_max_statistics(
    cfg: Config,
    attributes: zarr.Group,
) -> pl.DataFrame:
    attribute_name = []
    all_attr = []
    for idx, attribute in enumerate(cfg.params.attributes):
        attr = get_attribute(attributes, attribute)
        nan_free_attr = filter_nan(attr, idx, cfg)
        attribute_name.append(attribute)
        all_attr.append(nan_free_attr)
    try:
        data = torch.stack(all_attr, dim=1).numpy()
    except RuntimeError as e:
        msg = f"Ragged array: {e}"
        log.exception(msg)
        raise RuntimeError(msg)
    json_ = {
        "attribute": attribute_name,
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }
    df = pl.DataFrame(
        data=json_,
    )
    return df


# def scale_scipy(scaler: preprocessing.MinMaxScaler, x: torch.Tensor) -> torch.Tensor:
#     """
#     Scale the input tensor using the provided scaler.

#     Parameters:
#         scaler (preprocessing.MinMaxScaler): The scaler object used to scale the input tensor.
#         x (torch.Tensor): The input tensor to be scaled.

#     Returns:
#         torch.Tensor: The scaled tensor.

#     """
#     x_np = x.cpu().numpy()
#     x_tensor = torch.tensor(scaler.transform(x_np), dtype=torch.float32)
#     return x_tensor


def scale(
    df: pl.DataFrame,
    x: torch.Tensor,
    names: List[str] = ["all_attributes"],
) -> torch.Tensor:
    """
    Performing a min/max normalization for attributes:

    Parameters:
        df (pl.DataFrame): The scaler object used to scale the input tensor.
        x (torch.Tensor): The input tensor to be scaled.
        names (List[str]): The names of the attributes to be scaled. Defaults to ["all_attributes"].
    Returns:
        torch.Tensor: The scaled tensor.

    """
    if names == ["all_attributes"]:
        names = df["attribute"].to_list()

    min_values = df.filter(pl.col("attribute").is_in(names))["min"]
    max_values = df.filter(pl.col("attribute").is_in(names))["max"]

    ranges = torch.tensor(max_values - min_values)

    if 0 in ranges:
        msg = "min and max are the same. There is no range. Check input attributes"
        log.exception(msg)
        raise ValueError(msg)

    min_values = torch.tensor(min_values)

    normalized_data = ((x - min_values) / ranges).to(torch.float32)

    return normalized_data


def normalize_streamflow(
    data: torch.Tensor, gage_ids: np.ndarray, stat_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Taken from the transNorm function of hydroDL
    """
    output = torch.zeros([len(gage_ids), data.shape[1]], dtype=torch.float64)
    for idx, gage in enumerate(gage_ids):
        statistics = stat_dict[gage]
        output[idx] = (data[idx] - statistics[2]) / statistics[3]

    return output


def denormalize_streamflow(
    normalized_data: torch.Tensor,
    gage_ids: np.ndarray,
    stat_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Taken from the transNorm function of hydroDL
    """
    output = torch.zeros([len(gage_ids), normalized_data.shape[1]], dtype=torch.float64)
    for idx, gage in enumerate(gage_ids):
        statistics = stat_dict[gage]
        output[idx] = normalized_data[idx] * statistics[3] + statistics[2]

    return output

def change_param_range(param, bounds):
    out = param * (bounds[1] - bounds[0]) + bounds[0]
    return out

def param_bounds_2D(params, num, bounds, ndays, nmul):
    out_temp = (
            params[:, num * nmul: (num + 1) * nmul]
            * (bounds[1] - bounds[0])
            + bounds[0]
    )
    out = out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(
        ndays, params.shape[0], nmul
    )
    return out
