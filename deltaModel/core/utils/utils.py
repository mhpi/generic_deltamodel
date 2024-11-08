import logging
from typing import Dict, List

import numpy as np
import torch
from conf.config import Config

log = logging.getLogger(__name__)



def find_shared_keys(*dicts: dict) -> List[str]:
    """Find keys shared between multiple dictionaries.

    Parameters
    ----------
    *dicts : dict
        Variable number of dictionaries.

    Returns
    -------
    List[str]
        A list of keys shared between the input dictionaries.
    """
    if len(dicts) == 1:
        return list()

    # Start with the keys of the first dictionary
    shared_keys = set(dicts[0].keys())

    # Intersect with the keys of all other dictionaries
    for d in dicts[1:]:
        shared_keys.intersection_update(d.keys())

    return list(shared_keys)


def filter_nan(attr: torch.Tensor, idx: int, config: Config):
    nan_idx = torch.nonzero(input=torch.isnan(attr), as_tuple=True)
    if nan_idx[0].shape[0] > 0:
        # Masking NaN values with the default value from config.py
        try:
            default = config.params.attribute_defaults[idx]
        except IndexError:
            msg = (
                "Index of attribute defaults is out of range. Check your Config.py file"
            )
            log.exception(msg)
            raise IndexError(msg)
        attr[nan_idx] = default
    return attr


def normalize_streamflow(
    data: torch.Tensor, gage_ids: np.ndarray, stat_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Taken from the transNorm function of hydroDL

    TODO: goes to hydroDL2
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

    TODO: goes to hydroDL2
    """

    output = torch.zeros([len(gage_ids), normalized_data.shape[1]], dtype=torch.float64)
    for idx, gage in enumerate(gage_ids):
        statistics = stat_dict[gage]
        output[idx] = normalized_data[idx] * statistics[3] + statistics[2]

    return output
