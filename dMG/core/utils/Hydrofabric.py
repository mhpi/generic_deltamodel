# TODO: Decide if this is necessary; from dMCdev @Tadd Bindas.
import logging
from dataclasses import dataclass, field
from typing import Union

import torch
import xarray as xr
from archive.Mapping import MeritMap
from core.utils.Dates import Dates

# from dMC.dataset_modules.utils.Network import FullZoneNetwork, Network

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    attributes: Union[torch.Tensor, None] = field(default=None)
    dates: Union[Dates, None] = field(default=None)
    mapping: Union[MeritMap, None] = field(default=None)
    
    # network: Union[Network, FullZoneNetwork, None] = field(default=None)

    normalized_attributes: Union[torch.Tensor, None] = field(default=None)
    # normalized_forcings: Union[torch.Tensor, None] = field(default=None)
    observations: Union[xr.Dataset, None] = field(default=None)
