import logging
from typing import List, Tuple

from data import BaseDataset
from data.utils import create_hydrofabric_attributes  # scale_scipy,
from data.utils import create_hydrofabric_observations, pad_gage_id, scale
from data.utils.Hydrofabric import Hydrofabric

from archive.Mapping import MeritMap

# from data.utils.Network import Network

log = logging.getLogger(__name__)


class GeneralDataset(BaseDataset):
    def __init__(self, **kwargs):
        self.attributes = kwargs["attributes"]
        self.attribute_statistics = kwargs["attribute_statistics"]
        self.cfg = kwargs["cfg"]
        self.dates = kwargs["dates"]
        self.dropout = kwargs["dropout"]
        self.global_to_zone_mapping = kwargs["global_to_zone_mapping"]
        self.gage_dict = kwargs["gage_dict"]
        self.observations = kwargs["observations"]
        self.zone_to_global_mapping = kwargs["zone_to_global_mapping"]

    def __len__(self):
        """
        Returns the total number of gauges.
        """
        return len(self.gage_dict["STAID"])

    def __getitem__(self, idx) -> Tuple[int, str, str]:
        """
        Generates one sample of data.
        """
        if self.cfg.observations.name.lower() != "grdc":
            padded_gage_idx = pad_gage_id(self.gage_dict["STAID"][idx])
        else:
            padded_gage_idx = self.gage_dict["STAID"][idx]
        # using the first two digits of the merit COMID for location
        zone = str(self.gage_dict["COMID"][idx])[:2]

        return idx, padded_gage_idx, zone

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        data: List[Tuple[int, str, str]] = args[0]
        self.dates.calculate_time_period()
        network = Network(
            attributes=self.attributes,
            cfg=self.cfg,
            data=data,
            dropout=self.dropout,
            gage_dict=self.gage_dict,
            global_to_zone_mapping=self.global_to_zone_mapping,
            zone_to_global_mapping=self.zone_to_global_mapping,
        )
        mapping = MeritMap(self.cfg, self.dates, network)
        hydrofabric_attributes = create_hydrofabric_attributes(
            cfg=self.cfg,
            attributes=self.attributes,
            network=network,
            names=[
                "len",
                "len_dir",
                "sinuosity",
                "slope",
                "stream_drop",
                "uparea",
            ],
        )
        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_dict=self.gage_dict,
            network=network,
            observations=self.observations,
        )
        # scipy_attr = scale_scipy(scaler=self.scaler, x=hydrofabric_attributes)
        normalized_attributes = scale(
            df=self.attribute_statistics,
            x=hydrofabric_attributes,
            names=[
                "len",
                "len_dir",
                "sinuosity",
                "slope",
                "stream_drop",
                "uparea",
            ],
        )
        return Hydrofabric(
            attributes=hydrofabric_attributes,
            dates=self.dates,
            mapping=mapping,
            network=network,
            normalized_attributes=normalized_attributes,
            observations=hydrofabric_observations,
        )
