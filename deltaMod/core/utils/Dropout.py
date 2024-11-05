import logging
from collections import defaultdict
from typing import Any, Callable, DefaultDict, List, Optional, Tuple

from pydantic import BaseModel

log = logging.getLogger(__name__)


class Dropout(BaseModel):
    dropout_threshold: Optional[int] = None
    funcs: List[Callable] = []
    minimum_zones: Optional[int] = 3
    zone: Optional[List[int]] = None

    def __init__(self, **kwargs):
        super(Dropout, self).__init__(
            dropout_threshold=kwargs["dropout_threshold"],
            minimum_zones=kwargs["minimum_zones"],
            zone=kwargs["zone"],
        )
        if self.dropout_threshold is not None:
            log.info(
                f"Applying dropout with threshold {self.dropout_threshold} in train mode."
            )
            self.funcs.append(self._threshold_dropout)
        if self.zone is not None:
            log.info(f"Creating network matrix using Zone(s) {self.zone}")
            self.funcs.append(self._zone_dropout)

    def __call__(
        self,
        pairs: DefaultDict[Any, List[Any]],
        gage_order: DefaultDict[Any, List[Any]],
    ) -> Tuple[DefaultDict[Any, List[Any]], DefaultDict[Any, List[Any]]]:
        if self.funcs is None:
            return pairs, gage_order
        else:
            for func in self.funcs:
                pairs, gage_order = func(pairs, gage_order)
            return pairs, gage_order

    def _threshold_dropout(
        self,
        pairs: DefaultDict[Any, List[Any]],
        gage_order: DefaultDict[Any, List[Any]],
    ) -> Tuple[DefaultDict[Any, List[Any]], DefaultDict[Any, List[Any]]]:
        min_zones_to_retain = self.minimum_zones
        network_threshold = self.dropout_threshold
        network_size_per_zone = {zone: len(pairs[zone]) for zone in pairs.keys()}
        total_network_size = sum([v for _, v in network_size_per_zone.items()])
        zones_sorted_by_size = sorted(
            network_size_per_zone, key=network_size_per_zone.get, reverse=True
        )
        for zone in zones_sorted_by_size:
            if (
                total_network_size <= network_threshold
                or len(pairs) <= min_zones_to_retain
            ):
                # Stop if the network is small enough or minimum zones reached
                break

            if zone in pairs:
                total_network_size = total_network_size - network_size_per_zone[zone]
                _ = pairs.pop(zone)
                _ = gage_order.pop(zone)
                log.info(
                    f"Dropped zone: {zone}, "
                    f"new total network size: {total_network_size}"
                )
        return pairs, gage_order

    def _zone_dropout(
        self,
        pairs: DefaultDict[Any, List[Any]],
        gage_order: DefaultDict[Any, List[Any]],
    ) -> Tuple[DefaultDict[Any, List[Any]], DefaultDict[Any, List[Any]]]:
        new_pairs = defaultdict(list)
        new_gage_order = defaultdict(list)
        try:
            for zone in self.zone:
                zone = str(zone)
                new_pairs[zone] = pairs.pop(zone)
                new_gage_order[zone] = gage_order.pop(zone)
            return new_pairs, new_gage_order
        except KeyError:
            raise KeyError(f"Cannot find zone: {zone} in the selected gages")
