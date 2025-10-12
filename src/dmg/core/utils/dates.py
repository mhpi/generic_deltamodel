import logging
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator, Field

log = logging.getLogger(__name__)


class Dates(BaseModel):
    """Class to handle time-related operations and configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    daily_format: str = "%Y/%m/%d"
    origin_start_date: str = "1980/01/01"
    start_time: str
    end_time: str
    rho: Optional[int] = None

    # Use lambdas in default_factory to correctly initialize empty objects
    batch_daily_time_range: pd.DatetimeIndex = Field(
        default_factory=lambda: pd.DatetimeIndex([])
    )
    daily_time_range: pd.DatetimeIndex = Field(
        default_factory=lambda: pd.DatetimeIndex([])
    )
    hourly_indices: torch.Tensor = Field(default_factory=lambda: torch.empty(0))
    numerical_time_range: NDArray[np.float32] = Field(
        default_factory=lambda: np.empty(0)
    )

    def __init__(self, time_range: dict, rho: Optional[int]):
        super().__init__(
            start_time=time_range['start_time'],
            end_time=time_range['end_time'],
            rho=rho,
        )

    @model_validator(mode="after")
    def validate_dates(self) -> 'Dates':
        """Validate the dates configuration."""
        if isinstance(self.rho, int) and self.rho > len(self.daily_time_range):
            raise ValueError(
                "Rho needs to be smaller than the routed period between start and end times"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Dates class."""
        self.daily_time_range = pd.to_datetime(
            pd.date_range(start=self.start_time, end=self.end_time, freq="D")
        )
        # Set initial batch to the full range
        self.set_batch_time(self.daily_time_range)

    def set_batch_time(self, daily_time_range: pd.DatetimeIndex):
        """Set the batch time range using efficient arithmetic."""
        self.batch_daily_time_range = daily_time_range

        # Calculate the start hour index relative to the beginning of the entire period
        if not self.daily_time_range.empty and not daily_time_range.empty:
            start_offset_days = (daily_time_range[0] - self.daily_time_range[0]).days
            start_hour_index = start_offset_days * 24

            # Calculate the number of hours in the current batch
            num_hours = len(daily_time_range) * 24

            # Generate the hourly indices directly using torch.arange
            self.hourly_indices = torch.arange(
                start_hour_index, start_hour_index + num_hours, dtype=torch.long
            )

        origin_start_dt = datetime.strptime(self.origin_start_date, self.daily_format)

        # Calculate numerical day range (can be vectorized for slight speedup)
        start_day = (daily_time_range[0].to_pydatetime() - origin_start_dt).days
        end_day = (daily_time_range[-1].to_pydatetime() - origin_start_dt).days
        self.numerical_time_range = np.arange(start_day, end_day + 1)

    def calculate_time_period(self) -> None:
        """Select a random sub-period of length rho."""
        if self.rho is not None:
            sample_size = len(self.daily_time_range)
            if sample_size > self.rho:
                random_start = torch.randint(
                    low=0,
                    high=sample_size - self.rho,
                    size=(1,),
                ).item()
                self.set_date_range(slice(random_start, random_start + self.rho))

    def set_date_range(self, chunk: Union[NDArray, slice]) -> None:
        """Set the date range to a specific chunk."""
        new_daily_range = self.daily_time_range[chunk]
        self.set_batch_time(new_daily_range)

    def date_to_int(self) -> list[int]:
        """Convert date strings to integers."""
        start = int(self.start_time.replace('/', ''))
        end = int(self.end_time.replace('/', ''))
        return [start, end]
