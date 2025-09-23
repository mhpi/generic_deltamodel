import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger(__name__)



class Dates(BaseModel):
    """Class to handle time-related operations and configurations.

    Adapted from Tadd Bindas.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    daily_format: str = "%Y/%m/%d"
    hourly_format: str = "%Y/%m/%d %H:%M:%S"
    origin_start_date: str = "1980/01/01"
    start_time: str
    end_time: str
    rho: Optional[int] = None
    batch_daily_time_range: Optional[pd.DatetimeIndex] = pd.DatetimeIndex(
        [], dtype="datetime64[ns]",
    )
    batch_hourly_time_range: Optional[pd.DatetimeIndex] = pd.DatetimeIndex(
        [], dtype="datetime64[ns]",
    )
    daily_time_range: Optional[pd.DatetimeIndex] = pd.DatetimeIndex(
        [], dtype="datetime64[ns]",
    )
    hourly_indices: Optional[torch.Tensor] = torch.empty(0)
    hourly_time_range: Optional[pd.DatetimeIndex] = pd.DatetimeIndex(
        [], dtype="datetime64[ns]",
    )
    numerical_time_range: Optional[NDArray[np.float32]] = np.empty(0)

    def __init__(self, time_range, rho):
        super().__init__(
            start_time=time_range['start_time'],
            end_time=time_range['end_time'],
            rho = rho,
        )

    @model_validator(mode="after")
    @classmethod
    def validate_dates(cls, dates: Any) -> Any:
        """Validate the dates configuration.
        
        Parameters
        ----------
        dates : Any
            The dates configuration.

        Returns
        -------
        Any
            The validated dates configuration
        """
        rho = dates.rho
        if isinstance(rho, int):
            if rho > len(dates.daily_time_range):
                log.exception(
                    ValueError(
                        "Rho needs to be smaller than the routed period between start and end times",
                    ),
                )
                raise ValueError(
                    "Rho needs to be smaller than the routed period between start and end times",
                )
        return dates

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Dates class.

        Parameters
        ----------
        __context : Any
            The context of the model.
        """
        self.daily_time_range = pd.date_range(
            datetime.strptime(self.start_time, self.daily_format),
            datetime.strptime(self.end_time, self.daily_format),
            freq="D",
            inclusive="both",
        )
        self.hourly_time_range = pd.date_range(
            start=self.daily_time_range[0],
            end=self.daily_time_range[-1],
            freq="h",
            inclusive="left",
        )
        self.batch_daily_time_range = self.daily_time_range
        self.set_batch_time(self.daily_time_range)

    def set_batch_time(self, daily_time_range: pd.DatetimeIndex):
        """Set the batch time range.
        
        Parameters
        ----------
        daily_time_range : pd.DatetimeIndex
            The daily time range.
        """
        self.batch_hourly_time_range = pd.date_range(
            start=daily_time_range[0],
            end=daily_time_range[-1],
            freq="h",
            inclusive="left",
        )
        origin_start_date = datetime.strptime(self.origin_start_date, self.daily_format)
        origin_base_start_time = int(
            (daily_time_range[0].to_pydatetime() - origin_start_date).total_seconds()
            / 86400,
        )
        origin_base_end_time = int(
            (daily_time_range[-1].to_pydatetime() - origin_start_date).total_seconds()
            / 86400,
        )

        # The indices for the dates in your selected routing time range.
        self.numerical_time_range = np.arange(
            origin_base_start_time, origin_base_end_time + 1, 1,
        )

        common_elements = self.hourly_time_range.intersection(
            self.batch_hourly_time_range,
        )
        self.hourly_indices = torch.tensor(
            [self.hourly_time_range.get_loc(time) for time in common_elements],
        )

    def calculate_time_period(self) -> None:
        """Calculate the time period."""
        if self.rho is not None:
            sample_size = len(self.daily_time_range)
            random_start = torch.randint(
                low=0, high=sample_size - self.rho, size=(1, 1),
            )[0][0].item()
            self.batch_daily_time_range = self.daily_time_range[
                random_start : (random_start + self.rho)
            ]
            self.set_batch_time(self.batch_daily_time_range)

    def set_date_range(self, chunk: NDArray[np.float32]) -> None:
        """Set the date range.

        Parameters
        ----------
        chunk : NDArray[np.float32]
            The chunk of the date range.
        """
        self.batch_daily_time_range = self.daily_time_range[chunk]
        self.set_batch_time(self.batch_daily_time_range)

    def date_to_int(self):
        """Convert date strings to integers.

        Using this temporarily to convert config date values to compatible
        representations for data reading.

        Returns
        -------
        list
            The list of converted date values
        """
        date_time_format = "%Y/%m/%d"
        date_int = "%Y%m%d"
        start = datetime.strptime(self.start_time, date_time_format).strftime(date_int)
        end = datetime.strptime(self.end_time, date_time_format).strftime(date_int)

        return [int(start), int(end)]
