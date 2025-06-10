from abc import ABC, abstractmethod

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BaseSampler(Dataset, ABC):
    """Base class for data samplers extended from PyTorch Dataset.

    All data samplers should inherit from this class to enforce minimum
    requirements for use within dMG.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    """

    def __init__(
        self,
    ):
        super().__init__()
        # self.config = config

        # Set dtype and device from config or provide defaults
        # self.dtype = self.config.get("dtype", torch.float32)
        # self.device = self.config.get("device", torch.device("cpu"))

    @abstractmethod
    def load_data(self):
        """Load data from a specific source."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Preprocess the data as needed."""
        pass

    def to_tensor(self, data: NDArray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to convert.

        Returns
        -------
        torch.Tensor
            The data as a PyTorch tensor.
        """
        return torch.tensor(
            data, dtype=self.dtype, device=self.device, requires_grad=False,
        )

    def validate_config(self):
        """Validate the configuration dictionary to ensure required keys."""
        required_keys = ["dtype", "device"]  # Add required keys here
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
