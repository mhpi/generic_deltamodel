from abc import ABC, abstractmethod
from typing import Dict

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class BaseDataSampler(Dataset, ABC):
    """Base class for data samplers extended from PyTorch Dataset.
    
    All data samplers should inherit from this class to enforce minimum
    requirements for use within dMG.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    @abstractmethod
    def load_data(self):
        """Load data from a specific source."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Preprocess the data as needed."""
        pass

    def to_tensor(self, data: npt.NDArray) -> torch.Tensor:
        """Convert numpy array to Pytorch tensor."""
        return torch.Tensor(
            data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
