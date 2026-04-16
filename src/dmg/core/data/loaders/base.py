from abc import ABC, abstractmethod
from typing import Optional

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BaseLoader(Dataset, ABC):
    """Base class for data loaders extended from PyTorch Dataset.

    All data loaders should inherit from this class to enforce minimum
    requirements for use within dMG.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    test_split : bool, optional
        Whether to split data into training and testing sets. Default is False.
    overwrite : bool, optional
        Whether to overwrite existing data. Default is False.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    @abstractmethod
    def load_dataset(self) -> None:
        """Load dataset into dictionary of input arrays."""

    @abstractmethod
    def _preprocess_data(self, t_range: dict[str, str]) -> dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""

    def to_tensor(self, data: NDArray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        tensor = torch.from_numpy(data).to(dtype=self.dtype, device=self.device)
        return tensor.requires_grad_(False)
