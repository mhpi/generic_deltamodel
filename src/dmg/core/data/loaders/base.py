from abc import ABC, abstractmethod
from typing import Optional

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BaseLoader(Dataset, ABC):
    """Base class for data loaders extended from PyTorch Dataset.

    All data loaders should enherit from this class to enforce minimum
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
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        # self.config = config
        self.test_split = test_split
        self.overwrite = overwrite

    @abstractmethod
    def load_dataset(self) -> None:
        """Load dataset into dictionary of input arrays."""
        if self.test_split:
            try:
                train_range = self.config['train_t_range']
                test_range = self.config['test_t_range']
            except KeyError as e:
                raise KeyError("Missing train or test time range in configuration.") from e

            self.train_dataset = self._preprocess_data(train_range)
            self.test_dataset = self._preprocess_data(test_range)
        else:
            self.dataset = self._preprocess_data(self.config['t_range'])

    @abstractmethod
    def _preprocess_data(self, t_range: dict[str, str]) -> dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""
        pass

    def to_tensor(self, data: NDArray) -> torch.Tensor:
        """Convert numpy array to Pytorch tensor."""
        tensor = torch.from_numpy(data).to(dtype=self.dtype, device=self.device)
        return tensor.requires_grad_(False)
