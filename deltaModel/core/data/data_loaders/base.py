from abc import ABC, abstractmethod
from typing import Dict

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class BaseDataLoader(Dataset, ABC):
    @abstractmethod
    def load_dataset(self) -> None:
        """Load dataset into dictionary of input arrays."""    
        if self.test_split:
            try:
                train_range = self.config['train_t_range'] 
                test_range = self.config['test_t_range']
            except KeyError:
                raise KeyError("Missing train or test time range in configuration.")

            self.train_dataset = self._preprocess_data(train_range)
            self.test_dataset = self._preprocess_data(test_range)
        else:
            self.dataset = self._preprocess_data(self.config['t_range'])

    @abstractmethod
    def _preprocess_data(self, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""
        pass

    def to_tensor(self, data: npt.NDArray) -> torch.Tensor:
        """Convert numpy array to Pytorch tensor."""
        tensor = torch.from_numpy(data).to(dtype=self.dtype, device=self.device)
        return tensor.requires_grad_(False)
