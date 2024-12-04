from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
import numpy.typing as npt
from typing import Dict

class BaseDataLoader(Dataset, ABC):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def load_dataset(self) -> None:
        """Load dataset into dictionary of input arrays."""    
        if self.test_split:
            try:
                train_range = self.config['train_t_range'] 
                test_range = self.config['test_t_range']
            except KeyError:
                raise KeyError("Missing train or test time range in configuration.")

            self.train_dataset = self.preprocess_data(train_range)
            self.test_dataset = self.preprocess_data(test_range)
        else:
            self.dataset = self.preprocess_data(self.config['t_range'])

    @abstractmethod
    def preprocess_data(self, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Read, preprocess, and return data as dictionary of torch tensors."""
        pass

    def to_tensor(self, data: npt.NDArray) -> torch.Tensor:
        """Convert numpy array to Pytorch tensor."""
        return torch.Tensor(
            data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )