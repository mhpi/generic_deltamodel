from abc import ABC, abstractmethod

import numpy.typing as npt
import torch
from torch.utils.data import Dataset


class BaseDataSampler(Dataset, ABC):
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



##### Depr #####
class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def getDataTs(self, config, varLst):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, config, varLst):
        raise NotImplementedError
