import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class BaseDataSampler(Dataset, ABC):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @abstractmethod
    def load_data(self):
        """Load data from a specific source."""
        pass

    @abstractmethod
    def preprocess_data(self):
        """Preprocess the data as needed."""
        pass



##### Depr #####
class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def getDataTs(self, config, varLst):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, config, varLst):
        raise NotImplementedError
