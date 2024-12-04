from deltaModel.core.data.data_loaders.base import BaseDataLoader


class HydroDataSampler(BaseDataLoader):
    def __init__(self, data, labels):
        super().__init__(data, labels)

    def load_data(self):
        # Custom implementation for loading data
        print("Loading data...")

    def preprocess_data(self):
        # Custom implementation for preprocessing data
        print("Preprocessing data...")
