from typing import Dict, Optional

import numpy as np
import torch

from core.data import random_index
from core.data.data_samplers.base import BaseDataSampler


class HydroMSDataSampler(BaseDataSampler):
    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['dpl_model']['phy_model']['warm_up']
        self.rho = config['dpl_model']['rho']

    def load_data(self):
        """Custom implementation for loading data."""
        raise NotImplementedError

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        raise NotImplementedError

    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: np.ndarray,
        i_t: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
    ) -> torch.Tensor:
        """Select a subset of input tensor."""
        if self.config['mode'] == 'train':
            raise NotImplementedError("Method not implemented. Multiscale training with sampler will be enabled at a later date.")

    def get_training_sample(
        self,
        dataset: Dict[str, np.ndarray],
        ngrid_train: int,
        nt: int
    ) -> Dict[str, torch.Tensor]:
        """Generate a training batch."""
        raise NotImplementedError("Method not implemented. Multiscale training with sampler will be enabled at a later date.")

    def get_validation_sample(
        self,
        dataset: Dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> Dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        dataset_sample = {}
        for key, value in dataset.items():
            if value.ndim == 3:
                if key in ['x_phy', 'xc_nn_norm']:
                    warm_up = 0
                else:
                    warm_up = self.config['dpl_model']['phy_model']['warm_up']
                dataset_sample[key] = torch.tensor(
                    value[warm_up:, i_s:i_e, :],
                    dtype=torch.float32,
                    device = self.config['device']
                )
            elif value.ndim == 2:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e, :],
                    dtype=torch.float32,
                    device = self.config['device']
                )
            elif value.ndim == 1:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e],
                    dtype=torch.float32,
                    device = self.config['device']
                )
            else:
                raise ValueError(f"Incorrect input dimensions. {key} array must have 1, 2 or 3 dimensions.")
        return dataset_sample
