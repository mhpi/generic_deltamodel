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
        return {
            key: (
                value[:, i_s:i_e, :] if value.ndim == 3 else value[i_s:i_e, :]
            ).to(dtype=torch.float32, device=self.device)
            for key, value in dataset.items()
        }
