from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.samplers.base import BaseSampler


class MsHydroSampler(BaseSampler):
    """Multiscale hydrological data sampler.
    
    Parameters
    ----------
    config
        Configuration dictionary.
    """
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['delta_model']['phy_model']['warm_up']
        self.rho = config['delta_model']['rho']

    def load_data(self):
        """Custom implementation for loading data."""
        raise NotImplementedError

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        raise NotImplementedError

    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
    ) -> torch.Tensor:
        """Select a subset of input tensor."""
        if self.config['mode'] == 'train':
            raise NotImplementedError("Method not implemented. Multiscale training with sampler will be enabled at a later date.")

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch."""
        raise NotImplementedError("Method not implemented. Multiscale training with sampler will be enabled at a later date.")

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        dataset_sample = {}
        for key, value in dataset.items():
            if value.ndim == 3:
                if key in ['x_phy', 'xc_nn_norm']:
                    warm_up = 0
                else:
                    warm_up = self.config['delta_model']['phy_model']['warm_up']
                dataset_sample[key] = torch.tensor(
                    value[warm_up:, i_s:i_e, :],
                    dtype=torch.float32,
                    device = self.config['device'],
                )
            elif value.ndim == 2:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e, :],
                    dtype=torch.float32,
                    device = self.config['device'],
                )
            elif value.ndim == 1:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e],
                    dtype=torch.float32,
                    device = self.config['device'],
                )
            else:
                raise ValueError(f"Incorrect input dimensions. {key} array must have 1, 2 or 3 dimensions.")
        return dataset_sample
