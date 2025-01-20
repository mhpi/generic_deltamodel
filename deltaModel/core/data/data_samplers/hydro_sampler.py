from typing import Dict, Optional

import numpy as np
import torch

from core.data import random_index
from core.data.data_samplers.base import BaseDataSampler


class HydroDataSampler(BaseDataSampler):
    def __init__(
        self,
        config: Dict
    ):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['dpl_model']['phy_model']['warm_up']
        self.rho = config['dpl_model']['rho']

    def load_data(self):
        """Custom implementation for loading data."""
        print("Loading data...")

    def preprocess_data(self):
        """Custom implementation for preprocessing data."""
        print("Preprocessing data...")

    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: np.ndarray,
        i_t: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        tuple_out: bool = False,
        has_grad: bool = False
    ) -> torch.Tensor:
        """Select a subset of input tensor."""
        batch_size, nx, nt = len(i_grid), x.shape[-1], x.shape[0]
        rho, warm_up = self.rho, self.warm_up

        # Handle time indexing and create an empty tensor for selection
        if i_t is not None:
            x_tensor = torch.zeros(
                [rho + warm_up, batch_size, nx],
                device=self.device,
                requires_grad=has_grad
            )
            for k in range(batch_size):
                x_tensor[:, k:k + 1, :] = x[i_t[k] - warm_up:i_t[k] + rho, i_grid[k]:i_grid[k] + 1, :]
        else:
            x_tensor = x[:, i_grid, :].float().to(self.device) if x.ndim == 3 else x[i_grid, :].float().to(self.device)

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, rho + warm_up, 1)
            return (x_tensor, c_tensor) if tuple_out else torch.cat((x_tensor, c_tensor), dim=2)

        return x_tensor

    def get_training_sample(
        self,
        dataset: Dict[str, np.ndarray],
        ngrid_train: int,
        nt: int
    ) -> Dict[str, torch.Tensor]:
        """Generate a training batch."""
        batch_size = self.config['train']['batch_size']
        i_sample, i_t = random_index(ngrid_train, nt, (batch_size, self.rho), warm_up=self.warm_up)

        return {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'c_phy': dataset['c_phy'][i_sample],
            'c_nn': dataset['c_nn'][i_sample],
            'xc_nn_norm': self.select_subset(dataset['xc_nn_norm'], i_sample, i_t, has_grad=False),
            'target': self.select_subset(dataset['target'], i_sample, i_t)[self.warm_up:, :],
            'batch_sample': i_sample,
        }

    def get_validation_sample(
        self,
        dataset: Dict[str, torch.Tensor],
        i_s: int,
        i_e: int
    ) -> Dict[str, torch.Tensor]:
        """Generate a validation batch."""
        return {
            key: (
                value[:, i_s:i_e, :] if value.ndim == 3 else value[i_s:i_e, :]
            ).to(dtype=torch.float32, device=self.device)
            for key, value in dataset.items()
        }

    # def take_sample_old(self, dataset: Dict[str, torch.Tensor], days=730, basins=100) -> Dict[str, torch.Tensor]:
    #     """Take a sample for a specified time period and number of basins."""
    #     sample = {
    #         key: torch.tensor(
    #             value[self.warm_up:days, :basins, :] if value.ndim == 3 else value[:basins, :],
    #             dtype=torch.float32,
    #             device=self.device
    #         )
    #         for key, value in dataset.items()
    #     }
    #     # Adjust target for warm-up days if necessary
    #     if 'HBV_1_1p' not in self.config['dpl_model']['phy_model']['model'] or not self.config['dpl_model']['phy_model']['warm_up_states']:
    #         sample['target'] = sample['target'][self.warm_up:days, :basins]
    #     return sample
    
def take_sample(config: Dict, dataset_dict: Dict[str, torch.Tensor], days=730,
                basins=100) -> Dict[str, torch.Tensor]:
    """Take sample of data."""
    dataset_sample = {}
    for key, value in dataset_dict.items():
        if value.ndim == 3:
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['dpl_model']['phy_model']['warm_up']
            dataset_sample[key] = torch.tensor(value[warm_up:days, :basins, :]).float().to(config['device'])
        elif value.ndim == 2:
            dataset_sample[key] = torch.tensor(value[:basins, :]).float().to(config['device'])
        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")
    return dataset_sample

    # # Keep 'warmup' days for dHBV1.1p.
    # if ('HBV_1_1p' in config['dpl_model']['phy_model']['model']) and \
    # (config['dpl_model']['phy_model']['warm_up_states']) and (config['multimodel_type'] == 'none'):
    #     pass
    # else:
    #     dataset_sample['target'] = dataset_sample['target'][config['dpl_model']['phy_model']['warm_up']:days, :basins]
    # return dataset_sample
