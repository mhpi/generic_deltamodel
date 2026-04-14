from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler


class HydroSampler(BaseSampler):
    """Hydrological data sampler.

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
        self.rho = config['model']['rho']
        self.warmup = config['model'].get('warmup', 0)

    def select_subset(
        self,
        x: torch.Tensor,
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
        warmup: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Select a subset of input tensor.

        Parameters
        ----------
        x
            Input tensor to select from.
        i_grid
            Grid indices for selection.
        i_t
            Time indices for selection (optional).
        c
            Additional context for selection (optional).
        tuple_out
            Whether to return a tuple of (x_tensor, c_tensor) or concatenate them.
        has_grad
            Whether the output tensor requires gradients.
        warmup
            Optional warmup period to override the default.
        device
            Optional device to override the default.

        Returns
        -------
        torch.Tensor
            Selected subset of the input tensor, optionally concatenated with
            context.
        """
        device = device if device is not None else self.device
        warmup = warmup if warmup is not None else self.warmup
        batch_size, nx = len(i_grid), x.shape[-1]

        # Handle time indexing and create an empty tensor for selection
        if i_t is not None:
            x_tensor = torch.zeros(
                [self.rho + warmup, batch_size, nx],
                device=device,
                requires_grad=has_grad,
            )
            for k in range(batch_size):
                x_tensor[:, k : k + 1, :] = x[
                    i_t[k] - warmup : i_t[k] + self.rho,
                    i_grid[k] : i_grid[k] + 1,
                    :,
                ]
        else:
            x_tensor = (
                x[:, i_grid, :].float().to(device)
                if x.ndim == 3
                else x[i_grid, :].float().to(device)
            )

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(device)
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, self.rho + warmup, 1)
            return (
                (x_tensor, c_tensor)
                if tuple_out
                else torch.cat((x_tensor, c_tensor), dim=2)
            )

        return x_tensor

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch."""
        batch_size = self.config['train']['batch_size']
        i_sample, i_t = random_index(
            ngrid_train,
            nt,
            (batch_size, self.rho),
            warmup=self.warmup,
        )

        return {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'c_phy': dataset['c_phy'][i_sample],
            'c_nn': dataset['c_nn'][i_sample],
            'xc_nn_norm': self.select_subset(
                dataset['xc_nn_norm'],
                i_sample,
                i_t,
                has_grad=False,
            ),
            'target': self.select_subset(dataset['target'], i_sample, i_t, warmup=0),
            'batch_sample': i_sample,
        }

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        return {
            key: (value[:, i_s:i_e, :] if value.ndim == 3 else value[i_s:i_e, :]).to(
                dtype=torch.float32,
                device=self.device,
            )
            for key, value in dataset.items()
            if hasattr(value, 'ndim')
        }
