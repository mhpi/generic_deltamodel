from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.samplers.base import BaseSampler


class MsHydroSampler(BaseSampler):
    """Multiscale hydrological data sampler.

    Constructs training batches by sampling gages and gathering their
    constituent catchement data for multiscale model training.

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
        self.warmup = config['model']['warmup']
        self.rho = config['model']['rho']

    def select_subset(
        self,
        x: NDArray[np.float32],
        i_grid: NDArray[np.float32],
        i_t: Optional[NDArray[np.float32]] = None,
        c: Optional[NDArray[np.float32]] = None,
        tuple_out: bool = False,
        has_grad: bool = False,
        warmup: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Select a subset of input array for gage-level data.

        Handles temporal subsetting with random time indices per sample.

        Parameters
        ----------
        x
            Input data array [nt, nb, nvar] or [nb, nvar].
        i_grid
            Gage indices to select.
        i_t
            Time start indices for each sample.
        c
            Optional static data to concatenate.
        tuple_out
            If True, return a tuple of (x_tensor, c_tensor).
        has_grad
            If True, create tensors with gradient tracking.
        warmup
            Override for the warm-up window length. When None, uses
            self.warmup. Pass 0 for target data which should not
            include the warm-up period (matching the reference
            selectSubset(y, iGrid, iT, rho) call).
        device
            Device to place output tensors on.
        """
        device = device if device is not None else self.device
        warmup = warmup if warmup is not None else self.warmup
        batch_size = len(i_grid)
        nx = x.shape[-1]

        if i_t is not None:
            x_tensor = torch.zeros(
                [self.rho + warmup, batch_size, nx],
                device=device,
                requires_grad=has_grad,
            )
            for k in range(batch_size):
                x_tensor[:, k : k + 1, :] = torch.as_tensor(
                    x[
                        i_t[k] - warmup : i_t[k] + self.rho,
                        i_grid[k] : i_grid[k] + 1,
                        :,
                    ],
                    dtype=torch.float32,
                )
        else:
            if x.ndim == 3:
                x_tensor = torch.as_tensor(x[:, i_grid, :], dtype=torch.float32)
            else:
                x_tensor = torch.as_tensor(x[i_grid, :], dtype=torch.float32)

        if c is not None:
            c_tensor = torch.as_tensor(
                c[i_grid],
                dtype=torch.float32,
            )
            c_tensor = c_tensor.unsqueeze(1).repeat(1, self.rho + warmup, 1)
            if tuple_out:
                return (
                    x_tensor.to(device),
                    c_tensor.to(device),
                )
            return torch.cat(
                (x_tensor, c_tensor),
                dim=2,
            ).to(device)

        return x_tensor.to(device)

    def get_training_sample(
        self,
        dataset: dict[str, NDArray[np.float32]],
        ngrid_train: int,
        nt: int,
    ) -> dict[str, torch.Tensor]:
        """Generate a training batch."""
        raise NotImplementedError(
            "Method not implemented. Multiscale training with sampler will be enabled at a later date.",
        )

    def get_validation_sample(
        self,
        dataset: dict[str, torch.Tensor],
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Generate batch for model forwarding only."""
        dataset_sample = {}
        device = self.config['device']

        for key, value in dataset.items():
            if key in ('x_nn_norm', 'c_nn_norm'):
                continue
            if not hasattr(value, 'dtype') or not np.issubdtype(value.dtype, np.number):
                continue
            if value.ndim == 3:
                if key in ['x_phy']:
                    warmup = 0
                else:
                    warmup = self.config['model']['warmup']
                dataset_sample[key] = torch.tensor(
                    value[warmup:, i_s:i_e, :],
                    dtype=torch.float32,
                    device=device,
                )
            elif value.ndim == 2:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e, :],
                    dtype=torch.float32,
                    device=device,
                )
            elif value.ndim == 1:
                dataset_sample[key] = torch.tensor(
                    value[i_s:i_e],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Incorrect input dimensions. {key} array must have 1, 2 or 3 dimensions.",
                )

        x_nn_batch = torch.tensor(
            dataset['x_nn_norm'][:, i_s:i_e, :],
            dtype=torch.float32,
            device=device,
        )
        c_nn_batch = torch.tensor(
            dataset['c_nn_norm'][i_s:i_e, :],
            dtype=torch.float32,
            device=device,
        )
        c_nn_expanded = c_nn_batch.unsqueeze(0).expand(
            x_nn_batch.shape[0],
            -1,
            -1,
        )
        dataset_sample['xc_nn_norm'] = torch.cat(
            (x_nn_batch, c_nn_expanded),
            dim=-1,
        )
        dataset_sample['c_nn_norm'] = c_nn_batch

        return dataset_sample
