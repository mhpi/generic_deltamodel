from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

from dmg.core.data.data import random_index
from dmg.core.data.samplers.base import BaseSampler


class ModHydroSampler(BaseSampler):
    """Sampler for hydrology experiments.
    Custom additions over the base package sampler:
      - xc_pretrained_norm passthrough (for DirectFinetuneing / NnDualLoader)
      - temporal_features batch construction
      - get_validation_sample for windowed evaluation
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warmup = config['model']['warmup']
        self.rho = config['model']['rho']

    def load_data(self):
        raise NotImplementedError

    def preprocess_data(self):
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
        """Select a [time, batch, features] subset from a dataset tensor."""
        batch_size, nx = len(i_grid), x.shape[-1]

        if i_t is not None:
            input_timesteps = self.rho + self.warmup

            x_tensor = torch.zeros(
                [input_timesteps, batch_size, nx],
                device=self.device,
                requires_grad=has_grad,
            )

            for k in range(batch_size):
                idx = int(i_grid[k])
                start_idx = int(i_t[k]) - self.warmup
                end_idx = start_idx + input_timesteps

                if idx >= x.shape[1]:
                    continue

                if end_idx <= x.shape[0] and start_idx >= 0:
                    x_tensor[:, k, :] = x[start_idx:end_idx, idx, :]
                elif start_idx >= 0 and start_idx < x.shape[0]:
                    # Edge case: pad with last available timestep
                    available = x[start_idx:, idx, :]
                    avail_len = min(available.shape[0], input_timesteps)
                    x_tensor[:avail_len, k, :] = available[:avail_len]
                    if avail_len < input_timesteps:
                        x_tensor[avail_len:, k, :] = (
                            available[-1]
                            .unsqueeze(0)
                            .expand(input_timesteps - avail_len, -1)
                        )
        else:
            x_tensor = x[:, i_grid, :].float().to(self.device)

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            repeat_t = input_timesteps if i_t is not None else self.rho + self.warmup
            c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, repeat_t, 1)
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
        """Generate a random training mini-batch."""
        batch_size = self.config['train']['batch_size']

        i_sample, i_t = random_index(
            ngrid_train,
            nt,
            (batch_size, self.rho),
            warmup=self.warmup,
        )

        targets = self.select_subset(dataset['target'], i_sample, i_t)[self.warmup :, :]

        xc_nn_norm = self.select_subset(
            dataset['xc_nn_norm'], i_sample, i_t, has_grad=False
        )

        batch_data = {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'c_phy': dataset['c_phy'][i_sample],
            'c_nn': dataset['c_nn'][i_sample],
            'xc_nn_norm': xc_nn_norm,
            'target': targets,
            'batch_sample': i_sample,
        }

        # Pretrained encoder stream (NnDualLoader / DirectFinetuneing)
        if dataset.get('xc_pretrained_norm') is not None:
            batch_data['xc_pretrained_norm'] = self.select_subset(
                dataset['xc_pretrained_norm'], i_sample, i_t, has_grad=False
            )

        # Temporal features: [T, K] -> [warmup + rho, batch_size, K]
        if dataset.get('temporal_features') is not None:
            tf = dataset['temporal_features']  # [T, K]
            n_steps = self.warmup + self.rho
            tf_batch = torch.zeros(
                n_steps, batch_size, tf.shape[-1], device=self.device, dtype=tf.dtype
            )
            for k in range(batch_size):
                t_start = int(i_t[k]) - self.warmup
                t_end = t_start + n_steps
                if t_end <= tf.shape[0] and t_start >= 0:
                    tf_batch[:, k, :] = tf[t_start:t_end, :]
            batch_data['temporal_features'] = tf_batch

        return batch_data

    def get_validation_sample(
        self,
        dataset: dict,
        i_s: int,
        i_e: int,
    ) -> dict[str, torch.Tensor]:
        """Return a basin-sliced validation sample (time-major, warmup stripped)."""
        warmup = self.warmup
        i_grid = np.arange(i_s, i_e)

        sample = {
            'xc_nn_norm': dataset['xc_nn_norm'][warmup:, i_grid, :],
            'target': dataset['target'][warmup:, i_grid, :],
            'c_nn': dataset['c_nn'][i_grid],
            'c_phy': dataset['c_phy'][i_grid],
            'temporal_features': dataset['temporal_features'][warmup:, :],
        }

        if dataset.get('xc_pretrained_norm') is not None:
            sample['xc_pretrained_norm'] = dataset['xc_pretrained_norm'][
                warmup:, i_grid, :
            ]

        return {
            k: v.to(dtype=torch.float32, device=self.device)
            if torch.is_tensor(v)
            else v
            for k, v in sample.items()
        }
