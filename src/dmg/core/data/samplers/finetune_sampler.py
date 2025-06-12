from typing import Dict, Optional

import numpy as np
import torch

from dMG.core.data.data import random_index
from dMG.core.data.samplers.base import BaseSampler

import logging

log = logging.getLogger(__name__)

class FinetuneSampler(BaseSampler):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.warm_up = config['delta_model']['phy_model']['warm_up']
        self.rho = config['delta_model']['rho']
        self.batch_size = config['train']['batch_size']

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
        has_grad: bool = False,
        is_nn_data: bool = False
    ) -> torch.Tensor:
        """
        Select a subset of input tensor with different handling for NN and physics data.
        """
        batch_size, nx = len(i_grid), x.shape[-1]
        rho, warm_up = self.rho, self.warm_up

        # Check actual data dimensions to determine the correct indexing
        # Both physics and NN data appear to be [timesteps, basins, features] in your case
        if len(x.shape) == 3:
            if x.shape[0] > x.shape[1]:  # More timesteps than basins
                # Data is [timesteps, basins, features]
                data_format = "time_basin_feature"
            else:
                # Data is [basins, timesteps, features]
                data_format = "basin_time_feature"
        else:
            data_format = "unknown"

        if data_format == "time_basin_feature":
            # Handle data format [time, basins, features] - same for both physics and NN
            if i_t is not None:
                x_tensor = torch.zeros(
                    [rho + warm_up, batch_size, nx],
                    device=self.device,
                    requires_grad=has_grad
                )
                for k in range(batch_size):
                    # Calculate valid indices
                    start_idx = max(0, i_t[k] - warm_up)
                    end_idx = min(i_t[k] + rho, x.shape[0])
                    valid_length = end_idx - start_idx
                    
                    if valid_length > 0:
                        slice_data = x[start_idx:end_idx, i_grid[k]:i_grid[k] + 1, :]
                        x_tensor[:valid_length, k:k + 1, :] = slice_data
            else:
                x_tensor = x[:, i_grid, :].float().to(self.device)
                
        elif data_format == "basin_time_feature":
            # Handle data format [basins, time, features] - original NN data handling
            if i_t is not None:
                x_tensor = torch.zeros(
                    [batch_size, rho + warm_up, nx],
                    device=self.device,
                    requires_grad=has_grad
                )
                for k in range(batch_size):
                    # Calculate valid indices
                    start_idx = max(0, i_t[k] - warm_up)
                    end_idx = min(i_t[k] + rho, x.shape[1])
                    valid_length = end_idx - start_idx
                    
                    if valid_length > 0:
                        x_tensor[k, :valid_length, :] = x[i_grid[k], start_idx:end_idx, :]
            else:
                x_tensor = x[i_grid].float().to(self.device)
        else:
            raise ValueError(f"Unknown data format for tensor shape: {x.shape}")

        if c is not None:
            c_tensor = torch.from_numpy(c).float().to(self.device)
            if data_format == "time_basin_feature":
                c_tensor = c_tensor[i_grid].unsqueeze(0).repeat(rho + warm_up, 1, 1)
            else:
                c_tensor = c_tensor[i_grid].unsqueeze(1).repeat(1, rho + warm_up, 1)
            return (x_tensor, c_tensor) if tuple_out else torch.cat((x_tensor, c_tensor), dim=-1)

        return x_tensor

    def get_training_sample(
        self,
        dataset: Dict[str, np.ndarray],
        ngrid_train: int,
        nt: int
    ) -> Dict[str, torch.Tensor]:
        """Generate a training batch."""
        # Calculate the maximum valid time index
        min_required = self.warm_up + self.rho
        
        if nt < min_required:
            log.warning(f"Not enough timesteps ({nt}) for warm_up ({self.warm_up}) + rho ({self.rho}). "
                       f"Required: {min_required}. Adjusting rho to fit available data.")
            effective_rho = max(1, nt - self.warm_up - 1)
        else:
            effective_rho = self.rho
        
        # Calculate valid time range for sampling
        max_t = nt - effective_rho
        
        if max_t <= self.warm_up:
            log.error(f"Invalid time range: max_t ({max_t}) <= warm_up ({self.warm_up})")
            max_t = self.warm_up + 1
            effective_rho = 1
        
        # Generate random indices
        i_sample, i_t = random_index(ngrid_train, max_t, (self.batch_size, effective_rho), warm_up=self.warm_up)

        samples = {
            'x_phy': self.select_subset(dataset['x_phy'], i_sample, i_t),
            'target': self.select_subset(dataset['target'], i_sample, i_t)[self.warm_up:, :],
            'batch_sample': i_sample,
        }

        # Handle static features (check if they have actual data)
        if 'c_phy' in dataset and dataset['c_phy'].shape[1] > 0:
            samples['c_phy'] = dataset['c_phy'][i_sample]
        if 'c_nn' in dataset and dataset['c_nn'].shape[1] > 0:
            samples['c_nn'] = dataset['c_nn'][i_sample]

        # Handle NN-specific data
        if 'x_nn' in dataset:
            samples['x_nn'] = self.select_subset(dataset['x_nn'], i_sample, i_t)
        if 'xc_nn_norm' in dataset:
            samples['xc_nn_norm'] = self.select_subset(dataset['xc_nn_norm'], i_sample, i_t)

        return samples
    def get_validation_sample(
            self,
            dataset: Dict[str, torch.Tensor],
            start: int,
            end: int
        ) -> Dict[str, torch.Tensor]:
            """Get a validation/test batch that matches hydro implementation.
            
            Parameters
            ----------
            dataset : Dict[str, torch.Tensor]
                Dataset containing tensors
            start : int
                Start index for basin slice
            end : int
                End index for basin slice
                
            Returns
            -------
            Dict[str, torch.Tensor]
                Batch data dictionary with sliced tensors
            """
            batch = {}
            slice_idx = slice(start, end)
            
            # Process neural network data (dimensions: [basins, time, features])
            for key in ['x_nn', 'xc_nn_norm']:
                if key in dataset:
                    batch[key] = dataset[key][slice_idx].to(self.device)
            
            # Process physics data (dimensions: [time, basins, features])
            if 'x_phy' in dataset:
                batch['x_phy'] = dataset['x_phy'][:, slice_idx].to(self.device)
            
            # Process static features (dimensions: [basins, features])
            for key in ['c_nn', 'c_phy']:
                if key in dataset and dataset[key].numel() > 0:
                    batch[key] = dataset[key][slice_idx].to(self.device)
            
            # Process target (dimensions: [time, basins, features])
            if 'target' in dataset:
                batch['target'] = dataset['target'][:, slice_idx].to(self.device)
            
            return batch

    def get_time_window_sample(
        self, 
        dataset: Dict[str, torch.Tensor], 
        basin_start: int, 
        basin_end: int, 
        time_start: int, 
        time_end: int
    ) -> Dict[str, torch.Tensor]:
        """Get a batch of validation/test data for a specific time window.
        
        Parameters
        ----------
        dataset : Dict[str, torch.Tensor]
            Dataset tensors
        basin_start : int
            Start basin index
        basin_end : int
            End basin index
        time_start : int
            Start time index
        time_end : int
            End time index
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Batch data dictionary
        """
        log.info(f"Creating time window sample for basins {basin_start} to {basin_end-1}, time {time_start} to {time_end-1}")
        
        # Select basins and time window
        batch_size = basin_end - basin_start
        seq_len = time_end - time_start
        
        # Create batch tensors
        batch_data = {}
        
        # Process NN data (dimensions: [basins, time, features])
        if 'xc_nn_norm' in dataset:
            # Ensure we don't exceed bounds
            actual_time_end = min(time_start + seq_len, dataset['xc_nn_norm'].shape[1])
            actual_seq_len = actual_time_end - time_start
            
            batch_data['xc_nn_norm'] = dataset['xc_nn_norm'][basin_start:basin_end, time_start:actual_time_end].to(self.device)
            log.debug(f"Time window xc_nn_norm shape: {batch_data['xc_nn_norm'].shape}")
        
        # Process physics data (dimensions: [time, basins, features])
        if 'x_phy' in dataset:
            # Ensure we don't exceed bounds
            actual_time_end = min(time_start + seq_len, dataset['x_phy'].shape[0])
            actual_seq_len = actual_time_end - time_start
            
            batch_data['x_phy'] = dataset['x_phy'][time_start:actual_time_end, basin_start:basin_end].to(self.device)
            log.debug(f"Time window x_phy shape: {batch_data['x_phy'].shape}")
        
        # Process static attributes
        if 'c_phy' in dataset:
            batch_data['c_phy'] = dataset['c_phy'][basin_start:basin_end].to(self.device)
        
        if 'c_nn' in dataset:
            batch_data['c_nn'] = dataset['c_nn'][basin_start:basin_end].to(self.device)
        
        # Process target data
        if 'target' in dataset:
            # Ensure we don't exceed bounds
            actual_time_end = min(time_start + seq_len, dataset['target'].shape[0])
            actual_seq_len = actual_time_end - time_start
            
            batch_data['target'] = dataset['target'][time_start:actual_time_end, basin_start:basin_end].to(self.device)
        
        return batch_data
        
    def take_sample(
        self,
        dataset: Dict[str, torch.Tensor],
        days: int = 730,
        basins: int = 100
    ) -> Dict[str, torch.Tensor]:
        """Take a sample of data."""
        sample = {}
        
        for key, value in dataset.items():
            if key in ['x_nn', 'xc_nn_norm']:
                # Neural network data: [basins, timesteps, features]
                sample[key] = torch.tensor(
                    value[:basins, :days, :],
                    dtype=torch.float32,
                    device=self.device
                )
            elif value.ndim == 3:
                # Physics data: [time, basins, features]
                warm_up = 0 if key in ['x_phy'] else self.warm_up
                sample[key] = torch.tensor(
                    value[warm_up:days, :basins, :],
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                sample[key] = torch.tensor(
                    value[:basins, :],
                    dtype=torch.float32,
                    device=self.device
                )

        # Handle target data based on model configuration
        if not (
            'HBV1_1p' in self.config['delta_model']['phy_model']['model'] and
            self.config['delta_model']['phy_model']['warm_up_states'] and
            self.config['multimodel_type'] == 'none'
        ):
            sample['target'] = sample['target'][self.config['delta_model']['phy_model']['warm_up']:days, :basins]

        return sample