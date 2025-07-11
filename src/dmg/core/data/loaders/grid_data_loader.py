import logging
import os
import json
import pickle
import numpy as np
import torch
import xarray as xr
from typing import Dict, Any, Optional, Tuple, List
from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.data import intersect, extract_temporal_features
import pandas as pd
from numpy.typing import NDArray

# Import shared utility functions from your existing system
from dmg.core.data.loader_utils import *

log = logging.getLogger(__name__)

class GridDataLoader(BaseLoader):
    """
    Data loader for gridded NetCDF data (like fire occurrence).
    
    This follows the exact same pattern as NnDirectLoader but reads from NetCDF
    instead of pickle files. It produces the same output structure so the rest
    of the system works unchanged.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        
        # Configuration attributes following the exact same structure as NnDirectLoader
        self.nn_attributes = config['delta_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['delta_model']['nn_model'].get('forcings', [])
        
        # Physics model attributes (empty for direct NN comparison)
        self.phy_attributes = []
        self.phy_forcings = []
        
        # Data configuration
        self.data_name = config['observations'].get('name', '')
        self.forcing_names = config['observations'].get('all_forcings', [])
        self.attribute_names = config['observations'].get('all_attributes', [])
        self.target = config['train']['target']
        
        # NetCDF data path
        self.data_path = config.get('data_path')
        
        # Normalization configuration
        self.log_norm_vars = config['delta_model']['phy_model'].get('use_log_norm', [])

        # Device and data type configuration
        self.device = config['device']
        self.dtype = torch.float32
        
        # Dataset containers
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        
        # Temporal features storage
        self.temporal_features = None
        
        # Output path for normalization statistics
        self.out_path = os.path.join(
            config.get('out_path', 'results'),
            'normalization_statistics.json',
        )
        
        # Spatial testing configuration (same as original)
        self.test = config.get('test', {})
        self.is_spatial_test = (self.test and 
                            self.test.get('type') == 'spatial')
        if holdout_index is not None:
            self.holdout_index = holdout_index
        elif self.is_spatial_test and 'current_holdout_index' in self.test:
            self.holdout_index = self.test['current_holdout_index']
        elif self.is_spatial_test and self.test.get('holdout_indexs'):
            self.holdout_index = self.test['holdout_indexs'][0]
        else:
            self.holdout_index = None
        
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load data into dictionary of nn and physics model input tensors."""
        # Determine time ranges based on configuration (same as original)
        train_range = {
            'start': self.config['train']['start_time'],
            'end': self.config['train']['end_time']
        }
        test_range = {
            'start': self.config['test']['start_time'],
            'end': self.config['test']['end_time']
        }
        
        if self.is_spatial_test:
            # For spatial testing, load and preprocess one dataset
            train_data = self._preprocess_data('train', train_range)
            test_data = self._preprocess_data('test', test_range)

            # Split each dataset by grid cells using shared utilities
            self.train_dataset, _ = split_by_basin(
                train_data, self.config, self.test, self.holdout_index
            )
            _, self.eval_dataset = split_by_basin(
                test_data, self.config, self.test, self.holdout_index
            )
            
        else:
            # Standard temporal split (same as original)
            if self.test_split:
                self.train_dataset = self._preprocess_data('train', train_range)
                self.eval_dataset = self._preprocess_data('test', test_range)
            else:
                full_range = {
                    'start': self.config['train']['start_time'],
                    'end': self.config['test']['end_time']
                }
                self.dataset = self._preprocess_data('all', full_range)
                
    def _preprocess_data(
        self,
        scope: str,
        t_range: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """
        Read data, preprocess, and return as tensors for models.
        
        This produces the exact same output structure as NnDirectLoader._preprocess_data
        so the rest of the system works unchanged.
        """
        log.info(f"Starting preprocessing for {scope} data with time range: {t_range}")
        
        try:
            # Read data using the NetCDF read method
            x_phy, c_phy, x_nn, c_nn, target = self.read_netcdf_data(scope, t_range)

            # Extract temporal features (same as original)
            start_date = pd.to_datetime(t_range['start'].replace('/', '-'))
            end_date = pd.to_datetime(t_range['end'].replace('/', '-'))
            warmup_days = self.config['delta_model']['phy_model']['warm_up']
            start_date_with_warmup = start_date - pd.Timedelta(days=warmup_days)
            date_range = pd.date_range(start_date_with_warmup, end_date, freq='D')
            temporal_features = extract_temporal_features(date_range)

            # Normalize nn input data using shared utilities (same as original)
            self.norm_stats = load_norm_stats(
                self.out_path, self.overwrite, x_nn, c_nn, target,
                self.nn_forcings, self.nn_attributes, self.target,
                self.log_norm_vars, self.config
            )
            xc_nn_norm = normalize_data(
                x_nn, c_nn, self.nn_forcings, self.nn_attributes,
                self.norm_stats, self.log_norm_vars
            )

            # Build data dict of Torch tensors (exact same structure as original)
            dataset = {
                'x_phy': to_tensor(x_phy, self.device, self.dtype),           # [time, grid_cells, 0]
                'c_phy': to_tensor(c_phy, self.device, self.dtype),           # [grid_cells, 0]
                'x_nn': to_tensor(x_nn, self.device, self.dtype),             # [time, grid_cells, features]
                'c_nn': to_tensor(c_nn, self.device, self.dtype),             # [grid_cells, features]
                'xc_nn_norm': to_tensor(xc_nn_norm, self.device, self.dtype),
                'temporal_features': to_tensor(temporal_features, self.device, self.dtype),
                'target': to_tensor(target, self.device, self.dtype),         # [time, grid_cells, features]
            }
            return dataset
                
        except Exception as e:
            log.error(f"Error in data preprocessing: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

    def read_netcdf_data(self, scope: str, t_range: Dict[str, str]) -> tuple[NDArray[np.float32]]:
        """
        Read gridded data from NetCDF file.
        
        This replaces the original read_data method but produces the same output format:
        (x_phy, c_phy, x_nn, c_nn, target)
        """
        try:
            log.info(f"Loading NetCDF data from {self.data_path}")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"NetCDF data file not found: {self.data_path}")
            
            # Load the NetCDF dataset
            with xr.open_dataset(self.data_path) as ds:
                log.info(f"Available variables: {list(ds.data_vars.keys())}")
                log.info(f"Dataset dimensions: {dict(ds.dims)}")
                
                # Convert time range to datetime and select time subset
                start_date = pd.to_datetime(t_range['start'].replace('/', '-'))
                end_date = pd.to_datetime(t_range['end'].replace('/', '-'))
                
                # Handle time coordinate conversion with better error handling
                if 'time' in ds.coords:
                    time_coord = ds['time']
                    log.info(f"Time coordinate info: dtype={time_coord.dtype}, shape={time_coord.shape}")
                    log.info(f"Sample time values: {time_coord.values[:5]}")
                    
                    # Try direct pandas conversion first
                    try:
                        time_values = pd.to_datetime(time_coord.values)
                        log.info(f"Successfully converted time using direct pandas conversion")
                    except:
                        # If that fails, try to detect the time format
                        if hasattr(time_coord.values[0], 'item'):
                            sample_val = time_coord.values[0].item()
                            log.info(f"Numeric time detected, sample value: {sample_val}")
                            
                            # Check if values are nanoseconds since epoch
                            if sample_val > 1e15:
                                log.info("Converting from nanoseconds since epoch")
                                time_values = pd.to_datetime(time_coord.values, unit='ns')
                            # Check if values are seconds since epoch
                            elif sample_val > 1e9:
                                log.info("Converting from seconds since epoch")
                                time_values = pd.to_datetime(time_coord.values, unit='s')
                            # Check if values are days since a reasonable epoch
                            elif sample_val < 50000:  # Less than ~137 years
                                # Try different base dates
                                base_candidates = ['1900-01-01', '1970-01-01', '1992-01-01', '2000-01-01']
                                time_values = None
                                
                                for base_date_str in base_candidates:
                                    try:
                                        base_date = pd.to_datetime(base_date_str)
                                        test_date = base_date + pd.Timedelta(days=int(sample_val))
                                        # Check if result is reasonable (between 1990-2030)
                                        if 1990 <= test_date.year <= 2030:
                                            log.info(f"Using base date {base_date_str}")
                                            time_values = [base_date + pd.Timedelta(days=int(t.item())) 
                                                         for t in time_coord.values]
                                            break
                                    except:
                                        continue
                                
                                if time_values is None:
                                    raise ValueError(f"Could not determine time format for values like {sample_val}")
                            else:
                                raise ValueError(f"Unrecognized numeric time format: {sample_val}")
                        else:
                            raise ValueError("Could not convert time coordinate")
                    
                    # Convert to datetime list if it's not already
                    if not isinstance(time_values, list):
                        time_values = time_values.tolist()
                    
                    # Select time range
                    time_mask = [(t >= start_date) and (t <= end_date) for t in time_values]
                    time_indices = np.where(time_mask)[0]
                    
                    if len(time_indices) == 0:
                        raise ValueError(f"No data found in time range {start_date} to {end_date}")
                    
                    ds_subset = ds.isel(time=time_indices)
                else:
                    ds_subset = ds
                    log.warning("No time dimension found, using all data")
                
                # Extract land mask and convert grid to "stations" (valid land cells)
                land_mask = ds_subset['land_mask'].values.astype(bool)
                valid_cells = np.where(land_mask.ravel())[0]
                n_valid_cells = len(valid_cells)
                
                log.info(f"Total grid cells: {land_mask.size}")
                log.info(f"Valid land cells: {n_valid_cells}")
                
                # Extract target variable
                target_var = self.target[0]
                if target_var not in ds_subset.data_vars:
                    raise ValueError(f"Target variable '{target_var}' not found in dataset")
                
                target_data = ds_subset[target_var].values
                
                # Reshape and select valid cells: [time, y, x] -> [time, valid_cells]
                if target_data.ndim == 3:
                    target_reshaped = target_data.reshape(target_data.shape[0], -1)
                    target = target_reshaped[:, valid_cells]
                else:
                    raise ValueError(f"Unexpected target data shape: {target_data.shape}")
                
                # Add feature dimension to match expected format [time, cells, 1]
                target = np.expand_dims(target, axis=2)
                
                # Extract forcing variables (time-varying) - only use what exists
                x_nn_list = []
                available_forcings = []
                for var_name in self.nn_forcings:
                    if var_name in ds_subset.data_vars:
                        var_data = ds_subset[var_name].values
                        if var_data.ndim == 3:  # [time, y, x]
                            var_reshaped = var_data.reshape(var_data.shape[0], -1)
                            var_valid = var_reshaped[:, valid_cells]
                            x_nn_list.append(var_valid)
                            available_forcings.append(var_name)
                            log.info(f"  Added forcing variable: {var_name}")
                        else:
                            log.warning(f"Skipping forcing variable {var_name} with shape: {var_data.shape}")
                    else:
                        log.warning(f"Forcing variable {var_name} not found in dataset")
                
                # Update the actual forcings list to match what we loaded
                self.nn_forcings = available_forcings
                
                if x_nn_list:
                    x_nn = np.stack(x_nn_list, axis=2)  # [time, cells, features]
                else:
                    x_nn = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
                
                # Extract static attributes (time-invariant) - only use what exists
                c_nn_list = []
                available_attributes = []
                for var_name in self.nn_attributes:
                    if var_name in ['lat', 'latitude']:
                        coord_data = ds_subset['latitude'].values.ravel()[valid_cells]
                        c_nn_list.append(coord_data)
                        available_attributes.append('lat')
                        log.info(f"  Added attribute variable: lat")
                    elif var_name in ['lon', 'longitude']:
                        coord_data = ds_subset['longitude'].values.ravel()[valid_cells]
                        c_nn_list.append(coord_data)
                        available_attributes.append('lon')
                        log.info(f"  Added attribute variable: lon")
                    elif var_name in ds_subset.data_vars:
                        var_data = ds_subset[var_name].values
                        if var_data.ndim == 2:  # [y, x]
                            var_reshaped = var_data.ravel()
                            var_valid = var_reshaped[valid_cells]
                            c_nn_list.append(var_valid)
                            available_attributes.append(var_name)
                            log.info(f"  Added attribute variable: {var_name}")
                        else:
                            log.warning(f"Skipping attribute {var_name} with shape: {var_data.shape}")
                    else:
                        log.warning(f"Attribute variable {var_name} not found in dataset")
                
                # Update the actual attributes list to match what we loaded
                self.nn_attributes = available_attributes
                
                if c_nn_list:
                    c_nn = np.stack(c_nn_list, axis=1)  # [cells, features]
                else:
                    # Minimal attributes with lat/lon
                    lat_data = ds_subset['latitude'].values.ravel()[valid_cells]
                    lon_data = ds_subset['longitude'].values.ravel()[valid_cells]
                    c_nn = np.stack([lat_data, lon_data], axis=1)
                    self.nn_attributes = ['lat', 'lon']
                
                # Handle invalid values in target
                invalid_mask = (target < 0) | (target > 1)
                if invalid_mask.any():
                    log.warning(f"Target contains {invalid_mask.sum()} invalid values")
                    target[invalid_mask] = np.nan
                
                # Create empty physics model arrays (same as original NnDirectLoader)
                x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
                c_phy = np.zeros((c_nn.shape[0], 0), dtype=np.float32)
                
                log.info(f"NetCDF data shapes - x_nn: {x_nn.shape}, c_nn: {c_nn.shape}, target: {target.shape}")
                
                return x_phy, c_phy, x_nn, c_nn, target
                
        except Exception as e:
            log.error(f"Error reading NetCDF data: {str(e)}")
            raise

    # Keep the same normalize and _from_norm methods as original
    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network using shared utilities."""
        return normalize_data(
            x_nn, c_nn, self.nn_forcings, self.nn_attributes,
            self.norm_stats, self.log_norm_vars
        )

    def _from_norm(
            self,
            data_scaled: NDArray[np.float32],
            vars: list[str],
        ) -> NDArray[np.float32]:
        """De-normalize data using shared utilities."""
        return from_norm(
            data_scaled, vars, self.norm_stats, self.log_norm_vars
        )