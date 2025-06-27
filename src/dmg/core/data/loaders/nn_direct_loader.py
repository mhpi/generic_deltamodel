import logging
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.load_nc import NetCDFDataset
from dmg.core.data.data import intersect
import pandas as pd
from datetime import datetime
from sklearn.exceptions import DataDimensionalityWarning
from numpy.typing import NDArray

# Import shared utility functions
from dmg.core.data.loader_utils import *

log = logging.getLogger(__name__)

class NnDirectLoader(BaseLoader):
    """Data loader for running direct NN comparisons 
    Within the finetuning setup so removing you physics model.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nc_tool = NetCDFDataset()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        
        # Configuration attributes following paste-2 structure
        self.nn_attributes = config['delta_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['delta_model']['nn_model'].get('forcings', [])
        
        # Physics model attributes (empty for direct NN comparison)
        self.phy_attributes = []
        self.phy_forcings = []
        
        # Data configuration
        self.data_name = config['observations'].get('name', '')
        self.forcing_names = config['observations'].get('all_forcings', [])  # renamed from all_forcings
        self.attribute_names = config['observations'].get('all_attributes', [])  # renamed from all_attributes
        self.target = config['train']['target']
        
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
        
        # Spatial testing configuration
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

    def _extract_temporal_features(self, date_range: List[str]) -> np.ndarray:
        """Extract temporal features from date strings.
        
        Parameters
        ----------
        date_range : List[str]
            List of date strings in format 'YYYY-MM-DD'
            
        Returns
        -------
        np.ndarray
            Array of shape [time_steps, 7] containing temporal features:
            [day_of_month, week_of_year, month, quarter, day_of_week, year, special_events]
        """
        temporal_features = np.zeros((len(date_range), 7), dtype=np.float32)
        
        for i, date_str in enumerate(date_range):
            try:
                # Parse the date string
                if isinstance(date_str, str):
                    # Handle different date formats
                    if 'T' in date_str:
                        date_obj = datetime.fromisoformat(date_str.replace('T', ' ').split('.')[0])
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    # If it's already a datetime object
                    date_obj = date_str
                
                # Extract temporal features
                temporal_features[i, 0] = date_obj.day                        # day of month (1-31)
                temporal_features[i, 1] = date_obj.isocalendar()[1]          # week of year (1-53)
                temporal_features[i, 2] = date_obj.month                     # month (1-12)
                temporal_features[i, 3] = (date_obj.month - 1) // 3 + 1     # quarter (1-4)
                temporal_features[i, 4] = date_obj.weekday() + 1             # day of week (1-7, Monday=1)
                temporal_features[i, 5] = date_obj.year                      # year
                temporal_features[i, 6] = 0                                  # special events (can be customized)
                
            except Exception as e:
                log.warning(f"Error parsing date {date_str}: {e}, using defaults")
                # Use defaults for problematic dates
                temporal_features[i, :] = [1, 1, 1, 1, 1, 2020, 0]
        
        return temporal_features

    def load_dataset(self) -> None:
        """Load data into dictionary of nn and physics model input tensors."""
        # Determine time ranges based on configuration
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
            # Use test data time range for testing in all basins            
            train_data = self._preprocess_data('train', train_range)
            test_data = self._preprocess_data('test', test_range)

            # Split each dataset by basin using shared utilities
            self.train_dataset, _ = split_by_basin(
                train_data, self.config, self.test, self.holdout_index
            )
            _, self.eval_dataset = split_by_basin(
                test_data, self.config, self.test, self.holdout_index
            )
            
            # Log the split dataset sizes for verification
            if self.train_dataset and self.eval_dataset:
                for key in self.train_dataset:
                    if torch.is_tensor(self.train_dataset[key]):
                        log.info(f"Train {key} shape: {self.train_dataset[key].shape}")
                for key in self.eval_dataset:
                    if torch.is_tensor(self.eval_dataset[key]):
                        log.info(f"Test {key} shape: {self.eval_dataset[key].shape}")
        else:
            # Standard temporal split
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
        """Read data, preprocess, and return as tensors for models.
        
        Parameters
        ----------
        scope
            Scope of data to read ('train', 'test', or 'all').
        t_range
            Time range dictionary with 'start' and 'end' keys.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of data tensors for running models.
        """
        log.info(f"Starting preprocessing for {scope} data with time range: {t_range}")
        
        try:
            # Read data using the read_data method
            x_phy, c_phy, x_nn, c_nn, target = self.read_data(scope, t_range)
            
            # Normalize nn input data using shared utilities
            self.norm_stats = load_norm_stats(
                self.out_path, self.overwrite, x_nn, c_nn, target,
                self.nn_forcings, self.nn_attributes, self.target,
                self.log_norm_vars, self.config
            )
            xc_nn_norm = normalize_data(
                x_nn, c_nn, self.nn_forcings, self.nn_attributes,
                self.norm_stats, self.log_norm_vars
            )

            # Build data dict of Torch tensors
            dataset = {
                'x_phy': to_tensor(x_phy, self.device, self.dtype),           # [time, stations, 0]
                'c_phy': to_tensor(c_phy, self.device, self.dtype),           # [stations, 0]
                'x_nn': to_tensor(x_nn, self.device, self.dtype),             # [time, stations, features]
                'c_nn': to_tensor(c_nn, self.device, self.dtype),             # [stations, features]
                'xc_nn_norm': to_tensor(xc_nn_norm, self.device, self.dtype), # [time, stations, features]
                'target': to_tensor(target, self.device, self.dtype),         # [time, stations, features]
            }
            return dataset
                
        except Exception as e:
            log.error(f"Error in data preprocessing: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

    def read_data(self, scope: str, t_range: Dict[str, str]) -> tuple[NDArray[np.float32]]:
        """Read data from NetCDF files and extract target variables.
        
        Parameters
        ----------
        scope
            Scope of data to read ('train', 'test', or 'all').
        t_range
            Time range dictionary with 'start' and 'end' keys.

        Returns
        -------
        tuple[NDArray[np.float32]]
            Tuple of (x_phy, c_phy, x_nn, c_nn, target) data arrays.
        """
        try:
            # Load neural network data (including target variables) using shared utilities
            log.info("Loading neural network data...")
            nn_data = load_nn_data(
                self.config, scope, t_range, self.nn_forcings, 
                self.nn_attributes, self.target, self.device, self.nc_tool
            )
            
            log.info(f"Neural network data shapes - x_nn: {nn_data['x_nn'].shape}, c_nn: {nn_data['c_nn'].shape}")
            
            # Extract target data
            target = nn_data.get('target')
            
            if target is None:
                log.error("Target data not found in neural network data")
                raise ValueError("Target data missing from neural network data")
            
            # Handle invalid values in target data
            invalid_mask = (target < -10)  # Detect extreme negative values
            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                invalid_percent = (invalid_count / target.size) * 100
                log.warning(f"Target contains {invalid_count} invalid values ({invalid_percent:.2f}%)")
                target[invalid_mask] = np.nan
                
            nan_count = np.isnan(target).sum()
            if nan_count > 0:
                nan_percent = (nan_count / target.size) * 100
                log.warning(f"Target contains {nan_count} NaN values ({nan_percent:.2f}%)")
            
            # Convert tensors to numpy arrays for consistency
            x_nn = nn_data['x_nn'].cpu().numpy() if torch.is_tensor(nn_data['x_nn']) else nn_data['x_nn']
            c_nn = nn_data['c_nn'].cpu().numpy() if torch.is_tensor(nn_data['c_nn']) else nn_data['c_nn']
            
            # Create empty placeholder arrays for physics data (minimal memory footprint)
            x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32) # [time, stations, 0]
            c_phy = np.zeros((c_nn.shape[0], 0), dtype=np.float32)
            
            return x_phy, c_phy, x_nn, c_nn, target
            
        except Exception as e:
            log.error(f"Error reading data: {str(e)}")
            raise

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