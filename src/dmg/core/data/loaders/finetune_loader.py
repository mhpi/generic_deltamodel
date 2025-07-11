import logging
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from numpy.typing import NDArray
from dmg.core.data.loaders.base import BaseLoader
from dmg.core.data.loaders.load_nc import NetCDFDataset
from dmg.core.data.data import intersect, extract_temporal_features
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

# Import shared utility functions
from dmg.core.data.loader_utils import *

log = logging.getLogger(__name__)


class FinetuneLoader(BaseLoader):
    """Data loader for fine-tuning with both NetCDF and pickle data sources.
    
    Supports both temporal and spatial testing modes, combining NetCDF data
    for neural network inputs with pickle data for physics model inputs.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    test_split : Optional[bool]
        Whether to split data into training and testing sets
    overwrite : Optional[bool]
        Whether to overwrite existing normalization statistics
    holdout_index : Optional[int]
        Index for spatial holdout testing
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
        
        # Data configuration
        self.nn_attributes = config['delta_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['delta_model']['nn_model'].get('forcings', [])
        self.phy_attributes = config['delta_model']['phy_model'].get('attributes', [])
        self.phy_forcings = config['delta_model']['phy_model'].get('forcings', [])
        self.data_name = config['observations']['name']
        self.all_forcings = config['observations']['all_forcings']
        self.all_attributes = config['observations']['all_attributes']
        self.target = config['train']['target']
        self.log_norm_vars = config['delta_model']['phy_model'].get('use_log_norm', [])
        
        # Device and data type
        self.device = config['device']
        self.dtype = torch.float32
        
        # Initialize datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        
        # NetCDF tool for neural network data
        self.nc_tool = NetCDFDataset()
        
        # Normalization statistics path
        self.out_path = os.path.join(
            config.get('out_path', 'results'),
            'normalization_statistics.json',
        )
        
        # Spatial testing configuration
        self.test = config.get('test', {})
        self.is_spatial_test = (self.test and 
                               self.test.get('type') == 'spatial')
        
        # Set holdout index for spatial testing
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
        if self.is_spatial_test:
            # For spatial testing, load data and split by basin
            train_range = {
                'start': self.config['train']['start_time'],
                'end': self.config['train']['end_time']
            }
            test_range = {
                'start': self.config['test']['start_time'],
                'end': self.config['test']['end_time']
            }
            
            train_data = self._preprocess_data(train_range)
            test_data = self._preprocess_data(test_range)
            
            self.train_dataset, _ = split_by_basin(
                train_data, self.config, self.test, self.holdout_index
            )
            _, self.eval_dataset = split_by_basin(
                test_data, self.config, self.test, self.holdout_index
            )
        else:
            # Standard temporal split
            if self.test_split:
                train_range = {
                    'start': self.config['train']['start_time'],
                    'end': self.config['train']['end_time']
                }
                test_range = {
                    'start': self.config['test']['start_time'],
                    'end': self.config['test']['end_time']
                }
                self.train_dataset = self._preprocess_data(train_range)
                self.eval_dataset = self._preprocess_data(test_range)
            else:
                full_range = {
                    'start': self.config['train']['start_time'],
                    'end': self.config['test']['end_time']
                }
                self.dataset = self._preprocess_data(full_range)

    def _preprocess_data(self, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Read data, preprocess, and return as tensors for models.
        
        Parameters
        ----------
        t_range : Dict[str, str]
            Time range with 'start' and 'end' keys
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of data tensors for running models
        """
        # Load both NetCDF and pickle data
        x_phy, c_phy, x_nn, c_nn, target = self.read_data(t_range)

        #for tft
        start_date = pd.to_datetime(t_range['start'].replace('/', '-'))
        end_date = pd.to_datetime(t_range['end'].replace('/', '-'))
        warmup_days = self.config['delta_model']['phy_model']['warm_up']
        start_date_with_warmup = start_date - pd.Timedelta(days=warmup_days)
        date_range = pd.date_range(start_date_with_warmup, end_date, freq='D')
        temporal_features = extract_temporal_features(date_range)

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
            'x_phy': to_tensor(x_phy, self.device, self.dtype),
            'c_phy': to_tensor(c_phy, self.device, self.dtype),
            'x_nn': to_tensor(x_nn, self.device, self.dtype),
            'c_nn': to_tensor(c_nn, self.device, self.dtype),
            'xc_nn_norm': to_tensor(xc_nn_norm, self.device, self.dtype),
            'temporal_features': to_tensor(temporal_features, self.device, self.dtype),
            'target': to_tensor(target, self.device, self.dtype),
        }
        return dataset

    def read_data(self, t_range: Dict[str, str]) -> Tuple[np.ndarray, ...]:
        """Read data from both NetCDF and pickle sources.
        
        Parameters
        ----------
        t_range : Dict[str, str]
            Time range dictionary
            
        Returns
        -------
        Tuple[np.ndarray, ...]
            Tuple of physics and neural network inputs, and target data
        """
        scope = 'train' if t_range['end'] == self.config['train']['end_time'] else 'test'
        
        # Load neural network data from NetCDF using shared utilities
        nn_data = load_nn_data(
            self.config, scope, t_range, self.nn_forcings,
            self.nn_attributes, [], self.device, self.nc_tool  
        )
        x_nn = nn_data['x_nn']
        c_nn = nn_data['c_nn']
        
        # Load physics model data from pickle
        data_path = self.config['observations']['data_path']
   
        with open(data_path, 'rb') as f:
            forcings, target, attributes = pickle.load(f)
            
        start_date = pd.to_datetime(t_range['start'].replace('/', '-'))
        end_date = pd.to_datetime(t_range['end'].replace('/', '-'))
        
        # ADD WARMUP TO PHYSICS DATA TOO
        warmup_days = self.config['delta_model']['phy_model']['warm_up']
        start_date_with_warmup = start_date - pd.Timedelta(days=warmup_days)
        
        all_dates = pd.date_range(
            self.config['observations']['start_time'].replace('/', '-'),
            self.config['observations']['end_time'].replace('/', '-'),
            freq='D'
        )
        
        # DEBUG: 
        log.debug(f"DEBUG PHYSICS TIME RANGE:")
        log.debug(f"  Requested range: {t_range['start']} to {t_range['end']}")
        log.debug(f"  With warmup: {start_date_with_warmup} to {end_date}")
        log.debug(f"  All dates range: {all_dates[0]} to {all_dates[-1]}")
        log.debug(f"  Total physics data shape: {forcings.shape}")
        
        # Use the warmup-extended start date
        idx_start = all_dates.get_loc(start_date_with_warmup)
        idx_end = all_dates.get_loc(end_date) + 1
        
        log.debug(f"  Calculated indices: {idx_start} to {idx_end} (length: {idx_end - idx_start})")
        
        # Process forcings and target with warmup included
        forcings = np.transpose(forcings[:, idx_start:idx_end], (1, 0, 2))
        target = np.transpose(target[:, idx_start:idx_end], (1, 0, 2))
        
        # Get physics model variable indices
        phy_forc_idx = []
        for name in self.phy_forcings:
            if name not in self.all_forcings:
                raise ValueError(f"Forcing {name} not listed in available forcings.")
            phy_forc_idx.append(self.all_forcings.index(name))
        
        phy_attr_idx = []
        for attr in self.phy_attributes:
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in available attributes")
            phy_attr_idx.append(self.all_attributes.index(attr))
        
        x_phy = forcings[:, :, phy_forc_idx]
        c_phy = attributes[:, phy_attr_idx]
        

        if 'subset_path' in self.config['observations']:
            subset_path = self.config['observations']['subset_path']
            gage_id_path = self.config['observations']['gage_info']
            
            with open(subset_path, 'r') as f:
                selected_basins = json.load(f)
            gage_info = np.load(gage_id_path)
            
            subset_idx = intersect(selected_basins, gage_info)
            
            x_phy = x_phy[:, subset_idx, :]
            c_phy = c_phy[subset_idx, :]
            target = target[:, subset_idx, :]
        
        # Convert flow to mm/day 
        target = flow_conversion(c_nn, target, self.target, self.nn_attributes, self.config)
        
        return x_phy, c_phy, x_nn, c_nn, target

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