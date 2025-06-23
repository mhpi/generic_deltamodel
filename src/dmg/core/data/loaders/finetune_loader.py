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
from dmg.core.data.data import intersect
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

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
            
            self.train_dataset, _ = self._split_by_basin(train_data) 
            _, self.eval_dataset = self._split_by_basin(test_data)
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

        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn, target)
        xc_nn_norm = self.normalize(x_nn, c_nn)

        # Build data dict of Torch tensors
        dataset = {
            'x_phy': self.to_tensor(x_phy),
            'c_phy': self.to_tensor(c_phy),
            'x_nn': self.to_tensor(x_nn),
            'c_nn': self.to_tensor(c_nn),
            'xc_nn_norm': self.to_tensor(xc_nn_norm),
            'target': self.to_tensor(target),
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
        
        # Load neural network data from NetCDF
        nn_data = self._load_nn_data(scope, t_range)
        x_nn = nn_data['x_nn'].cpu().numpy() if torch.is_tensor(nn_data['x_nn']) else nn_data['x_nn']
        c_nn = nn_data['c_nn'].cpu().numpy() if torch.is_tensor(nn_data['c_nn']) else nn_data['c_nn']
        
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
        
        # Extract physics model data
        x_phy = forcings[:, :, phy_forc_idx]
        c_phy = attributes[:, phy_attr_idx]
        
        # Apply subsetting if needed
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
        
        # Convert flow to mm/day if necessary
        target = self._flow_conversion(c_nn, target)
        
        return x_phy, c_phy, x_nn, c_nn, target

    def _load_nn_data(self, scope: str, t_range: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load and process neural network data from NetCDF."""
        time_range = [t_range['start'].replace('/', '-'), t_range['end'].replace('/', '-')]
        warmup_days = self.config['delta_model']['phy_model']['warm_up']
        
        # Load ALL stations first to avoid station ID format mismatch
        # nc2array will automatically extend the time range backwards by warmup_days
        time_series_data, static_data, date_range = self.nc_tool.nc2array(
            self.config['data_path'],
            station_ids=None,  # Load all stations
            time_range=time_range,  # Original time range - nc2array handles warmup extension
            time_series_variables=self.nn_forcings,
            static_variables=self.nn_attributes,
            add_coords=True,
            warmup_days=warmup_days  # Let nc2array handle the warmup period
        )
        
        # Now filter stations if needed
        if 'subset_path' in self.config['observations']:
            subset_path = self.config['observations']['subset_path']
            gage_id_path = self.config['observations']['gage_info']
            
            with open(subset_path, 'r') as f:
                selected_basins = json.load(f)
            gage_info = np.load(gage_id_path)
            
            subset_idx = intersect(selected_basins, gage_info)
            
            # Filter the NetCDF data to match the subset
            time_series_data = time_series_data[subset_idx]
            static_data = static_data[subset_idx]
        
        # Handle coordinates (remove lat/lon from static data)
        if static_data.shape[1] >= 2:
            static_data = static_data[:, :-2]

        # Transform to match HydroLoader format [time, basins, features]
        time_series_data = np.transpose(time_series_data, (1, 0, 2))
        
        return {
            'x_nn': time_series_data.astype(np.float32),  # Now [time, basins, features] with warmup
            'c_nn': static_data.astype(np.float32)        # [basins, features]
        }
        
    def _split_by_basin(self, dataset):
        """Split dataset by basin for spatial testing."""
        if not dataset or not self.is_spatial_test:
            return dataset, dataset
        
        try:
            extent = self.test.get('extent')
            holdout_gages = []
            
            if extent == 'PUR':
                huc_regions = self.test.get('huc_regions', [])
                if self.holdout_index >= len(huc_regions):
                    raise ValueError(f"Invalid holdout index: {self.holdout_index}")
                    
                holdout_hucs = huc_regions[self.holdout_index]
                gage_file = self.config['observations']['gage_split_file']
                gageinfo = pd.read_csv(gage_file, dtype={"huc": int, "gage": str})
                holdout_hucs_int = [int(huc) for huc in holdout_hucs]
                holdout_gages = gageinfo[gageinfo['huc'].isin(holdout_hucs_int)]['gage'].tolist()
                
            elif extent == 'PUB':
                pub_ids = self.test.get('PUB_ids', [])
                if self.holdout_index >= len(pub_ids):
                    raise ValueError(f"Invalid holdout index: {self.holdout_index}")
                    
                holdout_pub = pub_ids[self.holdout_index]
                gage_file = self.config['observations']['gage_split_file']
                gageinfo = pd.read_csv(gage_file, dtype={"PUB_ID": int, "gage": str})
                holdout_gages = gageinfo[gageinfo['PUB_ID'] == holdout_pub]['gage'].tolist()
            
            # Get basin list
            subset_path = self.config['observations']['subset_path']
            with open(subset_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('[') and content.endswith(']'):
                    content = content.strip('[]')
                    all_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                else:
                    all_basins = [line.strip() for line in content.split() if line.strip()]
            
            # Create indices for train/test split
            holdout_gages_set = set(int(str(basin).strip()) for basin in holdout_gages)
            test_indices = []
            train_indices = []
            
            for i, basin in enumerate(all_basins):
                basin_int = int(str(basin).strip())
                if basin_int in holdout_gages_set:
                    test_indices.append(i)
                else:
                    train_indices.append(i)
            
            if not test_indices:
                raise ValueError("No test basins found!")
            
            # Split the dataset
            train_data = {}
            test_data = {}
            train_indices_tensor = torch.tensor(train_indices, device='cpu')
            test_indices_tensor = torch.tensor(test_indices, device='cpu')
            
            for key, tensor in dataset.items():
                if tensor is None:
                    continue
                    
                cpu_tensor = tensor.to('cpu')
                
                if len(cpu_tensor.shape) == 3:
                    if cpu_tensor.shape[0] == len(all_basins):  # [basins, time, features]
                        train_data[key] = cpu_tensor[train_indices_tensor]
                        test_data[key] = cpu_tensor[test_indices_tensor]
                    else:  # [time, basins, features]
                        train_data[key] = cpu_tensor[:, train_indices_tensor]
                        test_data[key] = cpu_tensor[:, test_indices_tensor]
                elif len(cpu_tensor.shape) == 2:  # [basins, features]
                    train_data[key] = cpu_tensor[train_indices_tensor]
                    test_data[key] = cpu_tensor[test_indices_tensor]
                else:
                    train_data[key] = tensor
                    test_data[key] = tensor
                
                # Move back to original device
                train_data[key] = train_data[key].to(tensor.device)
                test_data[key] = test_data[key].to(tensor.device)
            
            return train_data, test_data
            
        except Exception as e:
            log.error(f"Error splitting dataset by basin: {e}")
            return None, None

    def _flow_conversion(self, c_nn: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Convert hydraulic flow from ft3/s to mm/day."""
        for name in ['flow_sim', 'streamflow', 'sf']:
            if name in self.target:
                target_index = self.target.index(name)
                target_temp = target[:, :, target_index].copy()
                
                try:
                    area_name = self.config['observations']['area_name']
                    basin_area = c_nn[:, self.nn_attributes.index(area_name)]
                    area = np.expand_dims(basin_area, axis=0).repeat(target_temp.shape[0], 0)
                    
                    converted_flow = ((10 ** 3) * target_temp * 0.0283168 * 3600 * 24 / 
                                     (area * (10 ** 6)))
                    target[:, :, target_index] = converted_flow
                    
                except (KeyError, ValueError) as e:
                    log.warning(f"Could not convert flow units: {e}")
        
        return target

    def load_norm_stats(self, x_nn: np.ndarray, c_nn: np.ndarray, target: np.ndarray) -> None:
        """Load or calculate normalization statistics if necessary."""
        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                try:
                    with open(self.out_path, 'r') as f:
                        self.norm_stats = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    log.warning(f"Could not load norm stats: {e}")
                    self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)
        else:
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)

    def _init_norm_stats(self, x_nn: np.ndarray, c_nn: np.ndarray, target: np.ndarray) -> Dict[str, List[float]]:
        """Compile and save calculations of data normalization statistics."""
        stat_dict = {}
        
        # Get basin areas from attributes
        basin_area = self._get_basin_area(c_nn)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            try:
                if var in self.log_norm_vars:
                    stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
                else:
                    stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])
            except Exception as e:
                log.warning(f"Error calculating stats for {var}: {e}")
                stat_dict[var] = [0, 1, 0, 1]

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            try:
                stat_dict[var] = self._calc_norm_stats(c_nn[:, k])
            except Exception as e:
                log.warning(f"Error calculating stats for {var}: {e}")
                stat_dict[var] = [0, 1, 0, 1]

        # Target variable stats
        for i, name in enumerate(self.target):
            try:
                if name in ['flow_sim', 'streamflow', 'sf']:
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0).copy(),
                        basin_area,
                    )
                else:
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0),
                    )
            except Exception as e:
                log.warning(f"Error calculating stats for {name}: {e}")
                stat_dict[name] = [0, 1, 0, 1]

        # Save statistics to file
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, 'w') as f:
                json.dump(stat_dict, f, indent=4)
        except Exception as e:
            log.warning(f"Could not save norm stats: {e}")
        
        return stat_dict

    def _calc_norm_stats(self, x: np.ndarray, basin_area: np.ndarray = None) -> List[float]:
        """Calculate statistics for normalization with optional basin area adjustment."""
        # Handle invalid values
        x = x.copy()
        x[x == -999] = np.nan
        if basin_area is not None:
            x[x < 0] = 0

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if nd == 3 and x.shape[2] == 1:
                x = x[:, :, 0]
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
            x = flow

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            b = np.array([0])

        # Calculate statistics
        transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x: np.ndarray) -> List[float]:
        """Calculate gamma statistics for streamflow and precipitation data."""
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        
        if b.size == 0:
            return [0, 1, 0, 1]
        
        b = np.log10(np.sqrt(b) + 0.1)
        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _get_basin_area(self, c_nn: np.ndarray) -> Optional[np.ndarray]:
        """Get basin area from attributes."""
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
            return basin_area
        except (KeyError, ValueError) as e:
            log.warning(f"No area information found: {e}. Basin area norm will not be applied.")
            return None

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network"""
        x_nn_norm = self._to_norm(
            np.swapaxes(x_nn, 1, 0).copy(),  # [time, basins, features] -> [basins, time, features]
            self.nn_forcings,
        )
        c_nn_norm = self._to_norm(
            c_nn,  # [basins, features]
            self.nn_attributes,
        )

        # Remove nans
        x_nn_norm[x_nn_norm != x_nn_norm] = 0
        c_nn_norm[c_nn_norm != c_nn_norm] = 0

        c_nn_norm = np.repeat(
            np.expand_dims(c_nn_norm, 0),
            x_nn_norm.shape[0],
            axis=0
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
        del x_nn_norm, c_nn_norm

        return xc_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization - IDENTICAL to HydroLoader."""
        if not self.norm_stats:
            log.warning("No normalization statistics available, using identity normalization")
            return data
            
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            if var not in self.norm_stats:
                log.warning(f"No normalization stats for {var}, skipping")
                continue
                
            stat = self.norm_stats[var]

            if len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(np.maximum(data[:, :, k], 0)) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(np.maximum(data[:, k], 0)) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        # NOTE: Should be external, except altering order of first two dims
        # augments normalization...
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)  # Back to [time, basins, features]

    def _from_norm(
            self,
            data_scaled: NDArray[np.float32],
            vars: list[str],
        ) -> NDArray[np.float32]:
        """De-normalize data.
        
        Parameters
        ----------
        data
            Data to de-normalize.
        vars
            List of variable names in data to de-normalize.
        
        Returns
        -------
        NDArray[np.float32]
            De-normalized data.
        """
        data = np.zeros(data_scaled.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if len(data_scaled.shape) == 3:
                data[:, :, k] = data_scaled[:, :, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
            elif len(data_scaled.shape) == 2:
                data[:, k] = data_scaled[:, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)
