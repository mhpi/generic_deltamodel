import logging
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from core.data.data_loaders.base import BaseDataLoader
from core.data.data_loaders.load_nc import NetCDFDataset
from core.utils.transform import cal_statistics
from core.data import intersect
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning
from numpy.typing import NDArray

log = logging.getLogger(__name__)

class NNDirectDataLoader(BaseDataLoader):
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
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['dpl_model']['nn_model'].get('forcings', [])
        
        # Physics model attributes (empty for direct NN comparison)
        self.phy_attributes = []
        self.phy_forcings = []
        
        # Data configuration
        self.data_name = config['observations'].get('name', '')
        self.forcing_names = config['observations'].get('forcings_all', [])  # renamed from all_forcings
        self.attribute_names = config['observations'].get('attributes_all', [])  # renamed from all_attributes
        self.target = config['train']['target']
        
        # Normalization configuration
        self.log_norm_vars = config['dpl_model']['phy_model'].get('use_log_norm', [])

        # Device and data type configuration
        self.device = config['device']
        self.dtype = torch.float32
        
        # Dataset containers
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        
        # Output path for normalization statistics
        self.out_path = os.path.join(
            config.get('out_path', 'results'),
            'normalization_statistics.json',
        )
        
        # Spatial testing configuration
        self.test_mode = config.get('test_mode', {})
        self.is_spatial_test = (self.test_mode and 
                            self.test_mode.get('type') == 'spatial')
        if holdout_index is not None:
            self.holdout_index = holdout_index
        elif self.is_spatial_test and 'current_holdout_index' in self.test_mode:
            self.holdout_index = self.test_mode['current_holdout_index']
        elif self.is_spatial_test and self.test_mode.get('holdout_indexs'):
            self.holdout_index = self.test_mode['holdout_indexs'][0]
        else:
            self.holdout_index = None
        
        self.load_dataset()

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
            full_data = self._preprocess_data('test', test_range)
            
            # Split into train and test by station ID
            self.train_dataset, self.eval_dataset = self._split_by_basin(full_data)
            
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
            # Load neural network data (including target variables)
            log.info("Loading neural network data...")
            nn_data = self._load_nn_data(scope, t_range)
            log.info(f"Neural network data shapes - x_nn: {nn_data['x_nn'].shape}, c_nn: {nn_data['c_nn'].shape}")
            
            # Extract target data
            target_data = nn_data.get('target')
            
            if target_data is None:
                log.error("Target data not found in neural network data")
                raise ValueError("Target data missing from neural network data")
            
            # Convert target to proper format - from [stations, time, features] to [time, stations, features]
            target = np.transpose(target_data, (1, 0, 2))
            log.info(f"Target shape after transpose: {target.shape}")
            
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
            
            # Transpose x_nn to match expected format [time, stations, features]
            x_nn = np.transpose(x_nn, (1, 0, 2))
            
            # Create empty placeholder arrays for physics data (minimal memory footprint)
            x_phy = np.zeros((target.shape[0], target.shape[1], 0), dtype=np.float32)
            c_phy = np.zeros((c_nn.shape[0], 0), dtype=np.float32)
            
            return x_phy, c_phy, x_nn, c_nn, target
            
        except Exception as e:
            log.error(f"Error reading data: {str(e)}")
            raise

    def _split_by_basin(self, dataset):
        """Split dataset by HUC regions or PUB IDs with robust error handling."""
        if not dataset:
            return None, None
        
        try:
            extent = self.test_mode.get('extent')
            holdout_gages = []
            
            if extent == 'PUR':
                # Get the HUC regions to hold out
                huc_regions = self.test_mode.get('huc_regions', [])
                if not huc_regions or self.holdout_index >= len(huc_regions):
                    log.warning(f"Invalid holdout index: {self.holdout_index}")
                    return None, None
                    
                # Get the specific HUC regions for this holdout index
                holdout_hucs = huc_regions[self.holdout_index]
                log.info(f"Holding out basins from HUC regions: {holdout_hucs}")
                
                # Load the gage info file with HUC mappings
                gage_file = self.test_mode.get('gage_split_file')
                if not os.path.exists(gage_file):
                    log.error(f"Gage file not found: {gage_file}")
                    return None, None

                # Read the gage info CSV
                gageinfo = pd.read_csv(gage_file, dtype={"huc": int, "gage": str})
                
                # Get the basin IDs for the holdout HUCs
                holdout_hucs_int = [int(huc) for huc in holdout_hucs]
                holdout_gages = gageinfo[gageinfo['huc'].isin(holdout_hucs_int)]['gage'].tolist()
                
                log.info(f"Found {len(holdout_gages)} holdout basins from HUC regions {holdout_hucs}")
                
            elif extent == 'PUB':
                # Get the PUB IDs to hold out
                pub_ids = self.test_mode.get('PUB_ids', [])
                if not pub_ids or self.holdout_index >= len(pub_ids):
                    log.warning(f"Invalid holdout index: {self.holdout_index}")
                    return None, None
                    
                # Get the specific PUB ID for this holdout index
                holdout_pub = pub_ids[self.holdout_index]
                log.info(f"Holding out basins from PUB ID: {holdout_pub}")
                
                # Load the gage info file with PUB mappings
                gage_file = self.test_mode.get('gage_split_file')
                if not os.path.exists(gage_file):
                    log.error(f"Gage file not found: {gage_file}")
                    return None, None
                    
                # Read the gage info CSV
                gageinfo = pd.read_csv(gage_file, dtype={"PUB_ID": int, "gage": str})
                
                # Get the basin IDs for the holdout PUB ID
                holdout_gages = gageinfo[gageinfo['PUB_ID'] == holdout_pub]['gage'].tolist()
                
                log.info(f"Found {len(holdout_gages)} holdout basins from PUB ID {holdout_pub}")
                
            else:
                log.error(f"Unknown extent: {extent}")
                return None, None
            
            # Check if subset path exists, if not we'll use all basins
            subset_path = self.config['observations'].get('subset_path')
            all_basins = []
            
            if subset_path and os.path.exists(subset_path):
                log.info(f"Using subset file: {subset_path}")
                with open(subset_path, 'r') as f:
                    content = f.read().strip()
                    # Handle Python list format
                    if content.startswith('[') and content.endswith(']'):
                        content = content.strip('[]')
                        all_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                    else:
                        all_basins = [line.strip() for line in content.split() if line.strip()]
                
                log.info(f"Parsed {len(all_basins)} basins from subset file")
            else:
                # If no subset file, determine basins from the dataset dimensions
                # Use the first 3D tensor's first dimension to determine basin count
                for key, tensor in dataset.items():
                    if torch.is_tensor(tensor) and len(tensor.shape) == 3:
                        if tensor.shape[0] > 0:  # Check if it's shaped as [basins, time, features]
                            all_basins = [str(i) for i in range(tensor.shape[0])]
                            log.info(f"Using all {len(all_basins)} basins from dataset")
                            break
                
                if not all_basins:
                    log.error("Could not determine basin count from dataset")
                    return None, None
            
            # Convert holdout gages to integers for matching
            holdout_gages_int = set()
            for basin in holdout_gages:
                basin_str = str(basin).strip()
                holdout_gages_int.add(int(basin_str))

            # Determine train and test indices
            test_indices = []
            train_indices = []
            for i, basin in enumerate(all_basins):
                try:
                    basin_int = int(str(basin).strip())
                    if basin_int in holdout_gages_int:
                        test_indices.append(i)
                    else:
                        train_indices.append(i)
                except ValueError:
                    log.warning(f"Could not convert basin ID to integer: {basin}")
                    train_indices.append(i)  # Default to training set
            
            # Verify we have test basins
            if not test_indices:
                raise ValueError("No test basins found! Check your region settings and basin IDs.")
                
            # Now split the dataset using these indices
            train_data = {}
            test_data = {}
            
            # Create index tensors
            train_indices_tensor = torch.tensor(train_indices, device='cpu')
            test_indices_tensor = torch.tensor(test_indices, device='cpu')
            
            for key, tensor in dataset.items():
                if tensor is None:
                    continue
                    
                # Move tensor to CPU for safe indexing
                cpu_tensor = tensor.to('cpu')
                
                # Handle different tensor shapes
                if len(cpu_tensor.shape) == 3:
                    if cpu_tensor.shape[0] == len(all_basins):  # [basins, time, features]
                        train_data[key] = cpu_tensor[train_indices_tensor]
                        test_data[key] = cpu_tensor[test_indices_tensor]
                    else:  # [time, basins, features] for target
                        train_data[key] = cpu_tensor[:, train_indices_tensor]
                        test_data[key] = cpu_tensor[:, test_indices_tensor]
                elif len(cpu_tensor.shape) == 2:  # [basins, features]
                    train_data[key] = cpu_tensor[train_indices_tensor]
                    test_data[key] = cpu_tensor[test_indices_tensor]
                else:
                    # Just copy for unusual shapes
                    train_data[key] = tensor
                    test_data[key] = tensor
                
                # Move back to original device
                train_data[key] = train_data[key].to(tensor.device)
                test_data[key] = test_data[key].to(tensor.device)
            
            return train_data, test_data
        
        except Exception as e:
            log.error(f"Error splitting dataset by basin: {e}")
            import traceback
            log.error(traceback.format_exc())
            return None, None

    def load_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> None:
        """Load or calculate normalization statistics if necessary."""
        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                try:
                    with open(self.out_path, 'r') as f:
                        self.norm_stats = json.load(f)
                    log.info(f"Loaded normalization statistics from {self.out_path}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    log.warning(f"Could not load norm stats: {e}")
                    self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)

    def _init_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> Dict[str, List[float]]:
        """Compile and save calculations of data normalization statistics.
        
        Parameters
        ----------
        x_nn
            Neural network dynamic data.
        c_nn
            Neural network static data.
        target
            Target variable data.
        
        Returns
        -------
        Dict[str, List[float]]
            Dictionary of normalization statistics for each variable.
        """
        stat_dict = {}
        
        # Get basin areas from attributes (if available)
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
                stat_dict[var] = [0, 1, 0, 1]  # Default values

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            try:
                stat_dict[var] = self._calc_norm_stats(c_nn[:, k])
            except Exception as e:
                log.warning(f"Error calculating stats for {var}: {e}")
                stat_dict[var] = [0, 1, 0, 1]  # Default values

        # Target variable stats
        for i, name in enumerate(self.target):
            try:
                # Special handling for different target types
                if name in ['flow_sim', 'streamflow', 'sf']:
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0).copy(),
                        basin_area,
                    )
                else:
                    # For soil_moisture and other targets - no basin area normalization needed
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0),
                    )
            except Exception as e:
                log.warning(f"Error calculating stats for {name}: {e}")
                stat_dict[name] = [0, 1, 0, 1]  # Default values

        # Save statistics to file
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, 'w') as f:
                json.dump(stat_dict, f, indent=4)
            log.info(f"Saved normalization statistics to {self.out_path}")
        except Exception as e:
            log.warning(f"Could not save norm stats: {e}")
        
        return stat_dict

    def _calc_norm_stats(
        self,
        x: NDArray[np.float32],
        basin_area: NDArray[np.float32] = None,
    ) -> List[float]:
        """Calculate statistics for normalization with optional basin area adjustment.

        Parameters
        ----------
        x
            Input data array.
        basin_area
            Basin area array for normalization.
        
        Returns
        -------
        List[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        # Handle invalid values
        x = x.copy()  # Create a copy to avoid modifying original data
        x[x == -999] = np.nan
        
        # For soil_moisture and similar targets, negative values are likely invalid
        if basin_area is None:
            x[x < 0] = 0
        else:
            x[x < 0] = 0  # Specific to basin normalization

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if nd == 3 and x.shape[2] == 1:
                x = x[:, :, 0]  # Unsqueeze the original 3D matrix
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
            x = flow  # Replace x with flow for further calculations

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            log.warning("No valid values for statistics calculation, using defaults")
            b = np.array([0])

        # Calculate statistics
        if basin_area is not None:
            transformed = np.log10(np.sqrt(b) + 0.1)
        else:
            transformed = b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _calc_gamma_stats(self, x: NDArray[np.float32]) -> List[float]:
        """Calculate gamma statistics for streamflow and precipitation data.
        
        Parameters
        ----------
        x
            Input data array.
        
        Returns
        -------
        List[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        
        if b.size == 0:
            log.warning("No valid values for gamma statistics calculation, using defaults")
            return [0, 1, 0, 1]
        
        b = np.log10(np.sqrt(b) + 0.1)

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _get_basin_area(self, c_nn: NDArray[np.float32]) -> Optional[NDArray[np.float32]]:
        """Get basin area from attributes if available.
        
        Parameters
        ----------
        c_nn
            Neural network static data.
        
        Returns
        -------
        Optional[NDArray[np.float32]]
            1D array of basin areas (2nd dummy dim added for calculations) or None.
        """
        try:
            area_name = self.config['observations'].get('area_name')
            if area_name and area_name in self.nn_attributes:
                basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
                return basin_area
            return None
        except (KeyError, ValueError) as e:
            log.info(f"No area information found: {e}. Basin area norm will not be applied.")
            return None

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Normalize data for neural network.
        
        Parameters
        ----------
        x_nn
            Neural network dynamic data.
        c_nn
            Neural network static data.
        
        Returns
        -------
        NDArray[np.float32]
            Normalized x_nn and c_nn concatenated together.
        """
        x_nn_norm = self._to_norm(
            np.swapaxes(x_nn, 1, 0).copy(),
            self.nn_forcings,
        )
        c_nn_norm = self._to_norm(
            c_nn,
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
        vars: List[str],
    ) -> NDArray[np.float32]:
        """Standard data normalization.
        
        Parameters
        ----------
        data
            Data to normalize.
        vars
            List of variable names in data to normalize.
        
        Returns
        -------
        NDArray[np.float32]
            Normalized data.
        """
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            if var not in self.norm_stats:
                log.warning(f"No normalization stats for {var}, skipping")
                continue
                
            stat = self.norm_stats[var]

            try:
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
            except Exception as e:
                log.warning(f"Error normalizing {var}: {e}")
                # Copy original data if normalization fails
                if len(data.shape) == 3:
                    data_norm[:, :, k] = data[:, :, k]
                else:
                    data_norm[:, k] = data[:, k]

        # NOTE: Should be external, except altering order of first two dims
        # augments normalization...
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

    def _from_norm(
        self,
        data_scaled: NDArray[np.float32],
        vars: List[str],
    ) -> NDArray[np.float32]:
        """De-normalize data.
        
        Parameters
        ----------
        data_scaled
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
            if var not in self.norm_stats:
                log.warning(f"No normalization stats for {var}, skipping")
                continue
                
            stat = self.norm_stats[var]
            
            try:
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
            except Exception as e:
                log.warning(f"Error denormalizing {var}: {e}")
                # Copy normalized data if denormalization fails
                if len(data_scaled.shape) == 3:
                    data[:, :, k] = data_scaled[:, :, k]
                else:
                    data[:, k] = data_scaled[:, k]

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)

    def _load_nn_data(self, scope: str, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Load and process neural network data from NetCDF, including target variables."""
        time_range = [t_range['start'].replace('/', '-'), t_range['end'].replace('/', '-')]
        warmup_days = self.config['dpl_model']['phy_model']['warm_up']

        try:
            # Get subset path if exists
            subset_path = self.config['observations'].get('subset_path')
            selected_basins = None
            
            # Only load subset info if the file exists
            if subset_path and os.path.exists(subset_path):
                try:
                    with open(subset_path, 'r') as f:
                        content = f.read().strip()
                        if content.startswith('[') and content.endswith(']'):
                            content = content.strip('[]')
                            selected_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                        else:
                            selected_basins = [line.strip() for line in content.split() if line.strip()]
                    log.info(f"Loaded {len(selected_basins)} stations from subset file")
                except Exception as e:
                    log.warning(f"Error loading subset file: {e}, will use all stations")
                    selected_basins = None
            
            # Convert to string list if we have basins
            station_ids = [str(id) for id in selected_basins] if selected_basins else None
            
            # Create a combined list of variables to load (includes target)
            all_variables = self.nn_forcings.copy()
            for target_var in self.target:
                if target_var not in all_variables:
                    all_variables.append(target_var)
            
            # Load data from NetCDF
            log.info(f"Loading data from NetCDF with time range: {time_range}")
            log.info(f"Variables to load: {all_variables}")
            log.info(f"Static attributes: {self.nn_attributes}")
            
            # Let the NetCDF loader handle subsetting if station_ids is provided
            time_series_data, static_data, date_range = self.nc_tool.nc2array(
                self.config['data_path'],
                station_ids=station_ids,  # Will be None if no subsetting
                time_range=time_range,
                time_series_variables=all_variables,
                static_variables=self.nn_attributes,
                add_coords=True,
                warmup_days=warmup_days
            )
            
            log.info(f"Loaded data shapes - time_series: {time_series_data.shape}, static: {static_data.shape}")
            
            # Handle coordinates
            if static_data.shape[1] >= 2:  # Make sure we have enough columns
                lon = static_data[:, -1]
                lat = static_data[:, -2]
                static_data = static_data[:, :-2]
            
            # Extract target and forcing data
            target_indices = []
            for target_var in self.target:
                if target_var in all_variables:
                    target_indices.append(all_variables.index(target_var))
            
            # Extract targets from time_series_data
            target_data = None
            if target_indices:
                target_data = time_series_data[:, :, target_indices]
                log.info(f"Extracted target data with shape: {target_data.shape}")
            else:
                log.warning(f"No target variables found in data. Available: {all_variables}, Requested: {self.target}")
            
            # Extract forcing data (all non-target variables)
            forcing_indices = [i for i in range(len(all_variables)) if i not in target_indices]
            forcing_data = time_series_data[:, :, forcing_indices] if forcing_indices else None
            
            if forcing_data is None:
                log.error("No forcing data available after filtering")
                raise ValueError("Missing forcing data")
            
            log.info(f"Extracted forcing data with shape: {forcing_data.shape}")
            
            # Calculate normalization statistics
            epsilon = 1e-5
            
            # Ensure data is in float format and handle NaNs
            forcing_data = np.nan_to_num(forcing_data.astype(np.float32), nan=0.0)
            static_data = np.nan_to_num(static_data.astype(np.float32), nan=0.0)
            
            # Convert to tensors for return
            forcing_tensor = torch.tensor(forcing_data, dtype=torch.float32, device=self.device)
            static_tensor = torch.tensor(static_data, dtype=torch.float32, device=self.device)
            
            return {
                'x_nn': forcing_tensor,         # (stations, time, features)
                'c_nn': static_tensor,          # (stations, features)
                'target': target_data           # (stations, time, target_features) or None
            }
            
        except Exception as e:
            log.error(f"Error loading neural network data: {str(e)}")
            import traceback
            log.error(traceback.format_exc())
            raise