import json
import logging
import os

from math import log
from typing import Any, Dict, List, Optional, Tuple
import zarr

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

from core.data.data_loaders.base import BaseDataLoader

log = logging.getLogger(__name__)


class HydroMSDataLoader(BaseDataLoader):
    """Data loader for multiscale hydrological data loading.
    
    All data is read from Zarr store and loaded as PyTorch tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    test_split : bool, optional
        Whether to split data into training and testing sets. Default is False.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.supported_data = ['merit_forward']
        self.data_name = config['observations']['name']
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['dpl_model']['nn_model'].get('forcings', [])
        self.phy_attributes = config['dpl_model']['phy_model'].get('attributes', [])
        self.phy_forcings = config['dpl_model']['phy_model'].get('forcings', [])
        self.all_forcings = self.config['observations']['forcings_all']
        self.all_attributes = self.config['observations']['attributes_all']

        self.target = config['train']['target']
        self.log_norm_vars = config['dpl_model']['phy_model']['use_log_norm']
        self.device = config['device']
        self.dtype = config['dtype']
        
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        self.out_path = os.path.join(
            config['model_path'],
            'normalization_statistics.json',
        )

        if self.data_name not in self.supported_data:
            raise ValueError(f"Data source '{self.data_name}' not supported.")

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset into dictionary of nn and physics model input arrays."""    
        if self.test_split:
            self.train_dataset = self._preprocess_data(scope='train')
            self.eval_dataset = self._preprocess_data(scope='test')
        elif self.config['mode'] == 'predict':
            self.dataset = self._preprocess_data(scope='predict')
        else:
            self.dataset = self._preprocess_data(scope='all')

    def _preprocess_data(self, scope: Optional[str]) -> Dict[str, npt.NDArray]:
        """Read data from the dataset."""
        ac_all, elev_all, subbasin_id_all, x_nn, x_phy, c_nn = self.read_data(scope)


        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn)
        xc_nn_norm, c_nn_norm = self.normalize(x_nn, c_nn)

        # Build data dict of Torch tensors
        dataset = {
            'ac_all': self.to_tensor(ac_all),
            'elev_all': self.to_tensor(elev_all),
            'subbasin_id_all': self.to_tensor(subbasin_id_all),
            'c_nn': self.to_tensor(c_nn),
            'xc_nn_norm': self.to_tensor(xc_nn_norm),
            'c_nn_norm': self.to_tensor(c_nn_norm),
            'x_phy': self.to_tensor(x_phy),
        }
        return dataset

    def read_data(self, scope: Optional[str]) -> Tuple[npt.NDArray]:
        """Read data from the data file."""
        try:
            if scope == 'train':
                time = self.config['train_time']
            elif scope == 'test':
                time = self.config['test_time']
            elif scope == 'predict':
                time = self.config['predict_time']                
            elif scope == 'all':
                time = self.config['all_time']
            else:
                raise ValueError("Scope must be 'train', 'test', 'predict', or 'all'.")
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.")
        
        # Get time indicies
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1], 
            freq='d',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        root_zone = zarr.open_group(
            self.config['observations']['subbasin_data_path'],
            mode = 'r',
        )
        subbasin_id_all = np.array(
            root_zone[self.config['observations']['subbasin_id_name']][:]
        ).astype(int)

        # Forcing subset for phy model
        for i, forc in enumerate(self.phy_forcings):
            if forc not in self.all_forcings:
                raise ValueError(f"Forcing {forc} not listed in available forcings.")
            if i == 0:
                forc_array = np.expand_dims(root_zone[forc][:, idx_start:idx_end], -1) 
            else:
                forc_array = np.concatenate((
                    forc_array,
                    np.expand_dims(root_zone[forc][:, idx_start:idx_end], -1),
                ), axis = -1)
        forc_array = self._fill_nan(forc_array)

        # Attribute subset for nn model
        for i, attr in enumerate(self.nn_attributes):
            if attr not in self.all_attributes:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            if i == 0:
                attr_array = np.expand_dims(root_zone['attrs'][attr][:],-1)
            else:
                attr_array = np.concatenate((
                    attr_array,
                    np.expand_dims(root_zone['attrs'][attr][:], -1)
                ), axis = -1)
        
        # Get upstream area and elevation
        try:
            ac_name = self.config['observations']['upstream_area_name']
            ac_array = root_zone['attrs'][ac_name][:]
        except:
            raise ValueError(f"Upstream area is not provided. This is needed for high-resolution streamflow model.")
        try:
            elevation_name = self.config['observations']['elevation_name']
            elev_array = root_zone['attrs'][elevation_name][:] 
        except:
            raise ValueError(f"Elevation is not provided. This is needed for high-resolution streamflow model.")
        
        return [
            ac_array,
            elev_array,
            subbasin_id_all,
            forc_array,
            forc_array.copy(),
            attr_array,
        ]
    
    def _flow_conversion(self, c_nn, target) -> npt.NDArray:
        """Convert hydraulic flow from ft3/s to mm/day."""
        for name in ['flow_sim', 'streamflow', 'sf']:
            if name in self.target:
                target_temp = target[:, :, self.target.index(name)]
                area_name = self.config['observations']['area_name']
                basin_area = c_nn[:, self.nn_attributes.index(area_name)]

                area = np.expand_dims(basin_area, axis=0).repeat(target_temp.shape[0], 0)
                target[:, :, self.target.index(name)] = (
                    (10 ** 3) * target_temp * 0.0283168 * 3600 * 24 / (area * (10 ** 6))
                )
        return target
        
    def load_norm_stats(
        self,
        x_nn: npt.NDArray,
        c_nn: npt.NDArray,
    ) -> None:
        """Load or calculate normalization statistics if necessary."""
        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                with open(self.out_path, 'r') as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(x_nn, c_nn)
            # raise ValueError("Normalization statistics not found. Confirm 'normalization_statistics.json' is in your model directory.")
    
    def _init_norm_stats(
        self,
        x_nn: npt.NDArray,
        c_nn: npt.NDArray,
    ) -> Dict[str, List[float]]:
        """Compile calculations of data normalization statistics."""
        stat_dict = {}

        # Get basin areas from attributes.
        basin_area = self._get_basin_area(c_nn)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            if var in self.log_norm_vars:
                stat_dict[var] = self._calc_gamma_stats(x_nn[:, :, k])
            else:
                stat_dict[var] = self._calc_norm_stats(x_nn[:, :, k])

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            stat_dict[var] = self._calc_norm_stats(c_nn[:, k])

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)
        
        return stat_dict

    def _calc_norm_stats(
        self,
        x: npt.NDArray, 
        basin_area: npt.NDArray = None, 
    ) -> List[float]:
        """
        Calculate statistics for normalization with optional basin
        area adjustment.
        """
        # Handle invalid values
        x[x == -999] = np.nan
        if basin_area is not None:
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
            b = np.array([0])

        # Calculate statistics
        transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _calc_gamma_stats(self, x: npt.NDArray) -> List[float]:
        """Calculate gamma statistics for streamflow and precipitation data."""
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        b = np.log10(
            np.sqrt(b) + 0.1
        )

        p10, p90 = np.percentile(b, [10,90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _get_basin_area(self, c_nn: npt.NDArray) -> npt.NDArray:
        """Get basin area from attributes."""
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
        except KeyError:
            log.warning("No 'area_name' in observation config. Basin area norm will not be applied.")
            basin_area = None

        return basin_area

    def normalize(self, x_nn: npt.NDArray, c_nn: npt.NDArray) -> npt.NDArray:
        """Normalize data for neural network."""
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
            axis=0,
        )

        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm), axis=2)
        del x_nn_norm, x_nn

        return xc_nn_norm, c_nn_norm

    def _to_norm(self, data: npt.NDArray, vars: List[str]) -> npt.NDArray:
        """Standard data normalization."""
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            stat = self.norm_stats[var]

            if len(data.shape) == 3:
                if var in self.log_norm_vars:
                    data[:, :, k] = np.log10(np.sqrt(data[:, :, k]) + 0.1)
                data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
            elif len(data.shape) == 2:
                if var in self.log_norm_vars:
                    data[:, k] = np.log10(np.sqrt(data[:, k]) + 0.1)
                data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
            
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)  ##TODO move to outside of _to_norm

    def _from_norm(self, data_norm: npt.NDArray, vars: List[str]) -> npt.NDArray:
        """De-normalize data."""
        data = np.zeros(data_norm.shape)
                
        for k, var in enumerate(vars):
            stat = self.norm_stats[var]
            if len(data_norm.shape) == 3:
                data[:, :, k] = data_norm[:, :, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
            elif len(data_norm.shape) == 2:
                data[:, k] = data_norm[:, k] * stat[3] + stat[2]
                if var in self.log_norm_vars:
                    data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
            else:
                raise DataDimensionalityWarning("Data dimension must be 2 or 3.")

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)

    def _fill_nan(self, array_3d):
        # Define the x-axis for interpolation
        x = np.arange(array_3d.shape[1])

        # Iterate over the first and third dimensions to interpolate the second dimension
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[2]):
                # Select the 1D slice for interpolation
                slice_1d = array_3d[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Only interpolate if there are NaNs and at least two non-NaN values for reference
                if np.any(nans) and np.sum(non_nans) > 1:
                    # Perform linear interpolation using numpy.interp
                    array_3d[i, :, j] = np.interp(x, x[non_nans], slice_1d[non_nans], left=None, right=None)
        return array_3d
