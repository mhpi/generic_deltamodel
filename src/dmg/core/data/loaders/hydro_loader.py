"""
Data loader for HydroDL LSTMs and differentiable models.
- Leo Lonzarich, Yalan Song 2024.
"""

import json
import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.exceptions import DataDimensionalityWarning

from dmg.core.data.data import intersect, split_dataset_by_basin
from dmg.core.data.loaders.base import BaseLoader

log = logging.getLogger(__name__)


class HydroLoader(BaseLoader):
    """Data loader for hydrological data from CAMELS dataset.

    All data is loaded as PyTorch tensors. According to config settings,
    generates...
    - `dataset` for model inference,
    - `train_dataset` for training,
    - `eval_dataset` for testing.

    The CAMELS dataset is a large-sample watershed-scale hydrometeorological
    dataset for the contiguous USA and includes both meteorological forcings
    and basin attributes.

    CAMELS:
    - https://ral.ucar.edu/solutions/products/camels

    - A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett,
        2014. A large-sample watershed-scale hydrometeorological dataset for the
        contiguous USA. Boulder, CO: UCAR/NCAR.
        https://dx.doi.org/10.5065/D6MW2F4D

    Parameters
    ----------
    config
        Configuration dictionary.
    test_split
        Whether to split data into training and testing sets.
    overwrite
        Whether to overwrite existing normalization statistics.
    holdout_index
        Index for spatial holdout testing.

    NOTE: to support new datasets of similar form to CAMELS, add the dataset
    key name to `self.supported_data`.
    """

    def __init__(
        self,
        config: dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        self.holdout_index = holdout_index
        self.supported_data = [
            'camels_671',
            'camels_531',
            'prism_671',
            'prism_531',
            'camels_671_lstm',
            'camels_531_lstm',
        ]
        self.data_name = config['observations']['name']
        self.nn_attributes = config['model']['nn'].get('attributes', [])
        self.nn_forcings = config['model']['nn'].get('forcings', [])
        self.forcing_names = self.config['observations']['all_forcings']
        self.attribute_names = self.config['observations']['all_attributes']

        if config['model']['phy']:
            self.phy_attributes = config['model']['phy'].get('attributes', [])
            self.phy_forcings = config['model']['phy'].get('forcings', [])
        else:
            self.phy_attributes = []
            self.phy_forcings = []

        self.target = config['train']['target']
        self.log_norm_vars = config['model'].get('use_log_norm', [])
        self.flow_regime = config['model']['flow_regime']
        self.device = config['device']
        self.dtype = config['dtype']

        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None

        if self.data_name not in self.supported_data:
            raise ValueError(f"Data source '{self.data_name}' not supported.")

        if self.log_norm_vars is None:
            self.log_norm_vars = []

        if self.flow_regime == 'high':
            # High flow regime: Gaussian normalization for all variables
            self.log_norm_vars = []
            self.norm_target = True
        elif self.flow_regime == 'low':
            # Low flow regime: Log-Gamma normalization for runoff and precipitation
            self.log_norm_vars = ['prcp', 'runoff']
        else:
            self.norm_target = False

        self.load_dataset()

    def load_dataset(self) -> None:
        """Load data into dictionary of nn and physics model input tensors."""
        mode = self.config['mode']
        is_spatial_test = self.config.get('test', {}).get('type') == 'spatial'

        if mode == 'simulation':
            self.dataset = self._preprocess_data(scope='simulation')
        elif is_spatial_test:
            # For spatial testing, load data and split by basin using utility function
            train_dataset = self._preprocess_data(scope='train')
            test_dataset = self._preprocess_data(scope='test')

            self.train_dataset, _ = split_dataset_by_basin(
                train_dataset,
                self.config,
                self.holdout_index,
            )
            _, self.eval_dataset = split_dataset_by_basin(
                test_dataset,
                self.config,
                self.holdout_index,
            )
        elif self.test_split:
            self.train_dataset = self._preprocess_data(scope='train')
            self.eval_dataset = self._preprocess_data(scope='test')
        elif mode in ['train', 'test']:
            self.train_dataset = self._preprocess_data(scope=mode)
        else:
            self.dataset = self._preprocess_data(scope='all')

    def _preprocess_data(
        self,
        scope: Optional[str],
    ) -> dict[str, torch.Tensor]:
        """Read data, preprocess, and return as tensors for models.

        Parameters
        ----------
        scope
            Scope of data to read, affects what timespan of data is loaded.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of data tensors for running models.
        """
        x_phy, c_phy, x_nn, c_nn, target = self.read_data(scope)

        # Normalize nn input data
        self.load_norm_stats(x_nn, c_nn, target)
        xc_nn_norm, y_nn_norm = self.normalize(x_nn, c_nn, target)

        if y_nn_norm is not None:
            target = y_nn_norm
            del y_nn_norm

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

    def read_data(self, scope: Optional[str]) -> tuple[NDArray[np.float32]]:
        """Read data from the data file.

        Parameters
        ----------
        scope
            Scope of data to read, affects what timespan of data is loaded.

        Returns
        -------
        tuple[NDArray[np.float32]]
            Tuple of neural network + physics model inputes, and target data.
        """
        try:
            if self.config['observations']['data_path']:
                data_path = self.config['observations']['data_path']

            if scope == 'train':
                if not data_path:
                    # NOTE: still including 'train_path' etc. for backwards
                    # compatibility until all code is updated to use 'data_path'.
                    data_path = self.config['observations']['train_path']
                time = self.config['train_time']
            elif scope == 'test':
                if not data_path:
                    data_path = self.config['observations']['test_path']
                time = self.config['test_time']
            elif scope == 'simulation':
                if not data_path:
                    data_path = self.config['observations']['test_path']
                time = self.config['sim_time']
            elif scope == 'all':
                if not data_path:
                    data_path = self.config['observations']['test_path']
                time = self.config['all_time']
            else:
                raise ValueError(
                    "Scope must be 'train', 'test', 'simulation', or 'all'."
                )
        except KeyError as e:
            raise ValueError(f"Key {e} for data path not in dataset config.") from e

        # Get time indicies
        all_time = pd.date_range(
            self.config['all_time'][0],
            self.config['all_time'][-1],
            freq='d',
        )
        idx_start = all_time.get_loc(time[0])
        idx_end = all_time.get_loc(time[-1]) + 1

        # Load data
        with open(data_path, 'rb') as f:
            forcings, target, attributes = pickle.load(f)

        forcings = np.transpose(forcings[:, idx_start:idx_end], (1, 0, 2))

        # Forcing subset for phy model
        phy_forc_idx = []
        for forc in self.phy_forcings:
            if forc not in self.forcing_names:
                raise ValueError(f"Forcing {forc} not listed in available forcings.")
            phy_forc_idx.append(self.forcing_names.index(forc))

        # Attribute subset for phy model
        phy_attr_idx = []
        for attr in self.phy_attributes:
            if attr not in self.attribute_names:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            phy_attr_idx.append(self.attribute_names.index(attr))

        # Forcings subset for nn model
        nn_forc_idx = []
        for forc in self.nn_forcings:
            if forc not in self.forcing_names:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            nn_forc_idx.append(self.forcing_names.index(forc))

        # Attribute subset for nn model
        nn_attr_idx = []
        for attr in self.nn_attributes:
            if attr not in self.attribute_names:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            nn_attr_idx.append(self.attribute_names.index(attr))

        x_phy = forcings[:, :, phy_forc_idx]
        c_phy = attributes[:, phy_attr_idx]
        x_nn = forcings[:, :, nn_forc_idx]
        c_nn = attributes[:, nn_attr_idx]
        target = np.transpose(target[:, idx_start:idx_end], (1, 0, 2))

        # Subset basins if necessary
        if self.config['observations']['subset_path'] is not None:
            subset_path = self.config['observations']['subset_path']
            gage_id_path = self.config['observations']['gage_info']

            with open(subset_path) as f:
                selected_basins = json.load(f)
            gage_info = np.load(gage_id_path)

            subset_idx = intersect(selected_basins, gage_info)

            x_phy = x_phy[:, subset_idx, :]
            c_phy = c_phy[subset_idx, :]
            x_nn = x_nn[:, subset_idx, :]
            c_nn = c_nn[subset_idx, :]
            target = target[:, subset_idx, :]

        # Convert flow to mm/day if necessary
        target = self._flow_conversion(c_nn, target)

        return x_phy, c_phy, x_nn, c_nn, target

    def _flow_conversion(
        self,
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Convert hydraulic flow from ft3/s to mm/day.

        Parameters
        ----------
        c_nn
            Neural network static data.
        target
            Target variable data.
        """
        for name in ['flow_sim', 'streamflow', 'runoff']:
            if name in self.target:
                target_temp = target[:, :, self.target.index(name)]
                area_name = self.config['observations']['area_name']
                basin_area = c_nn[:, self.nn_attributes.index(area_name)]

                area = np.expand_dims(basin_area, axis=0).repeat(
                    target_temp.shape[0], 0
                )
                target[:, :, self.target.index(name)] = (
                    (10**3) * target_temp * 0.0283168 * 3600 * 24 / (area * (10**6))
                )

                if self.config['model']['phy'] is None:
                    # make target dimensionless
                    prcp_mean_name = self.config['observations']['prcp_mean_name']
                    prcp_mean = c_nn[:, self.nn_attributes.index(prcp_mean_name)]
                    target[:, :, self.target.index(name)] /= prcp_mean
        return target

    def load_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> None:
        """Load or calculate normalization statistics if necessary."""
        self.out_path = os.path.join(
            self.config['model_dir'],
            'normalization_statistics.json',
        )

        if os.path.isfile(self.out_path) and (not self.overwrite):
            if not self.norm_stats:
                with open(self.out_path) as f:
                    self.norm_stats = json.load(f)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True.
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)

    def _init_norm_stats(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
    ) -> dict[str, list[float]]:
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
        dict[str, list[float]]
            Dictionary of normalization statistics for each variable.
        """
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

        # Target variable stats
        for i, name in enumerate(self.target):
            if (name in ['flow_sim', 'streamflow', 'runoff']) and (
                not self.norm_target
            ):
                stat_dict[name] = self._calc_norm_stats(
                    np.swapaxes(target[:, :, i : i + 1], 1, 0).copy(),
                    basin_area,
                )
            else:
                stat_dict[name] = self._calc_norm_stats(
                    np.swapaxes(target[:, :, i : i + 1], 1, 0).copy(),
                )

        with open(self.out_path, 'w') as f:
            json.dump(stat_dict, f, indent=4)

        return stat_dict

    def _calc_norm_stats(
        self,
        x: NDArray[np.float32],
        basin_area: NDArray[np.float32] = None,
    ) -> list[float]:
        """
        Calculate statistics for normalization with optional basin
        area adjustment.

        Parameters
        ----------
        x
            Input data array.
        basin_area
            Basin area array for normalization.

        Returns
        -------
        list[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        # Handle invalid values
        x[x == -999] = np.nan
        if basin_area is not None:
            x[x < 0] = 0  # Specific to basin normalization

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if (nd == 3) and (x.shape[2] == 1):
                x = x[:, :, 0]  # Unsqueeze the original 3D matrix
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10**6)) * 10**3
            x = flow  # Replace x with flow for further calculations

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            if len(x.shape) > 1:
                a = np.swapaxes(x, 1, 0).flatten()
            else:
                a = x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            b = np.array([0])

        # Calculate stats
        if basin_area is not None:
            transformed = np.log10(np.sqrt(b) + 0.1)
        else:
            transformed = b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _calc_gamma_stats(self, x: NDArray[np.float32]) -> list[float]:
        """Calculate gamma statistics for streamflow and precipitation data.

        Parameters
        ----------
        x
            Input data array.

        Returns
        -------
        list[float]
            List of statistics [10th percentile, 90th percentile, mean, std].
        """
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        b = np.log10(np.sqrt(b) + 0.1)

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]

    def _get_basin_area(self, c_nn: NDArray[np.float32]) -> NDArray[np.float32]:
        """Get basin area from attributes.

        Parameters
        ----------
        c_nn
            Neural network static data.

        Returns
        -------
        NDArray[np.float32]
            1D array of basin areas (2nd dummy dim added for calculations).
        """
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
        except KeyError:
            log.warning(
                "No 'area_name' in observation config. Basin"
                "area norm will not be applied."
            )
            basin_area = None
        return basin_area

    def normalize(
        self,
        x_nn: NDArray[np.float32],
        c_nn: NDArray[np.float32],
        target: NDArray[np.float32],
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

        if not self.norm_target:
            y_nn_norm = None
        else:
            y_nn_norm = self._to_norm(
                np.swapaxes(target, 1, 0).copy(),
                self.target,
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
        del x_nn_norm, c_nn_norm, x_nn

        return xc_nn_norm, y_nn_norm

    def _to_norm(
        self,
        data: NDArray[np.float32],
        vars: list[str],
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

        # NOTE: Should be external, except altering order of first two dims
        # augments normalization...
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

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
