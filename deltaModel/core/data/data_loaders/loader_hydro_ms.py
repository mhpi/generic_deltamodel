import numpy as np
import pandas as pd
import zarr

# from core.data.data_loaders.base_data_loader import BaseDataLoader
# class HydroDataMSLoader(BaseDataLoader):


class choose_class_to_read_dataset():
    def __init__(self, config, trange, data_path):
        self.config = config
        self.trange = trange
        self.data_path = data_path
        self._get_dataset_class()
        
    def _get_dataset_class(self) -> None:
        if self.data_path.endswith(".feather") or self.data_path.endswith(".csv"):
            self.read_data = DataFrameDataset(config=self.config, tRange=self.trange, data_path=self.data_path)
        elif self.data_path.endswith(".npy") or self.data_path.endswith(".pt"):
            self.read_data = ArrayDataset(config=self.config, tRange=self.trange, data_path=self.data_path)


def load_data_subbasin(config, train=True):
    """ Load data into dictionaries for pNN and hydro model. """

    out_dict = dict()

    if config['observations']['name'] in ['camels_671', 'camels_531','merit_forward_zone']:
        if train:
            startdate =config['train']['start_time']
            enddate = config['train']['end_time']
            
        else:
            startdate =config['test']['start_time']
            enddate = config['test']['end_time']

        all_time = pd.date_range(config['observations']['start_time'], config['observations']['end_time'], freq='d')
        new_time = pd.date_range(startdate, enddate, freq='d')
        
        index_start = all_time.get_loc(new_time[0])
        index_end = all_time.get_loc(new_time[-1]) + 1

        subbasin_data_path = config['observations']['subbasin_data_path']
        subbasin_id_name = config['observations']['subbasin_id_name']
        root_zone = zarr.open_group(subbasin_data_path, mode = 'r')
        subbasin_ID_all = np.array(root_zone[subbasin_id_name][:]).astype(int)
        
        for attr_idx, attr in enumerate(config['dpl_model']['nn_model']['attributes']) :
            if attr not in config['observations']['attributes_all']:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            if attr_idx == 0:
                attr_array = np.expand_dims(root_zone['attrs'][attr][:],-1) 
            else:
                attr_array = np.concatenate((attr_array,np.expand_dims(root_zone['attrs'][attr][:],-1) ),axis = -1)

        try:
            Ac_name = config['observations']['upstream_area_name'] 
        except:
            raise ValueError(f"Upstream area is not provided. This is needed for high-resolution streamflow model")

        try:
            elevation_name = config['observations']['elevation_name'] 
        except:
            raise ValueError(f"Elevation is not provided. This is needed for high-resolution streamflow model")


        Ac_array = root_zone['attrs'][Ac_name][:]
 
        Ele_array = root_zone['attrs'][elevation_name][:]

        for forc_idx, forc in enumerate(config['dpl_model']['nn_model']['forcings']):
            if forc not in config['observations']['forcings_all']:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            if forc_idx == 0:
                forcing_array = np.expand_dims(root_zone[forc][:,index_start:index_end],-1) 
            else:
                forcing_array = np.concatenate((forcing_array,np.expand_dims(root_zone[forc][:,index_start:index_end],-1) ),axis = -1)

        forcing_array = fill_Nan(forcing_array)
        out_dict['Ac_all'] = Ac_array
        out_dict['Ele_all'] = Ele_array
        out_dict['subbasin_ID_all'] = subbasin_ID_all
        out_dict['x_nn'] = forcing_array  # Forcings for neural network (note, slight error from indexing)
        out_dict['x_phy'] = forcing_array.copy() # Forcings for physics model
        out_dict['c_nn'] = attr_array # Attributes

    else:    
        raise ValueError(f"Errors when loading data for multicale model")   
    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(config, gage_area, obs_sample):
    varTar_NN = config['train']['target']
    if 'flow_sim' in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index('flow_sim')]

        area = np.expand_dims(gage_area, axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index('flow_sim')] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample


def get_dataset_dict(config, train=False):
    """
    Create dictionary of datasets used by the models.
    Contains 'c_nn', 'target', 'x_phy', 'xc_nn_norm'.

    train: bool, specifies whether data is for training.
    """
    # Create stats for NN input normalizations.
    dataset_dict = load_data_subbasin(config, train=False)

    # Create stats for NN input normalizations.
    # init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'], dataset_dict['x_phy'])

    # Normalization
    x_nn_norm = trans_norm(config, dataset_dict['x_nn'],
                             var_lst=config['dpl_model']['nn_model']['forcings'], to_norm=True)
    x_nn_norm[x_nn_norm != x_nn_norm] = 0  # Remove nans

    c_nn_norm = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['dpl_model']['nn_model']['attributes'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_norm[c_nn_norm != c_nn_norm] = 0  # Remove nans
    c_nn_norm_repeat = np.repeat(np.expand_dims(c_nn_norm, 0), x_nn_norm.shape[0], axis=0)

    dataset_dict['xc_nn_norm'] = np.concatenate((x_nn_norm, c_nn_norm_repeat), axis=2)

    dataset_dict['c_nn_norm'] = c_nn_norm
    x_phy = np.swapaxes(dataset_dict['x_phy'], 1, 0)
    x_phy[x_phy != x_phy] = 0 
    dataset_dict['x_phy'] =  x_phy
    del x_nn_norm, c_nn_norm, dataset_dict['x_nn']

    return dataset_dict





#################################################
# Debugging code
#################################################




# TODO: place this whole file in hydroDL2
import json
import os
from typing import Any, Dict, List

import numpy as np


def calc_stat_basinnorm(x: np.ndarray, basin_area: np.ndarray, config: Dict[str, Any]) -> List[float]:
    """
    From hydroDL.

    Get stats for basin area normalization.

    Parameters
    ----------
    x
        data to be normalized or denormalized
    basin_area
        basins' area
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    x[x == (-999)] = np.nan
    x[x < 0] = 0

    nd = len(x.shape)
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # Unsqueeze the original 3 dim matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))

    flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3  # (m^3/day)/(m^3/day)

    # Apply tranformation to change gamma characteristics, add 0.1 for 0 values.
    a = flow.flatten()
    b = np.log10( np.sqrt(a[~np.isnan(a)]) + 0.1)

    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    # if std < 0.001: std = 1

    return [p10, p90, mean, max(std, 0.001)]


def calculate_statistics(x: np.ndarray) -> List[float]:
    """
    Calculate basic statistics excluding NaNs and specific invalid values.
    
    :param x: Input data
    :return: List of statistics [10th percentile, 90th percentile, mean, std]
    """
    if len(x.shape) > 1:
        a = np.swapaxes(x, 1, 0).flatten()  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    else:
        a = x.flatten()  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...

    b = a[(~np.isnan(a)) & (a != -999999)]
    if b.size == 0:
        b = np.array([0])

    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    # if std < 0.001: std = 1

    return [p10, p90, mean, max(std, 0.001)]


# TODO: Eventually replace calculate_statistics with the version below.
# def calculate_statistics_dmc(data: xr.Dataset, column: str = "time", row: str = "gage_id") -> Dict[str, torch.Tensor]:
#     """
#     Calculate statistics for the data in a similar manner to calStat from hydroDL.

#     :param data: xarray Dataset
#     :param column: Name of the column for calculations
#     :param row: Name of the row for calculations
#     :return: Dictionary with statistics
#     """
#     statistics = {}
#     p_10 = data.quantile(0.1, dim=column)
#     p_90 = data.quantile(0.9, dim=column)
#     mean = data.mean(dim=column)
#     std = data.std(dim=column)
#     col_names = data[row].values.tolist()
#     for idx, col in enumerate(
#         tqdm(col_names, desc="\rCalculating statistics", ncols=140, ascii=True)
#     ):
#         col_str = str(col)
#         statistics[col_str] = torch.tensor(
#             data=[
#                 p_10.streamflow.values[idx],
#                 p_90.streamflow.values[idx],
#                 mean.streamflow.values[idx],
#                 std.streamflow.values[idx],
#             ]
#         )
#     return statistics


def calculate_statistics_gamma(x: np.ndarray) -> List[float]:
    """
    Taken from the cal_stat_gamma function of hydroDL.

    Calculate gamma statistics for streamflow and precipitation data.

    :param x: Input data
    :return: List of statistics [10th percentile, 90th percentile, mean, std]
    """
    a = np.swapaxes(x, 1, 0).flatten()  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    b = a[(~np.isnan(a))]  # & (a != -999999)]
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # Transform to change gamma characteristics, add 0.1 for 0 values.

    p10, p90 = np.percentile(b, [10,90]).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)

    return [p10, p90, mean, max(std, 0.001)]


# def calculate_statistics_all(config: Dict[str, Any], x: np.ndarray, c: np.ndarray,
#                              y: np.ndarray, c_all=None) -> None:
#     """
#     Taken from the calStatAll function of hydroDL.
    
#     Calculate and save statistics for all variables in the config.

#     :param config: Configuration dictionary
#     :param x: Forcing data
#     :param c: Attribute data
#     :param y: Target data
#     """
#     stat_dict = {}

#     # Calculate basin area 
#     # NOTE: should probably move to separate function.
#     attr_list = config['dpl_model']['nn_model']['attributes']

#     area_name = config['observations']['area_name']
    
#     if c_all is not None:
#         # Basin area calculation for MERIT.
#         basin_area = np.expand_dims(c_all["area"].values,axis = 1)
#     else:
#         basin_area = c[:, attr_list.index(area_name)][:, np.newaxis]


#     # Target stats
#     for i, target_name in enumerate(config['train']['target']):
#         if target_name in ['flow_sim', 'streamflow']:
#             stat_dict[config['train']['target'][i]] = calc_stat_basinnorm(
#                 np.swapaxes(y[:, :, i:i+1], 1,0).copy(), basin_area, config
#             )  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
#         else:
#             stat_dict[config['train']['target'][i]] = calculate_statistics(
#                 np.swapaxes(y[:, :, i:i+1], 1,0)
#             )  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...

#     # Forcing stats
#     var_list = config['dpl_model']['nn_model']['forcings']
#     for k, var in enumerate(var_list):
#         if var in config['dpl_model']['phy_model']['use_log_norm']:
#             stat_dict[var] = calculate_statistics_gamma(x[:, :, k])
#         else:
#             stat_dict[var] = calculate_statistics(x[:, :, k])

#     # Attribute stats
#     varList = config['dpl_model']['nn_model']['attributes']
#     for k, var in enumerate(varList):
#         stat_dict[var] = calculate_statistics(c[:, k])

#     # Save all stats.
#     stat_file = os.path.join(config['out_path'], 'statistics_basinnorm.json')
#     with open(stat_file, 'w') as f:
#         json.dump(stat_dict, f, indent=4)

def calculate_statistics_all(config: Dict[str, Any], x: np.ndarray, c: np.ndarray,
                             y: np.ndarray, c_all=None) -> None:
    """
    Taken from the calStatAll function of hydroDL.
    
    Calculate and save statistics for all variables in the config.

    :param config: Configuration dictionary
    :param x: Forcing data
    :param c: Attribute data
    :param y: Target data
    """
    stat_dict = {}

    # Calculate basin area 
    # NOTE: should probably move to separate function.

    # Target stats
    target_list = config['train']['target']
    for i, target_name in enumerate(target_list):
        if target_name in config['dpl_model']['phy_model']['use_log_norm']:
            stat_dict[target_name] = calculate_statistics_gamma(
                y[:, :, i:i+1]
            )  ## NOTE: swap axes to match Yalan's HBV. This affects calculations...  ## Why swap in the beginning?
        else:
            stat_dict[target_name] = calculate_statistics(
                y[:, :, i:i+1]
            )             
    # Forcing stats
    var_list = config['dpl_model']['nn_model']['forcings']
    for k, var in enumerate(var_list):
        if var in config['dpl_model']['phy_model']['use_log_norm']:
            stat_dict[var] = calculate_statistics_gamma(x[:, :, k])
        else:
            stat_dict[var] = calculate_statistics(x[:, :, k])

    # Attribute stats
    varList = config['dpl_model']['nn_model']['attributes']
    for k, var in enumerate(varList):
        stat_dict[var] = calculate_statistics(c[:, k])

    # Save all stats.
    stat_file = os.path.join(config['out_path'], 'statistics_basinnorm.json')
    with open(stat_file, 'w') as f:
        json.dump(stat_dict, f, indent=4)



def basin_norm(
    x: np.array, basin_area: np.array, to_norm: bool
) -> np.array:
    """
    From HydroDL.

    Normalize or denormalize streamflow data with basin area and mean precipitation.

    The formula is as follows when normalizing (denormalize equation is its inversion):

    .. math:: normalized_x = \frac{x}{area * precipitation}

    Because units of streamflow, area, and precipitation are ft^3/s, km^2 and mm/day, respectively,
    and we need (m^3/day)/(m^3/day), we transform the equation as the code shows.

    Parameters
    ----------
    x
        data to be normalized or denormalized
    basin_area
        basins' area
    mean_prep
        basins' mean precipitation
    to_norm
        if true, normalize; else denormalize

    Returns
    -------
    np.array
        normalized or denormalized data
    """
    nd = len(x.shape)
    # meanprep = readAttr(gageid, ['q_mean'])
    if nd == 3 and x.shape[2] == 1:
        x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
    temparea = np.tile(basin_area, (1, x.shape[1]))

    if to_norm is True:
        # flow = (x * 0.0283168 * 3600 * 24) / (
        #     (temparea * (10**6)) * (tempprep * 10 ** (-3))
        # )  # (m^3/day)/(m^3/day)

        flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

    else:
        # flow = (
        #     x
        #     * ((temparea * (10**6)) * (tempprep * 10 ** (-3)))
        #     / (0.0283168 * 3600 * 24)
        # )
        flow = (
            x
            * ((temparea * (10**6)) * (10 ** (-3)))
            / (0.0283168 * 3600 * 24)
        )

    if nd == 3:
        flow = np.expand_dims(flow, axis=2)
    return flow


def trans_norm(config: Dict[str, Any], x: np.ndarray, var_lst: List[str], *, to_norm: bool) -> np.ndarray:
    """
    Taken from the trans_norm function of hydroDL.
    
    Transform normalization for the given data.

    :param config: Configuration dictionary
    :param x: Input data
    :param var_lst: List of variables
    :param to_norm: Whether to normalize or de-normalize
    :return: Transformed data
    """
    # Load in the statistics for forcings.
    stat_file = os.path.join(config['out_path'], 'statistics_basinnorm.json')
    with open(stat_file, 'r') as f:
        stat_dict = json.load(f)

    var_lst = [var_lst] if isinstance(var_lst, str) else var_lst  # Enforce list format

    ## TODO: fix this dataset variable typo. This is a workaround
    # if 'geol_porosity' in var_lst:
    #     var_lst[var_lst.index('geol_porosity')] = 'geol_porostiy'
    
    out = np.zeros(x.shape)
    
    for k, var in enumerate(var_lst):
        stat = stat_dict[var]

        if to_norm:
            if len(x.shape) == 3:
                if var in config['dpl_model']['phy_model']['use_log_norm']: # 'prcp', 'flow_sim', 'combine_discharge
                    x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if var in config['dpl_model']['phy_model']['use_log_norm']:
                    x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                out[:, k] = (x[:, k] - stat[2]) / stat[3]
            else:
                raise ValueError("Incorrect input dimensions. x array must have 2 or 3 dimensions.")
        
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                if var in config['dpl_model']['phy_model']['use_log_norm']:
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
            elif len(x.shape) == 2:
                out[:, k] = x[:, k] * stat[3] + stat[2]
                if var in config['dpl_model']['phy_model']['use_log_norm']:
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
            else:
                raise ValueError("Incorrect input dimensions. x array must have 2 or 3 dimensions.")

    if len(out.shape) < 3:
        return out
    else:
        return np.swapaxes(out, 1, 0) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...

   

def init_norm_stats(config: Dict[str, Any], x_NN: np.ndarray, c_NN: np.ndarray,
                    y: np.ndarray, c_NN_all=None) -> None:
    """
    Initialize normalization statistics and save them to a file.

    :param config: Configuration dictionary
    :param x_NN: Neural network input data
    :param c_NN: Attribute data
    :param y: Target data
    """
    stats_directory = config['out_path']
    stat_file = os.path.join(stats_directory, 'statistics_basinnorm.json')

    if not os.path.isfile(stat_file):
        calculate_statistics_all(config, x_NN, c_NN, y, c_NN_all)



def fill_Nan(array_3d):
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


"""TODO: convert to pytorch.dataset class format."""
import logging
import os
from abc import ABC, abstractmethod
from re import I
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class BaseDataset(ABC, torch.utils.data.Dataset):
    @abstractmethod
    def getDataTs(self, config, varLst):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, config, varLst):
        raise NotImplementedError


log = logging.getLogger(__name__)

# Adapted from dPL_Hydro_SNTEMP @ Farshid Rahmani.
import datetime as dt

import numpy as np


def time_to_date(t, hr=False):
    tOut = None
    if type(t) is int:
        if t < 30000000 and t > 10000000:
            t = dt.datetime.strptime(str(t), "%Y%m%d").date()
            tOut = t if hr is False else t.datetime()

    if type(t) is dt.date:
        tOut = t if hr is False else t.datetime()

    if type(t) is dt.datetime:
        tOut = t.date() if hr is False else t

    if tOut is None:
        raise Exception("Failed to change time to date.")
    return tOut


def trange_to_array(tRange, *, step=np.timedelta64(1, "D")):
    sd = time_to_date(tRange[0])
    ed = time_to_date(tRange[1])
    tArray = np.arange(sd, ed, step)
    return tArray


def intersect(tLst1, tLst2):
    C, ind1, ind2 = np.intersect1d(tLst1, tLst2, return_indices=True)
    return ind1, ind2


class DataFrameDataset(BaseDataset):
    """Credit: Farshid Rahmani."""
    def __init__(self, config, tRange, data_path, attr_path=None):
        self.time = trange_to_array(tRange)
        self.config = config
        self.inputfile = data_path
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.config['observations']['attr_path']))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data

    def getDataTs(self, config, varLst, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]

        if self.inputfile.endswith('.csv'):
            dfMain = pd.read_csv(self.inputfile)
            dfMain_attr = pd.read_csv(self.inputfile_attr)
        elif self.inputfile.endswith('.feather'):
            dfMain = pd.read_feather(self.inputfile)
            dfMain_attr = pd.read_feather(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain['site_no'].unique()
        tLst = trange_to_array(config['t_range'])
        tLstobs = trange_to_array(config['t_range'])
        # nt = len(tLst)
        ntobs = len(tLstobs)
        nNodes = len(sites)

        varLst_forcing = []
        varLst_attr = []
        for var in varLst:
            if var in dfMain.columns:
                varLst_forcing.append(var)
            elif var in dfMain_attr.columns:
                varLst_attr.append(var)
            else:
                print(var, "the var is not in forcing file nor in attr file")
        xt = dfMain.loc[:, varLst_forcing].values
        g = dfMain.reset_index(drop=True).groupby('site_no')
        xtg = [xt[i.values, :] for k, i in g.groups.items()]
        x = np.array(xtg)

        # Function to read static inputs for attr.
        if len(varLst_attr) > 0:
            x_attr_t = dfMain_attr.loc[:, varLst_attr].values
            x_attr_t = np.expand_dims(x_attr_t, axis=2)
            xattr = np.repeat(x_attr_t, x.shape[1], axis=2)
            xattr = np.transpose(xattr, (0, 2, 1))
            x = np.concatenate((x, xattr), axis=2)

        data = x
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, config, varLst, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        
        if 'geol_porosity' in varLst:
            # correct typo
            varLst[varLst.index('geol_porosity')] = 'geol_porostiy'

        inputfile = os.path.join(os.path.realpath(config['observations']['forcing_path']))
        if self.inputfile_attr.endswith('.csv'):
            dfMain = pd.read_csv(self.inputfile)
            dfC = pd.read_csv(self.inputfile_attr)
        elif inputfile.endswith('.feather'):
            dfMain = pd.read_feather(self.inputfile)
            dfC = pd.read_feather(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain['site_no'].unique()
        nNodes = len(sites)
        c = np.empty([nNodes, len(varLst)])

        for k, kk in enumerate(sites):
            data = dfC.loc[dfC['site_no'] == kk, varLst]
            if 'geol_porostiy' in varLst:
                # correct typo
                data = data.rename(columns={'geol_porostiy': 'geol_porosity'})
            data1 = data.to_numpy().squeeze()
            
            c[k, :] = data1

        return c
    



"""TODO: convert to pytorch.dataset class format."""
import logging
import os

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)




class ArrayDataset(BaseDataset):
    """Credit: Farshid Rahmani."""
    def __init__(self, config, tRange, data_path, attr_path=None):
        self.time = trange_to_array(tRange)
        self.config = config
        self.inputfile = data_path   # the dynamic data
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.config['observations']['attr_path']))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data
        # These are default forcings and attributes that are read from the dataset
        self.all_forcings_name = ['prcp', 'tmean', 'pet', 'Lwd', 'PET_hargreaves(mm/day)', 'prcp(mm/day)'
                                'Pres', 'RelHum', 'SpecHum', 'srad(W/m2)',
                                'tmean(C)', 'tmax(C)', 'tmin(C)', 'Wind', 'ccov',
                                'vp(Pa)', '00060_Mean', 'flow_sim', '00010_Mean','dayl(s)']  #
        self.attrLst_name = ['aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT', 'SoilGrids1km_sand',
                             'SoilGrids1km_clay', 'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel',
                             'HWSD_sand', 'HWSD_silt', 'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost',
                             'permeability','seasonality_P', 'seasonality_PET', 'snow_fraction',
                             'snowfall_fraction','T_clay','T_gravel','T_sand', 'T_silt','Porosity',
                             'DRAIN_SQKM', 'lat', 'site_no_int', 'stream_length_square', 'lon']

    def getDataTs(self, config, varLst):
        if type(varLst) is str:
            varLst = [varLst]
       # TODO: looking for a way to read different types of attr + forcings together
        if self.inputfile.endswith(".npy"):
            forcing_main = np.load(self.inputfile)
            if self.inputfile_attr.endswith(".npy"):
                attr_main = np.load(self.inputfile_attr)
            elif self.inputfile_attr.endswith(".feather"):
                attr_main = pd.read_feather(self.inputfile_attr)
        elif self.inputfile.endswith(".pt"):
            forcing_main = torch.load(self.inputfile)
            attr_main = torch.load(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()

        varLst_index_forcing = []
        varLst_index_attr = []
        for var in varLst:
            if var in self.all_forcings_name:
                varLst_index_forcing.append(self.all_forcings_name.index(var))
            elif var in self.attrLst_name:
                varLst_index_attr.append(self.attrLst_name.index(var))
            else:
                print(var, "the var is not in forcing file nor in attr file")
                exit()

        x = forcing_main[:, :, varLst_index_forcing]
        ## for attr
        if len(varLst_index_attr) > 0:
            x_attr_t = attr_main[:, varLst_index_attr]
            x_attr_t = np.expand_dims(x_attr_t, axis=2)
            xattr = np.repeat(x_attr_t, x.shape[1], axis=2)
            xattr = np.transpose(xattr, (0, 2, 1))
            x = np.concatenate((x, xattr), axis=2)

        data = x
        tLst = trange_to_array(config["tRange"])
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, config, varLst):
        if type(varLst) is str:
            varLst = [varLst]
        # inputfile = os.path.join(os.path.realpath(config['attr_path']))
        if self.inputfile_attr.endswith('.npy'):
            dfC = np.load(self.inputfile_attr)
        elif self.inputfile_attr.endswith('.pt'):
            dfC = torch.load(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()

        varLst_index_attr = []
        for var in varLst:
            if var in self.attrLst_name:
                varLst_index_attr.append(self.attrLst_name.index(var))
            else:
                print(var, "the var is not in forcing file nor in attr file")
                exit()
        c = dfC[:, varLst_index_attr]

        return c