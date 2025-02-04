import json
import pickle
import json
import zarr
import numpy as np
import pandas as pd
from core.calc.normalize import init_norm_stats, trans_norm,fill_Nan
from core.data.array_dataset import ArrayDataset
from core.data.dataframe_dataset import DataFrameDataset


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

    if config['observations']['name'] in ['camels_671', 'camels_531','MERIT_forward_zone']:
        if train:
            startdate =config['train']['start_time']
            enddate = config['train']['end_time']
            
        else:
            startdate =config['test']['start_time']
            enddate = config['test']['end_time']

        all_time = pd.date_range(config['observations']['start_date_all'], config['observations']['end_date_all'], freq='d')
        new_time = pd.date_range(startdate, enddate, freq='d')
        
        index_start = all_time.get_loc(new_time[0])
        index_end = all_time.get_loc(new_time[-1]) + 1

        subbasin_data_path = config['observations']['subbasin_data_path']
        subbasin_ID_name = config['observations']['subbasin_ID_name']
        root_zone = zarr.open_group(subbasin_data_path, mode = 'r')
        subbasin_ID_all = np.array(root_zone[subbasin_ID_name][:]).astype(int)
        
        for attr_idx, attr in enumerate(config['dpl_model']['nn_model']['attributes']) :
            if attr not in config['observations']['attributes_all']:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            if attr_idx == 0:
                attr_array = np.expand_dims(root_zone['attrs'][attr][:],-1) 
            else:
                attr_array = np.concatenate((attr_array,np.expand_dims(root_zone['attrs'][attr][:],-1) ),axis = -1)

        try:
            Ac_name = config['observations']['Upstream_area_name'] 
        except:
            raise ValueError(f"Upstream area is not provided. This is needed for high-resolution streamflow model")

        try:
            Elevation_name = config['observations']['Elevation_name'] 
        except:
            raise ValueError(f"Elevation is not provided. This is needed for high-resolution streamflow model")


        Ac_array = root_zone['attrs'][Ac_name][:]
 
        Ele_array = root_zone['attrs'][Elevation_name][:]

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
    Contains 'c_nn', 'target', 'x_phy', 'x_nn_scaled'.

    train: bool, specifies whether data is for training.
    """

    # Create stats for NN input normalizations.

    dataset_dict = load_data_subbasin(config, train=False)

    # Normalization
    x_nn_scaled = trans_norm(config, dataset_dict['x_nn'],
                             var_lst=config['dpl_model']['nn_model']['forcings'], to_norm=True)
    x_nn_scaled[x_nn_scaled != x_nn_scaled] = 0  # Remove nans

    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['dpl_model']['nn_model']['attributes'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_scaled[c_nn_scaled != c_nn_scaled] = 0  # Remove nans
    c_nn_scaled_repeat = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)

    dataset_dict['x_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled_repeat), axis=2)

    dataset_dict['c_nn_scaled'] = c_nn_scaled
    x_phy = np.swapaxes(dataset_dict['x_phy'], 1, 0)
    x_phy[x_phy != x_phy] = 0 
    dataset_dict['x_phy'] =  x_phy
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
    
    # Streamflow unit conversion.
    #### MOVED FROM LOAD_DATA
    # if 'flow_sim' in config['train']['target']:
    #     dataset_dict['target'] = converting_flow_from_ft3_per_sec_to_mm_per_day(
    #         config,
    #         dataset_dict['c_nn'],
    #         dataset_dict['target']
    #     )

    return dataset_dict
