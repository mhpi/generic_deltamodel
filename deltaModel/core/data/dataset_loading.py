import json
import pickle

import numpy as np
import pandas as pd
from core.calc.normalize import init_norm_stats, trans_norm
from core.data.dataframe_dataset import DataFrameDataset
from core.data.array_dataset import ArrayDataset


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


def load_data(config, t_range=None, train=True):
    """ Load data into dictionaries for pNN and hydro model. """
    if t_range == None:
        t_range = config['t_range']

    out_dict = dict()

    if config['observations']['name'] in ['camels_671', 'camels_531']:
        if train:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attributes = pickle.load(f)
            
            startdate =config['train']['start_time']
            enddate = config['train']['end_time']
            
        else:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attributes = pickle.load(f)
        
            startdate =config['test']['start_time']
            enddate = config['test']['end_time']
            
        all_time = pd.date_range(config['observations']['start_date_all'], config['observations']['end_date_all'], freq='d')
        new_time = pd.date_range(startdate, enddate, freq='d')
        
        index_start = all_time.get_loc(new_time[0])
        index_end = all_time.get_loc(new_time[-1]) + 1

        # Subset forcings and attributes.
        attr_subset_idx = []
        for attr in config['dpl_model']['nn_model']['attributes']:
            if attr not in config['observations']['attributes_all']:
                raise ValueError(f"Attribute {attr} not in the list of all attributes.")
            attr_subset_idx.append(config['observations']['attributes_all'].index(attr))

        forcings = np.transpose(forcing[:,index_start:index_end], (1,0,2))
        forcing_subset_idx = []
        for forc in config['dpl_model']['nn_model']['forcings']:
            if forc not in config['observations']['forcings_all']:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            forcing_subset_idx.append(config['observations']['forcings_all'].index(forc))
        
        forcing_phy_subset_idx = []
        for forc in config['dpl_model']['phy_model']['forcings']:
            if forc not in config['observations']['forcings_all']:
                raise ValueError(f"Forcing {forc} not in the list of all forcings.")
            forcing_phy_subset_idx.append(config['observations']['forcings_all'].index(forc))

        out_dict['x_nn'] = forcings[:,:, forcing_subset_idx]  # Forcings for neural network (note, slight error from indexing)
        out_dict['x_phy'] = forcings[:,:, forcing_phy_subset_idx]  # Forcings for physics model
        out_dict['c_nn'] = attributes[:, attr_subset_idx] # Attributes
        out_dict['target'] = np.transpose(target[:,index_start:index_end], (1,0,2))  # Observation target
        
        ## For running a subset (531 basins) of CAMELS.
        if config['observations']['name'] == 'camels_531':
            gage_info = np.load(config['observations']['gage_info'])

            with open(config['observations']['subset_path'], 'r') as f:
                selected_camels = json.load(f)

            [C, Ind, subset_idx] = np.intersect1d(selected_camels, gage_info, return_indices=True)

            out_dict['x_nn'] = out_dict['x_nn'][:, subset_idx, :]
            out_dict['x_phy'] = out_dict['x_phy'][:, subset_idx, :]
            out_dict['c_nn'] = out_dict['c_nn'][subset_idx, :]
            out_dict['target'] = out_dict['target'][:, subset_idx, :]
            
    else:
        # Farshid data extractions
        forcing_dataset_class = choose_class_to_read_dataset(config, t_range, config['observations']['forcing_path'])
        out_dict['x_nn'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['dpl_model']['nn_model']['forcings'])
        out_dict['x_phy'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['phy_forcings'])
        out_dict['c_nn'] = forcing_dataset_class.read_data.getDataConst(config, varLst=config['dpl_model']['nn_model']['attributes'])
        out_dict['target'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['train']['target'])
    
    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(config, c_NN_sample, obs_sample):
    varTar_NN = config['train']['target']
    if '00060_Mean' in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index('00060_Mean')]
        varC_NN = config['dpl_model']['nn_model']['attributes']
        area_name = config['observations']['area_name']
        
        c_area = c_NN_sample[:, varC_NN.index(area_name)]
        area = np.expand_dims(c_area, axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index('00060_Mean')] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample


def get_dataset_dict(config, train=False):
    """
    Create dictionary of datasets used by the models.
    Contains 'c_nn', 'target', 'x_phy', 'x_nn_scaled'.

    train: bool, specifies whether data is for training.
    """

    # Create stats for NN input normalizations.
    if train: 
        dataset_dict = load_data(config, config['train_t_range'])
        init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                              dataset_dict['target'])
    else:
        dataset_dict = load_data(config, config['test_t_range'], train=False)

    # Normalization
    x_nn_scaled = trans_norm(config, np.swapaxes(dataset_dict['x_nn'], 1, 0).copy(),
                             var_lst=config['dpl_model']['nn_model']['forcings'], to_norm=True)
    x_nn_scaled[x_nn_scaled != x_nn_scaled] = 0  # Remove nans

    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['dpl_model']['nn_model']['attributes'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_scaled[c_nn_scaled != c_nn_scaled] = 0  # Remove nans
    c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)

    dataset_dict['x_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
    
    # Streamflow unit conversion.
    #### MOVED FROM LOAD_DATA
    if '00060_Mean' in config['train']['target']:
        dataset_dict['target'] = converting_flow_from_ft3_per_sec_to_mm_per_day(
            config,
            dataset_dict['c_nn'],
            dataset_dict['target']
        )

    return dataset_dict
