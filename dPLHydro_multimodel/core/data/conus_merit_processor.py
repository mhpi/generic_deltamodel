"""
Code to support the processing of CONUS files in non-feather formats until full
support is offered in the PMI.

Adapted from Yalan Song 2024.
"""
import numpy as np
import pandas as pd
import json
import zarr
import pickle

from core.utils.Dates import Dates
from core.data.dataset_loading import (
    init_norm_stats,
    trans_norm
    )
from core.calc.normalize import basin_norm



def load_gages_merit(config, t_range=None):
    """
    Load data into dictionaries for pNN and hydro model.

    NOTE: This is a temporary addition to validate HBV2.0 models.
    """
    if t_range == None:
        t_range = config['t_range']

    out_dict = dict()

    # data_dir = config['observations']['data_dir']
    # with open(data_dir + 'train_data_dict.json') as f:
    #     train_data_dict = pickle.load(f)

    startYear = str(config['train_t_range'][0])[:4]
    endYear = str(config['train_t_range'][1])[:4]
    AllTime = pd.date_range('1980-01-01', f'2020-12-31', freq='d')
    newTime = pd.date_range(f'{startYear}-10-01', f'{endYear}-09-30', freq='d')

    # Load shape id list, target streamflow observations, attributes, forcings,
    # and merit gage info.
    shape_id_lst = np.load(config['observations']['shape_id_path'])
    streamflow = np.load(config['observations']['data_dir'] + 'train_flow.npy')

    attributes = pd.read_csv(config['observations']['attr_path'],index_col=0)
    attributes = attributes.sort_values(by='id')

    merit_gage_info = pd.read_csv(config['observations']['gage_info_path']) 
    merit_gage_info = merit_gage_info.sort_values(by='STAID')
    gage_ids_from_merit = merit_gage_info['STAID'].values    

    # Only keep data for gages we want.
    attributes = attributes[attributes['id'].isin(gage_ids_from_merit)]

    # lat = attributes["lat"].values
    id_list_new = attributes["id"].values
    id_list_old = [int(id) for id in shape_id_lst]
    [c, ind_1, sub_ind] = np.intersect1d(id_list_new, id_list_old, return_indices=True)

    # Only keep data for gages we want.
    streamflow = streamflow[sub_ind,:,:]
    if(not (id_list_new==np.array(id_list_old)[sub_ind]).all()):
        raise Exception("Ids of subset gage do not match with ids in the attribute file.")
    
    key_info = [str(x) for x in id_list_new]

    with open(config['observations']['area_info_path']) as f:
        area_info = json.load(f)
    
    merit_save_path = config['observations']['merit_path']

    with open(config['observations']['merit_idx_path']) as f:
        merit_idx = json.load(f)

    root_zone = zarr.open_group(merit_save_path, mode = 'r')
    merit_all = root_zone['COMID'][:]

    xTrain2 = np.full((len(merit_all),len(newTime),len(config['observations']['var_t_nn'])),np.nan)
    attr2 = np.full((len(merit_all),len(config['observations']['var_c_nn'])),np.nan)

    merit_time = pd.date_range('1980-10-01',f'2010-09-30', freq='d')
    merit_start_idx = merit_time.get_loc(newTime[0])
    merit_end_idx = merit_time.get_loc(newTime[-1])+1

    for fid, forcing_ in enumerate(config['observations']['var_t_nn']):
        xTrain2[:,:,fid] = root_zone[forcing_][:,merit_start_idx:merit_end_idx]

    for aid, attribute_ in enumerate(config['observations']['var_c_nn']):
        attr2[:,aid] =  root_zone['attr'][attribute_][:]



    # Adapting HBV2.0 data prep to PMI framework.
    out_dict['x_nn'] = np.transpose(xTrain2, (1,0,2))  # Forcings
    out_dict['c_nn_all'] = attributes  # Attributes
    out_dict['c_nn'] = attr2  # Attributes
    out_dict['obs'] = np.transpose(streamflow, (1,0,2))  # Observations

    out_dict['ac_all'] = root_zone['attr']["uparea"][:] 
    out_dict['ai_all'] = root_zone['attr']["catchsize"][:]
    out_dict['merit_all'] = merit_all
    out_dict['merit_idx'] = merit_idx


    out_dict['gage_key'] = key_info
    out_dict['area_info'] = area_info

    out_dict['x_hydro_model'] = out_dict['x_nn']
    out_dict['c_hydro_model'] = out_dict['c_nn']  # just a placeholder.

    return out_dict


def get_data_dict(config, train=False):
    """
    Create dictionary of datasets used by the models.
    Contains 'c_nn', 'obs', 'x_hydro_model', 'c_hydro_model', 'inputs_nn_scaled'.

    train: bool, specifies whether data is for training.
    """
    # Get date range
    config['train_t_range'] = Dates(config['train'], config['rho']).date_to_int()
    config['test_t_range'] = Dates(config['test'], config['rho']).date_to_int()
    config['t_range'] = [config['train_t_range'][0], config['test_t_range'][1]]

    # Get data and initialize normalization statistics.
    if train: 
        dataset_dict = load_gages_merit(config, config['train_t_range'])
        init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                              dataset_dict['obs'], dataset_dict['c_nn_all'])
        del dataset_dict['merit_all']
    else:
        dataset_dict = load_gages_merit(config, config['test_t_range'])    
    
    # Normalization
    x_nn_scaled = trans_norm(config, np.swapaxes(dataset_dict['x_nn'], 1, 0).copy(),
                             var_lst=config['observations']['var_t_nn'], to_norm=True)
    x_nn_scaled[x_nn_scaled != x_nn_scaled] = 0  # Remove nans

    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['observations']['var_c_nn'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_scaled[c_nn_scaled != c_nn_scaled] = 0  # Remove nans

    all_attrs = dataset_dict['c_nn_all']
    basin_area = np.expand_dims(all_attrs["area"].values,axis = 1)
    obs_scaled = basin_norm(np.transpose(dataset_dict['obs'], (1,0,2)),
                            basin_area, to_norm=True)

    dataset_dict['inputs_nn_scaled'] = np.concatenate((
        x_nn_scaled, 
        np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)), axis=2)
    dataset_dict['x_nn_scaled'] = x_nn_scaled
    dataset_dict['c_nn_scaled'] = c_nn_scaled
    dataset_dict['obs'] = np.transpose(obs_scaled, (1,0,2))
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']

    return dataset_dict, config
