import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import pickle
import zarr
import json

from core.utils.Dates import Dates
from core.utils.time import trange_to_array
from core.calc.normalize import init_norm_stats, trans_norm



class Data_Reader(ABC):
    @abstractmethod
    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError


class DataFrame_dataset(Data_Reader):
    def __init__(self, args, tRange, data_path, attr_path=None):
        self.time = trange_to_array(tRange)
        self.args = args
        self.inputfile = data_path
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.args['observations']['attr_path']))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data

    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
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
        tLst = trange_to_array(args['t_range'])
        tLstobs = trange_to_array(args['t_range'])
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

    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        
        if 'geol_porosity' in varLst:
            # correct typo
            varLst[varLst.index('geol_porosity')] = 'geol_porostiy'

        inputfile = os.path.join(os.path.realpath(args['observations']['forcing_path']))
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

        data = c
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return data


class numpy_dataset(Data_Reader):
    def __init__(self, args, tRange, data_path, attr_path=None):
        self.time = trange_to_array(tRange)
        self.args = args
        self.inputfile = data_path   # the dynamic data
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.args['observations']['attr_path']))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data
        # These are default forcings and attributes that are read from the dataset
        self.all_forcings_name = ['Lwd', 'PET_hargreaves(mm/day)', 'prcp(mm/day)',
                                'Pres', 'RelHum', 'SpecHum', 'srad(W/m2)',
                                'tmean(C)', 'tmax(C)', 'tmin(C)', 'Wind', 'ccov',
                                'vp(Pa)', '00060_Mean', '00010_Mean','dayl(s)']  #
        self.attrLst_name = ['aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT', 'SoilGrids1km_sand',
                             'SoilGrids1km_clay', 'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel',
                             'HWSD_sand', 'HWSD_silt', 'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost',
                             'permeability','seasonality_P', 'seasonality_PET', 'snow_fraction',
                             'snowfall_fraction','T_clay','T_gravel','T_sand', 'T_silt','Porosity',
                             'DRAIN_SQKM', 'lat', 'site_no_int', 'stream_length_square', 'lon']

    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
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
        tLst = trange_to_array(args["tRange"])
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        # inputfile = os.path.join(os.path.realpath(args['attr_path']))
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

        data = c
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return data


class choose_class_to_read_dataset():
    def __init__(self, args, trange, data_path):
        self.args = args
        self.trange = trange
        self.data_path = data_path
        self._get_dataset_class()
        
    def _get_dataset_class(self) -> None:
        if self.data_path.endswith(".feather") or self.data_path.endswith(".csv"):
            self.read_data = DataFrame_dataset(args=self.args, tRange=self.trange, data_path=self.data_path)
        elif self.data_path.endswith(".npy") or self.data_path.endswith(".pt"):
            self.read_data = numpy_dataset(args=self.args, tRange=self.trange, data_path=self.data_path)


def load_data(config, t_range=None, train=True):
    """
    Load data into dictionaries for pNN and hydro model.
    """
    if t_range == None:
        t_range = config['t_range']

    out_dict = dict()

    if config['observations']['name'] in ['camels_671_yalan', 'camels_531_yalan']:
        if train:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attr = pickle.load(f)
            
            startYear = str(config['train_t_range'][0])[:4]
            endYear = str(config['train_t_range'][1])[:4]
            
        else:
            with open(config['observations']['train_path'], 'rb') as f:
                forcing, target, attr = pickle.load(f)
            
            startYear = str(config['test_t_range'][0])[:4]
            endYear = str(config['test_t_range'][1])[:4]
        
        AllTime = pd.date_range('1980-10-01', f'2014-09-30', freq='d')
        newTime = pd.date_range(f'{startYear}-10-01', f'{endYear}-09-30', freq='d')
        
        index_start = AllTime.get_loc(newTime[0])
        index_end = AllTime.get_loc(newTime[-1]) + 1

        out_dict['x_nn'] = np.transpose(forcing[:,index_start:index_end], (1,0,2))  # Forcings
        out_dict['c_nn'] = attr # Attributes
        out_dict['obs'] = np.transpose(target[:,index_start:index_end], (1,0,2))  # Observation target
        
        ## For running a subset (531 basins) of CAMELS.
        if config['observations']['name'] == 'camels_531_yalan':
            gage_info = np.load(config['observations']['gage_info'])

            with open(config['observations']['subset_path'], 'r') as f:
                selected_camels = json.load(f)

            [C, Ind, subset_idx] = np.intersect1d(selected_camels, gage_info, return_indices=True)

            out_dict['x_nn'] = out_dict['x_nn'][:, subset_idx, :]
            out_dict['c_nn'] = out_dict['c_nn'][subset_idx, :]
            out_dict['obs'] = out_dict['obs'][:, subset_idx, :]
            
        out_dict['x_hydro_model'] = out_dict['x_nn']
        out_dict['c_hydro_model'] = out_dict['c_nn']  # just a placeholder.

    else:
        # Original data handling for Farshid's extractions.
        forcing_dataset_class = choose_class_to_read_dataset(config, t_range, config['observations']['forcing_path'])
        # getting inputs for neural network:
        out_dict['x_nn'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['observations']['var_t_nn'])
        out_dict['c_nn'] = forcing_dataset_class.read_data.getDataConst(config, varLst=config['observations']['var_c_nn'])
        out_dict['obs'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['target'])

        out_dict['x_hydro_model'] = forcing_dataset_class.read_data.getDataTs(config, varLst=config['observations']['var_t_hydro_model'])
        out_dict['c_hydro_model'] = forcing_dataset_class.read_data.getDataConst(config, varLst=config['observations']['var_c_hydro_model'])
    
    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(config, c_NN_sample, obs_sample):
    varTar_NN = config['target']
    if '00060_Mean' in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index('00060_Mean')]
        varC_NN = config['observations']['var_c_nn']
        area_name = config['observations']['area_name']
        
        c_area = c_NN_sample[:, varC_NN.index(area_name)]
        area = np.expand_dims(c_area, axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index('00060_Mean')] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample


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

    # Create stats for NN input normalizations.
    if train: 
        dataset_dict = load_data(config, config['train_t_range'])
        init_norm_stats(config, dataset_dict['x_nn'], dataset_dict['c_nn'],
                              dataset_dict['obs'])
    else:
        dataset_dict = load_data(config, config['test_t_range'], train=False)

    # Normalization
    x_nn_scaled = trans_norm(config, np.swapaxes(dataset_dict['x_nn'], 1, 0).copy(),
                             var_lst=config['observations']['var_t_nn'], to_norm=True)
    x_nn_scaled[x_nn_scaled != x_nn_scaled] = 0  # Remove nans

    c_nn_scaled = trans_norm(config, dataset_dict['c_nn'],
                             var_lst=config['observations']['var_c_nn'], to_norm=True) ## NOTE: swap axes to match Yalan's HBV. This affects calculations...
    c_nn_scaled[c_nn_scaled != c_nn_scaled] = 0  # Remove nans
    c_nn_scaled = np.repeat(np.expand_dims(c_nn_scaled, 0), x_nn_scaled.shape[0], axis=0)

    dataset_dict['inputs_nn_scaled'] = np.concatenate((x_nn_scaled, c_nn_scaled), axis=2)
    del x_nn_scaled, c_nn_scaled, dataset_dict['x_nn']
    
    # Streamflow unit conversion.
    #### MOVED FROM LOAD_DATA
    if '00060_Mean' in config['target']:
        dataset_dict['obs'] = converting_flow_from_ft3_per_sec_to_mm_per_day(
            config,
            dataset_dict['c_nn'],
            dataset_dict['obs']
        )

    return dataset_dict, config


def extract_data(config):
    """
    Extract forcings and attributes from dataset feather files.
    """
    # Get date range.
    config['train_t_range'] = Dates(config['train'], config['rho']).date_to_int()
    config['test_t_range'] = Dates(config['test'], config['rho']).date_to_int()
    config['t_range'] = [config['train_t_range'][0], config['test_t_range'][1]]
    
    # Lists of forcings and attributes for nn + physics model.
    forcing_list = list(
        dict.fromkeys(config['observations']['var_t_nn'] + config['observations']['var_t_hydro_model'])
        )
    attribute_list = list(
        dict.fromkeys(config['observations']['var_c_nn'] + config['observations']['var_c_hydro_model'])
        )
     
    out_dict = {}
    forcing_dataset_class = choose_class_to_read_dataset(config, config['test_t_range'], config['observations']['forcing_path'])

    forcing_dat = forcing_dataset_class.read_data.getDataTs(config, varLst=forcing_list)
    attribute_dat = forcing_dataset_class.read_data.getDataConst(config, varLst=attribute_list)

    out_dict['x_all'] = {key: forcing_dat[:,:, i] for i, key in enumerate(forcing_list)}
    out_dict['c_all'] = {key: attribute_dat[:, i] for i, key in enumerate(attribute_list)}
    # obs_raw = forcing_dataset_class.read_data.getDataTs(config, varLst=config['target'])

    return out_dict
