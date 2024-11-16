"""TODO: convert to pytorch.dataset class format."""
import logging
import os

import numpy as np
import pandas as pd
import torch
from core.data import BaseDataset
from core.utils.time import trange_to_array

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
