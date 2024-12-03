"""TODO: convert to pytorch.dataset class format."""
import logging
import os

import numpy as np
import pandas as pd
from core.data import BaseDataset, trange_to_array

log = logging.getLogger(__name__)


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
