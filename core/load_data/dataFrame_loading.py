import json
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
import torch
from core.load_data.time import tRange2Array
from datetime import datetime, timedelta

class Data_Reader(ABC):
    @abstractmethod
    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError

    @abstractmethod
    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        raise NotImplementedError


class DataFrame_dataset(Data_Reader):
    def __init__(self, args, tRange, data_path, attr_path=None):
        self.time = tRange2Array(tRange)
        self.args = args
        self.inputfile = data_path
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.args["attr_path"]))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data

    def getDataTs(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]

        if self.inputfile.endswith(".csv"):
            dfMain = pd.read_csv(self.inputfile)
            dfMain_attr = pd.read_csv(self.inputfile_attr)
        elif self.inputfile.endswith(".feather"):
            dfMain = pd.read_feather(self.inputfile)
            dfMain_attr = pd.read_feather(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain["site_no"].unique()
        tLst = tRange2Array(args["tRange"])
        tLstobs = tRange2Array(args["tRange"])
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
        g = dfMain.reset_index(drop=True).groupby("site_no")
        xtg = [xt[i.values, :] for k, i in g.groups.items()]
        x = np.array(xtg)

        # TODO: Ths part is commented because I don't think we need to have it here as there ais a different
        #  function to read static inputs
        ## for attr
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
        inputfile = os.path.join(os.path.realpath(args["forcing_path"]))
        if self.inputfile_attr.endswith(".csv"):
            dfMain = pd.read_csv(self.inputfile)
            dfC = pd.read_csv(self.inputfile_attr)
        elif inputfile.endswith(".feather"):
            dfMain = pd.read_feather(self.inputfile)
            dfC = pd.read_feather(self.inputfile_attr)
        else:
            print("data type is not supported")
            exit()
        sites = dfMain["site_no"].unique()
        nNodes = len(sites)
        c = np.empty([nNodes, len(varLst)])

        for k, kk in enumerate(sites):
            data = dfC.loc[dfC["site_no"] == kk, varLst].to_numpy().squeeze()
            c[k, :] = data

        data = c
        # if doNorm is True:
        #     data = transNorm(data, varLst, toNorm=True)
        # if rmNan is True:
        #     data[np.where(np.isnan(data))] = 0
        return data

class numpy_dataset(Data_Reader):
    def __init__(self, args, tRange, data_path, attr_path=None):
        self.time = tRange2Array(tRange)
        self.args = args
        self.inputfile = data_path   # the dynamic data
        if attr_path == None:
            self.inputfile_attr = os.path.join(os.path.realpath(self.args["attr_path"]))  # the static data
        else:
            self.inputfile_attr = os.path.join(os.path.realpath(attr_path))  # the static data
        # These are default forcings and attributes that are read from the dataset
        ## getting the forcings and attrs names
        self.all_forcings_name, self.attrLst_name = self.get_forcing_attr_names(args)
        # self.all_forcings_name = ['Lwd', 'PET_hargreaves(mm/day)', 'prcp(mm/day)',
        #                         'Pres', 'RelHum', 'SpecHum', 'srad(W/m2)',
        #                         'tmean(C)', 'tmax(C)', 'tmin(C)', 'Wind', 'ccov',
        #                         'vp(Pa)', "00060_Mean", "00010_Mean",'dayl(s)']  #
        # self.attrLst_name = ['aridity', 'p_mean', 'ETPOT_Hargr', 'NDVI', 'FW', 'SLOPE_PCT', 'SoilGrids1km_sand',
        #                      'SoilGrids1km_clay', 'SoilGrids1km_silt', 'glaciers', 'HWSD_clay', 'HWSD_gravel',
        #                      'HWSD_sand', 'HWSD_silt', 'ELEV_MEAN_M_BASIN', 'meanTa', 'permafrost',
        #                      'permeability','seasonality_P', 'seasonality_PET', 'snow_fraction',
        #                      'snowfall_fraction','T_clay','T_gravel','T_sand', 'T_silt','Porosity',
        #                      "DRAIN_SQKM", "lat", "site_no_int", "stream_length_square", "lon"]
    def get_forcing_attr_names(self, args):
        # forcing
        forcing_file_name = os.path.basename(args["forcing_path"])
        forcing_name = forcing_file_name.split(".npy")[0] + "_name.json"
        forcing_name_path = os.path.join(os.path.dirname(args["forcing_path"]), forcing_name)
        with open(forcing_name_path, "r") as json_file:
            forcing_name = json.load(json_file)
        # attr
        attr_file_name = os.path.basename(args["attr_path"])
        attr_name = attr_file_name.split(".npy")[0] + "_name.json"
        attr_name_path = os.path.join(os.path.dirname(args["attr_path"]), attr_name)
        with open(attr_name_path, "r") as json_file:
            attr_name = json.load(json_file)
        return forcing_name, attr_name
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
        tLst = tRange2Array(args["tRange"])
        C, ind1, ind2 = np.intersect1d(self.time, tLst, return_indices=True)
        data = data[:, ind2, :]
        return np.swapaxes(data, 1, 0)

    def getDataConst(self, args, varLst, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        # inputfile = os.path.join(os.path.realpath(args["attr_path"]))
        if self.inputfile_attr.endswith(".npy"):
            dfC = np.load(self.inputfile_attr)
        elif self.inputfile_attr.endswith(".pt"):
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
        self.get_dataset_class()
    def get_dataset_class(self) -> None:
        if self.data_path.endswith(".feather") or self.data_path.endswith(".csv"):
            self.read_data = DataFrame_dataset(args=self.args, tRange=self.trange, data_path=self.data_path)
        elif self.data_path.endswith(".npy") or self.data_path.endswith(".pt"):
            self.read_data = numpy_dataset(args=self.args, tRange=self.trange, data_path=self.data_path)







def loadData(args, trange):

    out_dict = dict()
    forcing_dataset_class = choose_class_to_read_dataset(args, trange, args["forcing_path"])
    # getting inputs for NN model:
    out_dict["x_NN"] = forcing_dataset_class.read_data.getDataTs(args, varLst=args["varT_NN"])
    out_dict["c_NN"] = forcing_dataset_class.read_data.getDataConst(args, varLst=args["varC_NN"])
    obs_raw = forcing_dataset_class.read_data.getDataTs(args, varLst=args["target"])
    ## converting the
    if "00060_Mean" in args["target"]:
        out_dict["obs"] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
                                                                         out_dict["c_NN"],
                                                                         obs_raw)
    else:
        out_dict["obs"] = obs_raw
    if args["hydro_model_name"] != "None":
        out_dict["x_hydro_model"] = forcing_dataset_class.read_data.getDataTs(args, varLst=args["varT_hydro_model"])
        out_dict["c_hydro_model"] = forcing_dataset_class.read_data.getDataConst(args, varLst=args["varC_hydro_model"])
    ## if it is None --> we read the flow from a dataset
    else:
        # defining a new dataset class to read the flow data
        flow_dataset_class = choose_class_to_read_dataset(args, trange, args["flow_data_path"])
        # this is the flowdata columns' names. it is
        flow_dataset_class.read_data.all_forcings_name = ["srflow", "ssflow", "bas_shallow", "gwflow", "flow_sim"]
        for col in flow_dataset_class.read_data.all_forcings_name:
            out_dict[col] = flow_dataset_class.read_data.getDataTs(args, varLst=[col])
        # reading PET from forcings
        out_dict["PET_hydro"] = forcing_dataset_class.read_data.getDataTs(args, varLst=["PET_hargreaves(mm/day)"])

    return out_dict


def converting_flow_from_ft3_per_sec_to_mm_per_day(args, c_NN_sample, obs_sample):
    varTar_NN = args["target"]
    if "00060_Mean" in varTar_NN:
        obs_flow_v = obs_sample[:, :, varTar_NN.index("00060_Mean")]
        varC_NN = args["varC_NN"]
        if "DRAIN_SQKM" in varC_NN:
            area_name = "DRAIN_SQKM"
        elif "area_gages2" in varC_NN:
            area_name = "area_gages2"
        # area = (c_NN_sample[:, varC_NN.index(area_name)]).unsqueeze(0).repeat(obs_flow_v.shape[0], 1)  # torch version
        area = np.expand_dims(c_NN_sample[:, varC_NN.index(area_name)], axis=0).repeat(obs_flow_v.shape[0], 0)  # np ver
        obs_sample[:, :, varTar_NN.index("00060_Mean")] = (10 ** 3) * obs_flow_v * 0.0283168 * 3600 * 24 / (area * (10 ** 6)) # convert ft3/s to mm/day
    return obs_sample