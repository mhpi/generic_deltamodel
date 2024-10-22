import os
import numpy as np
import json
from core.load_data.dataFrame_loading import (
    loadData
)

def calStatbasinnorm(
    y, c, args
):  # for daily streamflow normalized by basin area and precipitation
    """

    :param y: streamflow data to be normalized
    :param x: x is forcing+attr numpy matrix
    :param args: config file
    :return: statistics to be used for flow normalization
    """
    y[y == (-999)] = np.nan
    y[y < 0] = 0
    attr_list = args["varC_NN"]
    # attr_data = read_attr_data(args, idLst=idLst)
    if "DRAIN_SQKM" in attr_list:
        area_name = "DRAIN_SQKM"
    elif "area_gages2" in attr_list:
        area_name = "area_gages2"
    basinarea = c[:, attr_list.index(area_name)]  #  'DRAIN_SQKM'
    if "PPTAVG_BASIN" in attr_list:
        p_mean_name = "PPTAVG_BASIN"
    elif "p_mean" in attr_list:
        p_mean_name = "p_mean"
    meanprep = c[:, attr_list.index(p_mean_name)]  #   'PPTAVG_BASIN'
    temparea = np.repeat(np.expand_dims(basinarea, axis=(1,2)), y.shape[0]).reshape(y.shape)
    tempprep = np.repeat(np.expand_dims(meanprep, axis=(1, 2)), y.shape[0]).reshape(y.shape)
    flowua = (y * 0.0283168 * 3600 * 24) / (
        (temparea * (10**6)) * (tempprep * 10 ** (-2)) / 365
    )  # unit (m^3/day)/(m^3/day)
    a = flowua.flatten()
    b = a[~np.isnan(a)]  # kick out Nan
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics plus 0.1 for 0 values
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStatgamma(x):  # for daily streamflow and precipitation
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    b = np.log10(
        np.sqrt(b) + 0.1
    )  # do some tranformation to change gamma characteristics
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]

def calStat(x):
    a = x.flatten()
    bb = a[~np.isnan(a)]  # kick out Nan
    b = bb[bb != (-999999)]
    p10 = np.percentile(b, 10).astype(float)
    p90 = np.percentile(b, 90).astype(float)
    mean = np.mean(b).astype(float)
    std = np.std(b).astype(float)
    if std < 0.001:
        std = 1
    return [p10, p90, mean, std]
def calStatAll(args, x, c, y):
    statDict = dict()
    # target
    for i, target_name in enumerate(args["target"]):
        # calculating especialized statistics for streamflow
        if target_name == "00060_Mean":
            statDict[args["target"][i]] = calStatbasinnorm(y[:, :, i: i+1], c, args)
        else:
            statDict[args["target"][i]] = calStat(y[:, :, i: i+1])

    # forcing
    varList = args["varT_NN"]
    for k in range(len(varList)):
        var = varList[k]
        # if var == "prcp(mm/day)":
        #     statDict[var] = calStatgamma(x[:, :, k])
        if (var == "00060_Mean") or (var == "combine_discharge"):
            statDict[var] = calStatbasinnorm(x[:, :, k: k + 1], x, args)
        else:
            statDict[var] = calStat(x[:, :, k])
    # attributes
    varList = args["varC_NN"]
    for k, var in enumerate(varList):
        statDict[var] = calStat(c[:, k])

    statFile = os.path.join(args["out_dir"], "Statistics_basinnorm.json")
    with open(statFile, "w") as fp:
        json.dump(statDict, fp, indent=4)


def transNorm(args, x, varLst, *, toNorm):
    statFile = os.path.join(args["out_dir"], "Statistics_basinnorm.json")
    with open(statFile, "r") as fp:
        statDict = json.load(fp)
    if type(varLst) is str:
        varLst = [varLst]
    out = np.zeros(x.shape)
    x_temp = x.copy()
    for k in range(len(varLst)):
        var = varLst[k]

        stat = statDict[var]
        if toNorm is True:
            if len(x.shape) == 3:
                if (
                    # var == "prcp(mm/day)"
                    var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    x_temp[:, :, k] = np.log10(np.sqrt(x_temp[:, :, k]) + 0.1)

                out[:, :, k] = (x_temp[:, :, k] - stat[2]) / stat[3]
            elif len(x.shape) == 2:
                if (
                    # var == "prcp(mm/day)"
                    var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    x_temp[:, k] = np.log10(np.sqrt(x_temp[:, k]) + 0.1)
                out[:, k] = (x_temp[:, k] - stat[2]) / stat[3]
        else:
            if len(x.shape) == 3:
                out[:, :, k] = x_temp[:, :, k] * stat[3] + stat[2]
                if (
                    # var == "prcp(mm/day)"
                    var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2

            elif len(x.shape) == 2:
                out[:, k] = x_temp[:, k] * stat[3] + stat[2]
                if (
                    # var == "prcp(mm/day)"
                    var == "00060_Mean"
                    or var == "combine_discharge"
                ):
                    out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2

    return out
def init_norm_stats(args, x_NN, c_NN, y):
    stats_directory = args["out_dir"]
    statFile = os.path.join(stats_directory, "Statistics_basinnorm.json")

    if not os.path.isfile(statFile):
        # read all data in training for just the inputs used in NN
        # x_NN, c_NN, y = loadData(args, trange=args["t_train"], data=["x_NN", "c_NN", "y"])
        # calculate the stats
        calStatAll(args, x_NN, c_NN, y)
    # with open(statFile, "r") as fp:
    #     statDict = json.load(fp)
