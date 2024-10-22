"""All functions related to loading the data"""
import numpy as np
import torch
from core.load_data.normalizing import transNorm
from core.load_data.time import tRange2Array

# def load_df(args):
#     """
#     A function that loads the data into a
#     :return:
#     """
#     df, x, y, c, c_hydro_model, x_hydro_model, c_SNTEMP, x_SNTEMP = master.loadData(args)
#     nx = x.shape[-1] + c.shape[-1]
#     x_total = np.zeros((x.shape[0], x.shape[1], nx))
#     nx_SNTEMP = x_SNTEMP.shape[-1] + c_SNTEMP.shape[-1]
#     x_tot_SNTEMP = np.zeros((x.shape[0], x.shape[1], nx_SNTEMP))
#     ct = np.repeat(c, repeats=x.shape[1], axis=0)
#     for k in range(x.shape[0]):
#         x_total[k, :, :] = np.concatenate(
#             (x[k, :, :], np.tile(c[k], (x.shape[1], 1))), axis=1
#         )
#         x_tot_SNTEMP[k, :, :] = np.concatenate(
#             (x_SNTEMP[k, :, :], np.tile(c_SNTEMP[k], (x_SNTEMP.shape[1], 1))), axis=1
#         )
#
#
#     # streamflow values should not be negative
#     # vars = args['optData']['varT'] + args['optData']['varC']
#     # x_total[x_total[:, :, vars.index("00060_Mean")] < 0] = 0
#     return np.float32(x_total), np.float32(y), np.float32(c), np.float32(c_hydro_model), \
#         np.float32(x_hydro_model), np.float32(c_SNTEMP), np.float32(x_tot_SNTEMP)


def scaling(args, x, y, c):
    """
    creates our datasets
    :param set_name:
    :param args:
    :param time1:
    :param x_total_raw:
    :param y_total_raw:
    :return:  x, y, ngrid, nIterEp, nt
    """
    # initcamels(args, x, y)
    # Normalization
    x_total_scaled = transNorm(
        x, args["varT_NN"] + args["varC_NN"], toNorm=True
    )
    y_scaled = transNorm(y, args["target"], toNorm=True)
    c_scaled = transNorm(c, args["varC_NN"], toNorm=True)
    return x_total_scaled, y_scaled, c_scaled


def train_val_test_split(set_name, args, time1, x_total, y_total):
    t = tRange2Array(args[set_name])
    c, ind1, ind2 = np.intersect1d(time1, t, return_indices=True)
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]


    return x, y

def No_iter_nt_ngrid(set_name, args, x):
    nt, ngrid, nx = x.shape
    t = tRange2Array(args[set_name])
    if t.shape[0] < args["rho"]:
        rho = t.shape[0]
    else:
        rho = args["rho"]
    nIterEp = int(
        np.ceil(
            np.log(0.01)
            / np.log(1 - args["batch_size"] * rho / ngrid / (nt - args["warm_up"]))
        )
    )
    return ngrid, nIterEp, nt, args["batch_size"]

def train_val_test_split_action1(set_name, args, time1, x_total, y_total):
    t = tRange2Array(args[set_name])
    c, ind1, ind2 = np.intersect1d(time1, t, return_indices=True)
    x = x_total[:, ind1, :]
    y = y_total[:, ind1, :]
    ngrid, nt, nx = x.shape
    if t.shape[0] < args["rho"]:
        rho = t.shape[0]
    else:
        rho = args["rho"]


    return x, y, ngrid, nt, args["batch_size"]


def selectSubset(args, x, iGrid, iT, rho, *, c=None, tupleOut=False, has_grad=False, warm_up=0):
    nx = x.shape[-1]
    nt = x.shape[0]
    # if x.shape[0] == len(iGrid):   #hack
    #     iGrid = np.arange(0,len(iGrid))  # hack
    #     if nt <= rho:
    #         iT.fill(0)

    if iT is not None:
        batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho + warm_up, batchSize, nx], requires_grad=has_grad)
        for k in range(batchSize):
            temp = x[np.arange(iT[k] - warm_up, iT[k] + rho), iGrid[k] : iGrid[k] + 1, :]
            xTensor[:, k : k + 1, :] = torch.from_numpy(temp)
    else:
        if len(x.shape) == 2:
            # Used for local calibration kernel
            # x = Ngrid * Ntime
            xTensor = torch.from_numpy(x[iGrid, :]).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(x[:, iGrid, :]).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho + warm_up, axis=1)
        cTensor = torch.from_numpy(temp).float()

        if tupleOut:
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        # out = out.cuda()
        out = out.to(args["device"])
    return out


def randomIndex(ngrid, nt, dimSubset, warm_up=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+warm_up, nt - rho, [batchSize])
    return iGrid, iT


def create_tensor(rho, mini_batch, x, y):
    """
    Creates a data tensor of the input variables and incorporates a sliding window of rho
    :param mini_batch: min batch length
    :param rho: the seq len
    :param x: the x data
    :param y: the y data
    :return:
    """
    j = 0
    k = rho
    _sample_data_x = []
    _sample_data_y = []
    for i in range(x.shape[0]):
        _list_x = []
        _list_y = []
        while k < x[0].shape[0]:
            """In the format: [total basins, basin, days, attributes]"""
            _list_x.append(x[1, j:k, :])
            _list_y.append(y[1, j:k, 0])
            j += mini_batch
            k += mini_batch
        _sample_data_x.append(_list_x)
        _sample_data_y.append(_list_y)
        j = 0
        k = rho
    sample_data_x = torch.tensor(_sample_data_x).float()
    sample_data_y = torch.tensor(_sample_data_y).float()
    return sample_data_x, sample_data_y


def create_tensor_list(x, y):
    """
    we want to return the :
    x_list = [[[basin_1, num_samples_x, num_attr_x], [basin_1, num_samples_y, num_attr_y]]
        .
        .
        .
        [[basin_20, num_samples_x, num_attr_x], [basin_20, num_samples_y, num_attr_y]]]
    :param data:
    :return:
    """
    tensor_list = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            _var = (torch.tensor(x[i][j][:, :]), y[i, j])
            tensor_list.append(_var)
    return tensor_list

def take_sample_train(args, dataset_dictionary, ngrid_train, nt, batchSize):
    dimSubset = [batchSize, args["rho"]]
    iGrid, iT = randomIndex(ngrid_train, nt, dimSubset, warm_up=args["warm_up"])
    dataset_dictionary_sample = dict()
    dataset_dictionary_sample["iGrid"] = iGrid
    dataset_dictionary_sample["inputs_NN_scaled"] = selectSubset(args, dataset_dictionary["inputs_NN_scaled"],
                                                                        iGrid, iT, args["rho"], has_grad=False,
                                                                        warm_up=args["warm_up"])
    dataset_dictionary_sample["c_NN"] = torch.tensor(
        dataset_dictionary["c_NN"][iGrid], device=args["device"], dtype=torch.float32
    )
    # collecting observation samples
    dataset_dictionary_sample["obs"] = selectSubset(
        args, dataset_dictionary["obs"], iGrid, iT, args["rho"], has_grad=False, warm_up=args["warm_up"]
    )[args["warm_up"]:, :, :]
    # dataset_dictionary_sample["obs"] = converting_flow_from_ft3_per_sec_to_mm_per_day(args,
    #                                                                                          dataset_dictionary_sample[
    #                                                                                              "c_NN"],
    #                                                                                          obs_sample_v)
    # Hydro model sampling
    if args["hydro_model_name"] != "None":
        dataset_dictionary_sample["x_hydro_model"] = selectSubset(
            args, dataset_dictionary["x_hydro_model"], iGrid, iT, args["rho"], has_grad=False, warm_up=args["warm_up"]
        )
        dataset_dictionary_sample["c_hydro_model"] = torch.tensor(
            dataset_dictionary["c_hydro_model"][iGrid], device=args["device"], dtype=torch.float32
        )
    else:
        dataset_dictionary_sample["srflow"] = selectSubset(
            args, dataset_dictionary["srflow"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )
        dataset_dictionary_sample["ssflow"] = selectSubset(
            args, dataset_dictionary["ssflow"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )
        dataset_dictionary_sample["gwflow"] = selectSubset(
            args, dataset_dictionary["gwflow"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )
        dataset_dictionary_sample["bas_shallow"] = selectSubset(
            args, dataset_dictionary["bas_shallow"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )
        dataset_dictionary_sample["flow_sim"] = selectSubset(
            args, dataset_dictionary["flow_sim"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )
        dataset_dictionary_sample["PET_hydro"] = selectSubset(
            args, dataset_dictionary["PET_hydro"], iGrid, iT, args["rho"], has_grad=False, warm_up=0
        )

    return dataset_dictionary_sample



def take_sample_test(args, dataset_dictionary, iS, iE):
    dataset_dictionary_sample = dict()
    for key in dataset_dictionary.keys():
        if len(dataset_dictionary[key].shape) == 3:
            # we need to remove the warm up period for all except airT_memory and inputs for temp model
            if (key == "airT_mem_temp_model") or (key == "x_temp_model") or (key == "x_hydro_model") or (
                    key == "inputs_NN_scaled"):
                warm_up = 0
            else:
                warm_up = args["warm_up"]
            dataset_dictionary_sample[key] = dataset_dictionary[key][warm_up:, iS: iE, :].to(
                args["device"])
        elif len(dataset_dictionary[key].shape) == 2:
            dataset_dictionary_sample[key] = dataset_dictionary[key][iS: iE, :].to(
                args["device"])
    return dataset_dictionary_sample

def sub_Nans_for_mean(dataset_dictionary):
    for k in dataset_dictionary.keys():
        if k != "obs":  # for observations, we are allowed to have Nans
            data = dataset_dictionary[k].copy()
            if len(data.shape) == 3:    # for forcings
                for i in range(data.shape[2]):
                    data_temp = data[:, :, i].copy()
                    data_temp[np.isnan(data_temp)] = np.nanmean(data_temp)
                    data[:, :, i] = data_temp
            elif len(data.shape) == 2:   # for attributes
                for i in range(data.shape[1]):
                    data_temp = data[:, i].copy()
                    data_temp[np.isnan(data_temp)] = np.nanmean(data_temp)
                    data[:, i] = data_temp
            dataset_dictionary[k] = data
    return dataset_dictionary
# TODO add batch size into calculations here
