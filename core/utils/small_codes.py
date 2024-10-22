import torch
import os
from core.load_data.time import tRange2Array
import json


def make_tensor(*values, has_grad=False, dtype=torch.float32, device="cuda"):

    if len(values) > 1:
        tensor_list = []
        for value in values:
            t = torch.tensor(value, requires_grad=has_grad, dtype=dtype, device=device)
            tensor_list.append(t)
    else:
        for value in values:
            if type(value) != torch.Tensor:
                tensor_list = torch.tensor(
                    value, requires_grad=has_grad, dtype=dtype, device=device
                )
            else:
                tensor_list = value.clone().detach()
    return tensor_list


def create_output_dirs(args):
    seed = args["randomseed"][0]
    # checking rho value first
    t = tRange2Array(args["t_train"])
    if t.shape[0] < args["rho"]:
        args["rho"] = t.shape[0]

    # checking the directory
    if not os.path.exists(args["output_model"]):
        os.makedirs(args["output_model"])
    if args["hydro_model_name"]!= "None":
        hydro_name = "_" + args["hydro_model_name"]
    else:
        hydro_name = ""

    out_folder = args["NN_model_name"] + \
            hydro_name + \
            '_E' + str(args['EPOCHS']) + \
             '_R' + str(args['rho']) + \
             '_B' + str(args['batch_size']) + \
             '_H' + str(args['hidden_size']) + \
             "_tr" + str(args["t_train"][0])[:4] + "_" + str(args["t_train"][1])[:4] + \
            "_n" + str(args["nmul"]) + \
            "_" + str(seed)

    if not os.path.exists(os.path.join(args["output_model"], out_folder)):
        os.makedirs(os.path.join(args["output_model"], out_folder))

    ## make a folder for static and dynamic parametrization
    dyn_params = ""
    if args["hydro_model_name"]!= "None":
        if len(args["dyn_params_list_hydro"]) > 0:
            dyn_list_sorted = sorted(args["dyn_params_list_hydro"])
            for i in dyn_list_sorted:
                dyn_params = dyn_params + i + "_"
        else:
            dyn_params = "hydro_stat_"

    testing_dir = "ts" + str(args["t_test"][0])[:4] + "_" + str(args["t_test"][1])[:4]
    if not os.path.exists(os.path.join(args["output_model"], out_folder, dyn_params, testing_dir)):
        os.makedirs(os.path.join(args["output_model"], out_folder, dyn_params, testing_dir))
    # else:
    #     shutil.rmtree(os.path.join(args['output']['model'], out_folder))
    #     os.makedirs(os.path.join(args['output']['model'], out_folder))
    args["out_dir"] = os.path.join(args["output_model"], out_folder, dyn_params)
    args["testing_dir"] = testing_dir

    # saving the args file in output directory
    config_file = json.dumps(args)
    config_path = os.path.join(args["out_dir"], "config_file.json")
    if os.path.exists(config_path):
        os.remove(config_path)
    f = open(config_path, "w")
    f.write(config_file)
    f.close()

    return args


def update_args(args, **kw):
    for key in kw:
        if key in args:
            try:
                args[key] = kw[key]
            except ValueError:
                print("Something went wrong in args when updating " + key)
        else:
            print("didn't find " + key + " in args")
    return args

def source_flow_calculation(args, flow_out, c_NN, after_routing=True):
    varC_NN = args["varC_NN"]
    if "DRAIN_SQKM" in varC_NN:
        area_name = "DRAIN_SQKM"
    elif "area_gages2" in varC_NN:
        area_name = "area_gages2"
    else:
        print("area of basins are not available among attributes dataset")
    area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
        flow_out["flow_sim"].shape[
            0], 1, 1)
    # flow calculation. converting mm/day to m3/sec
    if after_routing == True:
        srflow = (1000 / 86400) * area * (flow_out["srflow"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * (flow_out["ssflow"]).repeat(1, 1, args["nmul"])  # ras
        gwflow = (1000 / 86400) * area * (flow_out["gwflow"]).repeat(1, 1, args["nmul"])
        # if args["hydro_model_name"] == "marrmot_PRMS_gw0":   # there are four flow outputs
        if "bas_shallow" in flow_out.keys():
            bas_shallow = (1000 / 86400) * area * (flow_out["bas_shallow"]).repeat(1, 1, args["nmul"])
    else:
        srflow = (1000 / 86400) * area * (flow_out["srflow_no_rout"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * (flow_out["ssflow_no_rout"]).repeat(1, 1, args["nmul"])  # ras
        gwflow = (1000 / 86400) * area * (flow_out["gwflow_no_rout"]).repeat(1, 1, args["nmul"])
        # if args["hydro_model_name"] == "marrmot_PRM_gw0":   # there are four flow outputs
        if "bas_shallow_no_rout" in flow_out.keys():
            bas_shallow = (1000 / 86400) * area * (flow_out["bas_shallow_no_rout"]).repeat(1, 1, args["nmul"])
    # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
    # ssflow = torch.clamp(ssflow, min=0.0)
    # gwflow = torch.clamp(gwflow, min=0.0)
    if "bas_shallow" in flow_out.keys():  # there are four flow outputs
        return dict(srflow=srflow,
                    ssflow=ssflow,
                    gwflow=gwflow,
                    bas_shallow=bas_shallow)
    else:  # there is three flow outputs
        return dict(srflow=srflow,
                    ssflow=ssflow,
                    gwflow=gwflow)
