import importlib
import numpy as np

def get_lossFun(args, obs):
    # module = importlib.import_module(args["loss_function"])
    spec = importlib.util.spec_from_file_location(args["loss_function"], "./MODELS/loss_functions/" + args["loss_function"] + ".py")
    module = spec.loader.load_module()
    loss_function_default = getattr(module, args["loss_function"])
    if (args["loss_function"] =="RmseLoss_flow_temp") or (args["loss_function"] =="RmseLoss_flow_temp_BFI") or \
            (args["loss_function"] =="RmseLoss_flow_temp_BFI_PET") or (args["loss_function"] =="RmseLoss_BFI_temp"):
        lossFun = loss_function_default(w1=args["loss_function_weights"]["w1"],
                                        w2=args["loss_function_weights"]["w2"])
    elif (args["loss_function"] == "NSEsqrtLoss_flow_temp"):
        std_obs_flow = np.nanstd(obs[:, :, args["target"].index("00060_Mean")], axis=0)
        std_obs_flow[std_obs_flow != std_obs_flow] = 1.0

        std_obs_temp = np.nanstd(obs[:, :, args["target"].index("00010_Mean")], axis=0)
        std_obs_temp[std_obs_temp != std_obs_temp] = 1.0

        lossFun = loss_function_default(stdarray_flow=std_obs_flow,
                                        stdarray_temp=std_obs_temp)
    elif (args["loss_function"] == "NSEsqrtLoss_flow"):
        std_obs_flow = np.nanstd(obs[:, :, args["target"].index("00060_Mean")], axis=0)
        std_obs_flow[std_obs_flow != std_obs_flow] = 1.0
        lossFun = loss_function_default(stdarray_flow=std_obs_flow)
    else:
        lossFun = loss_function_default()
    return lossFun

