import torch
import numpy as np
import json
import os

class NSEsqrtLoss_flow_temp(torch.nn.Module):
    # Similar as Fredrick 2019, batch NSE loss, use RMSE and STD instead
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray_flow, stdarray_temp, eps=0.1):
        super(NSEsqrtLoss_flow_temp, self).__init__()
        self.std_flow = stdarray_flow
        self.std_temp = stdarray_temp
        self.eps = eps

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        sim_flow = y_sim["flow_sim"].squeeze()
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        sim_temp = y_sim["temp_sim"].squeeze()

        ## for flow
        if len(obs_flow[obs_flow == obs_flow]) > 0:
            nt = obs_flow.shape[0]
            stdse = np.tile(self.std_flow[igrid], (nt, 1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(args["device"])
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            stdw = stdbatch[mask_flow1]

            # sqRes = torch.sqrt(args["NEARZERO"] + (p - t)**2)
            # normRes = sqRes / (stdw + self.eps)
            # yalan's version
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps) ** 2
            loss_flow = torch.mean(normRes)
        else:
            loss_flow = 0.0

        # for temp
        if len(obs_temp[obs_temp==obs_temp]) > 0:
            nt = obs_temp.shape[0]
            stdse_temp = np.tile(self.std_temp[igrid], (nt, 1))
            stdbatch_temp = torch.tensor(stdse_temp, requires_grad=False).float().to(args["device"])
            mask_temp1 = obs_temp == obs_temp
            p_temp = sim_temp[mask_temp1]
            t_temp = obs_temp[mask_temp1]
            stdw_temp = stdbatch_temp[mask_temp1]

            # sqRes_temp = torch.sqrt(args["NEARZERO"] + (p_temp - t_temp) ** 2)
            # normRes_temp = sqRes_temp / (stdw_temp + self.eps)

            # yalan's version
            sqRes_temp = (p_temp - t_temp) ** 2
            normRes_temp = sqRes_temp / (stdw_temp + self.eps) ** 2

            loss_temp = torch.mean(normRes_temp)
        else:
            loss_temp = 0.0
        loss = loss_flow + loss_temp
        return loss