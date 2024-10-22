import torch
import numpy as np
import json
import os

class NSEsqrtLoss_flow(torch.nn.Module):
    # Similar as Fredrick 2019, batch NSE loss, use RMSE and STD instead
    # stdarray: the standard deviation of the runoff for all basins
    def __init__(self, stdarray_flow, eps=0.1):
        super(NSEsqrtLoss_flow, self).__init__()
        self.std_flow = stdarray_flow
        self.eps = eps

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        sim_flow = y_sim["flow_sim"].squeeze()

        ## for flow
        if len(obs_flow[obs_flow == obs_flow]) > 0:
            nt = obs_flow.shape[0]
            stdse = np.tile(self.std_flow[igrid], (nt, 1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(args["device"])
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            stdw = stdbatch[mask_flow1]
            # p = torch.where((p == t),
            #                 p + args["NEARZERO"],
            #                 p)
            sqRes = torch.sqrt(args["NEARZERO"] + (p - t)**2)
            normRes = sqRes / (stdw + self.eps)
            #yalan's version
            sqRes = (p - t) ** 2
            normRes = sqRes / (stdw + self.eps)**2

            loss = torch.mean(normRes)
        else:
            loss = 0.0
        return loss