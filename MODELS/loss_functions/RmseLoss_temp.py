import torch
import json
import os


class RmseLoss_temp(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=1e-6):
        super(RmseLoss_temp, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        sim_temp = y_sim["temp_sim"].squeeze()
        if len(obs_temp[obs_temp == obs_temp]) > 0:
            mask_temp1 = obs_temp == obs_temp
            p_temp = sim_temp[mask_temp1]
            t_temp = obs_temp[mask_temp1]
            loss_temp = torch.sqrt(((p_temp - t_temp) ** 2).mean())  # RMSE item
        else:
            loss_temp = 0.0
        return loss_temp