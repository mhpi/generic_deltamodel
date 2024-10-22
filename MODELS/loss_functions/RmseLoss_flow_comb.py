import torch
import json
import os


class RmseLoss_flow_comb(torch.nn.Module):
    def __init__(self, alpha=0.25, beta=1e-6):
        super(RmseLoss_flow_comb, self).__init__()
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        sim_flow = y_sim["flow_sim"].squeeze()
        if len(obs_flow[obs_flow == obs_flow]) > 0:
            mask_flow1 = obs_flow == obs_flow
            p = sim_flow[mask_flow1]
            t = obs_flow[mask_flow1]
            loss_flow1 = torch.sqrt(((p - t) ** 2).mean())  # RMSE item

            p1 = torch.log10(torch.sqrt(sim_flow + self.beta) + 0.1)
            t1 = torch.log10(torch.sqrt(obs_flow + self.beta) + 0.1)
            mask_flow2 = t1 == t1
            pa = p1[mask_flow2]
            ta = t1[mask_flow2]
            loss_flow2 = torch.sqrt(((pa - ta) ** 2).mean())  # Log-Sqrt RMSE item
            loss_flow_total = (1.0 - self.alpha) * loss_flow1 + self.alpha * loss_flow2
        else:
            loss_flow_total = 0.0
        return loss_flow_total