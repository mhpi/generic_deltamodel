import torch
import json
import os

class RmseLoss_flow_temp_BFI_PET(torch.nn.Module):
    def __init__(self, w1=0.5, w2=None, w3=0.05, w4=0.01, alpha=0.25, beta=1e-6):
        super(RmseLoss_flow_temp_BFI_PET, self).__init__()
        self.w1 = w1
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

    def forward(self, args, y_sim, y_obs, igrid):
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1
        # flow
        varTar_NN = args["target"]
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        obs_BFI = y_obs[0, :, varTar_NN.index("BFI_AVE")]
        obs_PET = y_obs[0, :, varTar_NN.index("PET")]
        sim_flow = y_sim["flow_sim"].squeeze()    #  simulation
        sim_temp = y_sim["temp_sim"].squeeze()
        sim_BFI = y_sim["BFI_sim"].squeeze()
        sim_PET = torch.sum(y_sim["PET_hydro"][:, :, 0], dim=0).squeeze()

        if len(obs_flow[obs_flow==obs_flow]) > 0:
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
            loss_flow_total = (1.0-self.alpha) * loss_flow1 + self.alpha * loss_flow2
        else:
            loss_flow_total = 0.0

        # temp
        if len(obs_temp[obs_temp==obs_temp]) > 0:
            mask_temp1 = obs_temp == obs_temp
            p_temp = sim_temp[mask_temp1]
            t_temp = obs_temp[mask_temp1]
            loss_temp = torch.sqrt(((p_temp - t_temp) ** 2).mean())  # RMSE item
        else:
            loss_temp = 0.0

        # BFI calculation
        if len(obs_BFI[obs_BFI==obs_BFI]) > 0:
            mask_BFI1 = obs_BFI == obs_BFI
            p_BFI = sim_BFI[mask_BFI1]
            t_BFI = obs_BFI[mask_BFI1]

            p_BFI2 = torch.where((torch.abs((p_BFI - t_BFI)) / t_BFI > 0.25), p_BFI, t_BFI)
            loss_BFI = torch.sqrt(((p_BFI2 - t_BFI) ** 2).mean())  # RMSE item
        else:
            loss_BFI = 0.0

        # PET calculation
        # Todo: PET loss doesn't make sense. it should be AET. but compared with what?
        if len(obs_PET[obs_PET == obs_PET]) > 0:
            mask_PET1 = obs_PET == obs_PET
            p_PET = sim_PET[mask_PET1]
            t_PET = obs_PET[mask_PET1]
            p_PET2 = torch.where((torch.abs((p_PET - t_PET)) / t_PET > 0.1), p_PET, t_PET)
            loss_PET = torch.sqrt(((p_PET2 - t_PET) ** 2).mean())  # RMSE item
        else:
            loss_PET = 0.0

        loss = self.w1 * loss_flow_total + self.w2 * loss_temp + self.w3 * loss_BFI + self.w4 * loss_PET
        return loss
