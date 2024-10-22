import torch
class RmseLoss_BFI_temp(torch.nn.Module):
    def __init__(self, w1=0.05, w2=None):
        super(RmseLoss_BFI_temp, self).__init__()
        self.w1 = w1
        if w2 == None:  # w1 + w2 =1
            self.w2 = 1 - self.w1
        else:
            self.w2 = w2

    def forward(self, args, y_sim, y_obs, igrid):
        # flow
        varTar_NN = args["target"]
        obs_temp = y_obs[:, :, varTar_NN.index("00010_Mean")]
        obs_BFI = y_obs[0, :, varTar_NN.index("BFI_AVE")]
        sim_temp = y_sim["temp_sim"].squeeze()
        sim_BFI = y_sim["BFI_sim"].squeeze()

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
            # if the difference is more than 0.3, the calculate the loss, if not, loss is zero
            p_BFI2 = torch.where((torch.abs((p_BFI - t_BFI)) / t_BFI > 0.25), p_BFI, t_BFI)
            loss_BFI = torch.sqrt(((p_BFI2 - t_BFI) ** 2).mean())  # RMSE item
        else:
            loss_BFI = 0.0
        loss = self.w1 * loss_BFI + self.w2 * loss_temp
        return loss
