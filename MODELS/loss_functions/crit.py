import torch
import json
import os


class SigmaLoss(torch.nn.Module):
    def __init__(self, prior="gauss"):
        super(SigmaLoss, self).__init__()
        self.reduction = "elementwise_mean"
        if prior == "":
            self.prior = None
        else:
            self.prior = prior.split("+")

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == "gauss":
                loss = torch.exp(-s).mul((p - t) ** 2) / 2 + s / 2
            elif self.prior[0] == "invGamma":
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = (
                    torch.exp(-s).mul((p - t) ** 2 + c2 / nt) / 2
                    + (1 / 2 + c1 / nt) * s
                )
            lossMean = lossMean + torch.mean(loss)
        return lossMean


class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0

        for k in range(ny):
            ### to calculate loss based on measurements
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            maskt = t0 == t0
            # maskp = p0 == p0
            # mask = maskp * maskt > 0
            p = p0[maskt]
            t = t0[maskt]
            temp = torch.sqrt(((p - t) ** 2).mean())
            loss = loss + temp

            #################################################
            ### to calculate loss based on basins
            # Chaopeng Version ##############
            # Loss = RMSEbasin.RMSEbasinLosstest()
            # l = Loss(output, target)
            # loss = loss + l
            # Farshid ############# It is true but takes too much time to run

            # temp1 = torch.zeros((t0.shape[1])).cuda()
            # for i in range(t0.shape[1]):
            #     mask = t0[:, i] == t0[:, i]
            #     p = p0[:, i][mask]
            #     t = t0[:, i][mask]
            #     A = (p-t)**2
            #     temp1[i] = (A.mean())
            # mask = temp1==temp1
            # if mask.sum()==0:
            #     temp1[mask]=0
            # else:
            #     temp1 = temp1[mask]
            # temp = torch.sqrt(temp1).mean()
            #
            # loss = loss+temp

            ###########################

        return loss


class RmseLossANN(torch.nn.Module):
    def __init__(self, get_length=False):
        super(RmseLossANN, self).__init__()
        self.ind = get_length

    def forward(self, output, target):
        if len(output.shape) == 2:
            p0 = output[:, 0]
            t0 = target[:, 0]
        else:
            p0 = output[:, :, 0]
            t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t) ** 2).mean())
        if self.ind is False:
            return loss
        else:
            Nday = p.shape[0]
            return loss, Nday


class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p - pmean
            t_ub = t - tmean
            temp = torch.sqrt(((p_ub - t_ub) ** 2).mean())
            loss = loss + temp
        return loss


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t) ** 2).mean()
            loss = loss + temp
        return loss


class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample + 1
        # minimize the opposite average NSE
        loss = -(losssum / nsample)
        return loss


class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST) + 0.1) ** 2)
                losssum = losssum + temp
                nsample = nsample + 1
        loss = losssum / nsample
        return loss



class RmseLoss_temp_flow(torch.nn.Module):
    def __init__(self, w1=0.5, w2=None, alpha=0.25, beta=1e-6):
        super(RmseLoss_temp_flow, self).__init__()
        self.w1 = w1
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.w2 = w2

    def forward(self, obs_flow, obs_temp, sim_flow, sim_temp):
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1
        # flow
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

        loss = self.w1 * loss_flow_total + self.w2 * loss_temp
        return loss





class RmseLoss_temp_flow_BFI(torch.nn.Module):
    def __init__(self, w1=0.5, w2=None, w3=2.0, alpha=0.25, beta=1e-6):
        super(RmseLoss_temp_flow_BFI, self).__init__()
        self.w1 = w1
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.w2 = w2
        self.w3 = w3

    def forward(self, obs_flow, obs_temp, sim_flow, sim_temp, obs_BFI, sim_BFI):
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1
        # flow
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
            loss_BFI = torch.sqrt(((p_BFI - t_BFI) ** 2).mean())  # RMSE item
        else:
            loss_BFI = 0.0

        loss = self.w1 * loss_flow_total + self.w2 * loss_temp + self.w3 * loss_BFI
        return loss


class RmseLoss_temp_flow_norm(torch.nn.Module):
    def __init__(self, args, w1=0.5, w2=None, alpha=0.25, beta=1e-6, normalizing_flag=True):
        super(RmseLoss_temp_flow_norm, self).__init__()
        self.args = args
        self.w1 = w1
        self.alpha = alpha  # weights of log-sqrt RMSE
        self.beta = beta
        self.normalizing_flag = normalizing_flag
        self.w2=w2
    def forward(self, obs_flow, obs_temp, sim_flow, sim_temp):
        # flow
        if self.w2 == None:    # w1 + w2 =1
            self.w2 = 1 - self.w1
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

        if self.normalizing_flag == True:
            statFile = os.path.join(self.args["out_dir"], "Statistics_basinnorm.json")
            with open(statFile, "r") as fp:
                statDict = json.load(fp)
            loss_flow_total = loss_flow_total / abs(statDict["00060_Mean"][2])  # divided by mean
            loss_temp = loss_temp / statDict["00010_Mean"][2]

        loss = self.w1 * loss_flow_total + self.w2 * loss_temp
        return loss