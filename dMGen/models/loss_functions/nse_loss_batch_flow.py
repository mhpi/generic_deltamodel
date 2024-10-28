import torch
import numpy as np

class NseLossBatchFlow(torch.nn.Module):
    """
    Same as Fredrick 2019, batch NSE loss.
    Adapted from Yalan Song.

    inputs:
    - stdarray: the standard deviation of the runoff for all basins.
    - eps: epsilon stability term.
    """
    def __init__(self, stdarray, eps=0.1):
        super(NseLossBatchFlow, self).__init__()
        self.std = stdarray
        self.eps = eps

    def forward(self, args, y_sim, y_obs, igrid):
        varTar_NN = args["target"]
        sim_flow = y_sim["flow_sim"].squeeze()
        obs_flow = y_obs[:, :, varTar_NN.index("00060_Mean")]

        if len(obs_flow) > 0:
            nt = obs_flow.shape[0]
            stdse = np.tile(self.std[igrid].T, (nt, 1))
            stdbatch = torch.tensor(stdse, requires_grad=False).float().to(args["device"])  #.cuda
            
            mask = obs_flow == obs_flow
            p = sim_flow[mask]
            t = obs_flow[mask]
            stdw = stdbatch[mask]
            sqRes = (p - t)**2
            normRes = sqRes / (stdw + self.eps)**2
            loss = torch.mean(normRes)
        else:
            loss = 0.0
        return loss
