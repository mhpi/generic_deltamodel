import logging

import torch
import torch.nn as nn
from conf.config import Config

log = logging.getLogger(__name__)



class RangeBoundLoss(nn.Module):
    """
    From dMCdev @ Tadd Bindas;

    Calculate a loss value based on the distance of inputs from the
    upper and lower bounds of a pre-defined range.
    """
    def __init__(self, config: Config):
        super(RangeBoundLoss, self).__init__()
        self.config = config
        self.lb = torch.tensor([self.config['weighting_nn']['loss_lower_bound']], device=config['device'])
        self.ub = torch.tensor([self.config['weighting_nn']['loss_upper_bound']],device=config['device'])
        self.factor = torch.tensor(self.config['weighting_nn']['loss_factor'])
        log.info(f"wNN Loss Factor: {self.factor}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        loss = 0
        for i in range(len(inputs)):
            lb = self.lb[i]
            ub = self.ub[i]
            upper_bound_loss = torch.relu(inputs[i] - ub)
            lower_bound_loss = torch.relu(lb - inputs[i])
            mean_loss = self.factor * (upper_bound_loss + lower_bound_loss).mean() / 2.0
            loss += mean_loss

        # upper_bound_loss = torch.relu(inputs - self.ub)
        # lower_bound_loss = torch.relu(self.lb - inputs)
        # mean_loss = self.factor * (upper_bound_loss + lower_bound_loss).mean() / 2.0
        # return mean_loss

        return loss
    