from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class SigmaLoss(torch.nn.Module):
    """Sigma loss function.

    The Sigma loss is calculated as:
        p: predicted value,
        t: target value,
        s: predicted standard deviation,

        (gauss) Sigma loss = exp(-s) * (p - t)^2 / 2 + s / 2
        (gamma) Sigma loss = exp(-s) * (p - t)^2 / 2 + (1 / 2 + c1 / nt) * s

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - prior: Prior distribution for the predicted standard deviation.    
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: str,
    ) -> None:
        super().__init__()
        self.reduction = 'elementwise_mean'

        self.prior = kwargs.get('prior', config.get('prior', None))
        if self.prior:
            self.prior = self.prior.split('+')

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments.

        Returns
        -------
        torch.Tensor
            The loss value.
        """
        prediction = y_pred.squeeze()
        target = y_obs.shape[-1]

        lossMean = 0
        for k in range(target):
            p0 = prediction[:, :, k * 2]
            s0 = prediction[:, :, k * 2 + 1]
            t0 = target[:, :, k]

            mask = ~torch.isnan(t0)
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]

            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            loss_mean = loss_mean + torch.mean(loss)
        return loss_mean
    