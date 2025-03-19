from typing import Any, Dict, Optional, List

import torch
import numpy as np
import math


class TrendLoss(torch.nn.Module):
    """Trend learning-based loss function.
    
    Similar to the Tre-Loss of Haibing Liao et al. (2024).
     
    The trend loss is calculated as:
        p: predicted value,
        t: target value,
        Trend Loss = RMSE + mean annual trend loss + quantile trend loss

        1. RMSE = sqrt(mean((p - t)^2))
        2. Mean annual trend loss = (slope(t) - slope(p))^2
        3. Quantile trend loss = (slope(t) / - slope(p))^2
        
    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - plist: List of percentiles for quantile trend loss.
            Default is [100, 98, 50, 30, 2].
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: List[int],
    ) -> None:
        super().__init__()
        self.name = 'Trend Loss'
        self.config = config
        self.device = device

        self.plist = kwargs.get('plist', self.config.get(
            'plist', [100, 98, 50, 30, 2])
        )

    def get_slope(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the slope of the trend line.
        
        Parameters
        ----------
        x
            Tensor of input data.
        
        Returns
        -------
        torch.Tensor
            The slope of the trend line.
        """
        idx = 0
        n = len(x)
        d = torch.ones(int(n * (n - 1) / 2))

        for i in range(n - 1):
            j = torch.arange(start=i + 1, end=n)
            d[idx: idx + len(j)] = (x[j] - x[i]) / (j - i).type(torch.float)
            idx = idx + len(j)

        return torch.median(d)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute loss.
        
        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments for interface compatibility, not used.

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]

        n_samples = target.shape[1]

        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]

        # 1. RMSE
        rmse_loss = torch.sqrt(((p_sub - t_sub)**2).mean())

        trend_loss = 0
        idx = 0

        for i in range(n_samples):
            pgage0 = p_sub[:, i].reshape(-1, 365)
            tgage0 = t_sub[:, i].reshape(-1, 365)
            gbool = np.zeros(tgage0.shape[0]).astype(int)
            pgageM = torch.zeros(tgage0.shape[0])
            pgageQ = torch.zeros(tgage0.shape[0], len(self.plist))
            tgageM = torch.zeros(tgage0.shape[0])
            tgageQ = torch.zeros(tgage0.shape[0], len(self.plist))

            for j in range(tgage0.shape[0]):
                pgage = pgage0[j, :]
                tgage = tgage0[j, :]
                maskg = tgage == tgage

                # Quality control
                if maskg.sum() > (1 - 2/12)*365:
                    gbool[j] = 1
                    pgage = pgage[maskg]
                    tgage = tgage[maskg]
                    pgageM[j] = pgage.mean()
                    tgageM[j] = tgage.mean()

                    for perc in range(len(self.plist)):
                        k = math.ceil(self.plist[perc] / 100 * 365)
                        # pgageQ[j, perc] = torch.kthvalue(pgage, k)[0]
                        # tgageQ[j, perc] = torch.kthvalue(tgage, k)[0]
                        pgageQ[j, perc] = torch.sort(pgage)[0][k-1]
                        tgageQ[j, perc] = torch.sort(tgage)[0][k-1]

            # Quality control
            if gbool.sum() > 6:
                idx += 1
                pgageM = pgageM[gbool]
                tgageM = tgageM[gbool]

                # 2. Mean annual trend loss
                trend_loss += (self.get_slope(tgageM)-self.get_slope(pgageM))**2
                pgageQ = pgageQ[gbool, :]
                tgageQ = tgageQ[gbool, :]

                # 3. Quantile trend loss
                for ii in range(tgageQ.shape[1]):
                    trend_loss += (self.get_slope(tgageQ[:, ii]) /
                                        - self.get_slope(pgageQ[:, ii]))**2

        return rmse_loss + trend_loss/idx
