import numpy as np
from numpy.typing import NDArray
import torch
from typing import Any, Optional, Dict



class RmseLossComb(torch.nn.Module):
    """ Root mean squared error loss function.

    This loss combines the RMSE of the target variable with the RMSE of
    the log-transformed target variable.

    The RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((p - t)^2))
    
    The log-sqrt RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((log(sqrt(p)) - log(sqrt(t)))^2))

    Parameters
    ----------
    target : np.ndarray
        The target data array.
    config : dict
        The configuration dictionary.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.
    
    Optional Parameters: (Set in config)
    --------------------
    alpha : float
        Weighting factor for the log-sqrt RMSE. The default is 0.25.
    beta : float
        Stability term to prevent division by zero. The default is 1e-6.
    """
    def __init__(
        self,
        target: NDArray[np.float32],
        config: Dict[str, Any],
        device: Optional[str] = 'cpu'
    ) -> None:
        super(RmseLossComb, self).__init__()
        self.config = config
        self.device = device

        # Weights of log-sqrt RMSE
        self.alpha = config.get('alpha', 0.25)
        self.beta = config.get('beta', 1e-6)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_obs : torch.Tensor
            The observed values.
        n_samples : torch.Tensor
            The number of samples in each batch.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]
        n_samples = n_samples.cpu().detach().numpy().astype(int)

        if len(target) > 0:
            # Mask where observations are valid (not NaN).            
            mask1 = ~torch.isnan(target)
            p_sub = prediction[mask1]
            t_sub = target[mask1]
            
            # RMSE
            p_sub1 = torch.log10(torch.sqrt(prediction + self.beta) + 0.1)
            t_sub1 = torch.log10(torch.sqrt(target + self.beta) + 0.1)
            loss1 = torch.sqrt(((p_sub - t_sub) ** 2).mean())  # RMSE item

            # Log-Sqrt RMSE
            mask2 = ~torch.isnan(t_sub1)
            p_sub3 = p_sub1[mask2]
            t_sub3 = t_sub1[mask2]
            loss2 = torch.sqrt(((p_sub3 - t_sub3) ** 2).mean())

            # Combined losses
            loss_total = (1.0 - self.alpha) * loss1 + self.alpha * loss2
        else:
            loss_total = torch.tensor(0.0, device=self.device)
        return loss_total
    