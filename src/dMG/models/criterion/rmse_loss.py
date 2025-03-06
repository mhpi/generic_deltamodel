from typing import Any, Dict, Optional

import torch


class RmseLoss(torch.nn.Module):
    """ Root mean squared error (RMSE) loss function.

    The RMSE is calculated as:
        p: predicted value,
        t: target value,
        RMSE = sqrt(mean((p - t)^2))

    Parameters
    ----------
    target : torch.Tensor
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
        target: torch.Tensor,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'RMSE Loss'
        self.config = config
        self.device = device

        # Weights of log-sqrt RMSE
        self.alpha = config.get('alpha', 0.25)
        self.beta = config.get('beta', 1e-6)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor,
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

        Returns
        -------
        torch.Tensor
            The combined loss.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]

        if len(target) > 0:
            # Mask where observations are valid (not NaN).            
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask] 
            loss = torch.sqrt(((p_sub - t_sub) ** 2).mean())
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
