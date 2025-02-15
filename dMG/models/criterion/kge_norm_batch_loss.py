from typing import Any, Dict, Optional

import torch


class KgeNormBatchLoss(torch.nn.Module):
    """Normalized Kling-Gupta efficiency (N-KGE) loss function.

    The N-KGE is calculated as:
        p: predicted value,
        t: target value,
        r: correlation coefficient,
        beta: variability ratio,
        gamma: variability error,
        KGE = 1 - sqrt((r - 1)^2 + (beta - 1)^2 + (gamma - 1)^2)
        N-KGE = 1 - KGE/(2 - KGE)

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
    eps : float
        Stability term to prevent division by zero. The default is 0.1. 
    """
    def __init__(
        self,
        target: torch.Tensor,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'Batch NKGE Loss'
        self.config = config
        self.device = device

        # Stability term
        self.eps = config.get('eps', 0.1)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        n_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_obs : torch.Tensor, optional
            The observed values. The default is None.
        n_samples : torch.Tensor, optional
            The number of samples in each batch. The default is None.
        
        Returns
        -------
        torch.Tensor
            The loss value.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]

        # Mask where observations are valid (not NaN).            
        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]

        # Compute mean and standard deviation for predicted and observed values
        mean_p = torch.mean(p_sub)
        mean_t = torch.mean(t_sub)
        std_p = torch.std(p_sub)
        std_t = torch.std(t_sub)

        # Compute correlation coefficient (r)
        numerator = torch.sum((p_sub - mean_p) * (t_sub - mean_t))
        denominator = torch.sqrt(torch.sum((p_sub - mean_p)**2) * torch.sum((t_sub - mean_t)**2))
        r = numerator / (denominator + self.eps)

        # Compute variability ratio (beta)
        beta = mean_p / (mean_t + self.eps)

        # Compute variability error (gamma)
        gamma = std_p / (std_t + self.eps)

        # Compute KGE
        kge = 1 - torch.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

        # Return KGE loss (1 - KGE)
        loss = 1 - kge/(2 - kge)

        return loss
