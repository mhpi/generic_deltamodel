from typing import Any, Dict, Optional

import numpy as np
import torch
from numpy.typing import NDArray


class NseSqrtLossBatch(torch.nn.Module):
    """Sqrt normalized squared error loss function.

    Same as Fredrick 2019, batch NSE loss.
    Adapted from Yalan Song.

    Uses the first variable of the target array as the target variable.
    
    The sNSE is calculated as:
        p: predicted value,
        t: target value,
        sNSE = 1 - sum((p - t)^2) / sum(t - mean(t))

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
    eps : float
        Stability term to prevent division by zero. The default is 0.1.
    nearzero : float
        Small value to perturb square root. The default is 1e-6.
    """
    def __init__(
        self,
        target: NDArray[np.float32],
        config: Dict[str, Any],
        device: Optional[str] = 'cpu'
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.std = np.nanstd(target[:, :, 0], axis=0)
        
        # Stability terms
        self.eps = config.get('eps', 0.1)
        self.nearzero = config.get('nearzero', 1e-6)

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
            # Prepare grid-based standard deviations for normalization.
            n_timesteps = target.shape[0]
            std_batch = torch.tensor(
                np.tile(self.std[n_samples].T, (n_timesteps, 1)),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device
            )
            
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]
            std_sub = std_batch[mask]

            sq_res = torch.sqrt((p_sub - t_sub)**2 + self.nearzero)
            norm_res = sq_res / (std_sub + self.eps)
            loss = torch.mean(norm_res)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
    