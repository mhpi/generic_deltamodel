from typing import Any, Dict, Optional

import numpy as np
import torch
from numpy.typing import NDArray


class RangeBoundLoss(torch.nn.Module):
    """Loss function that penalizes values outside of a specified range.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    target : np.ndarray
        The target data array. The default is None.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.

    Optional Parameters: (Set in config)
    --------------------
    lb : float
        The lower bound for the loss. The default is 0.9.
    ub : float
        The upper bound for the loss. The default is 1.1.
    loss_factor : float
        The scaling factor for the loss. The default is 1.0.
    Adapted from Tadd Bindas.
    """
    def __init__(
            self,
            config: Dict[str, Any],
            target: NDArray[np.float32] = None,
            device: Optional[str] = 'cpu'
        ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.lb = config.get('lb', 0.9)
        self.ub = config.get('ub', 1.1)
        self.scale_factor = config.get('loss_factor', 1.0)
        self._lb = torch.tensor(
            [self.lb],
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )
        self._ub = torch.tensor(
            [self.ub],
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )
        self._scale_factor = torch.tensor(
            self.scale_factor,
            dtype=torch.float32,
            device=device,
            requires_grad=False
        )

    def forward(
            self,
            y_pred: torch.Tensor,
            y_obs: Optional[torch.Tensor] = None,
            n_samples: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """Compute the range-bound loss.
        
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
            The range-bound loss.
        """
        # Calculate the deviation from the bounds
        upper_bound_loss = torch.relu(y_pred - self._ub)
        lower_bound_loss = torch.relu(self._lb - y_pred)

        # Mean loss across all predictions
        loss = self._scale_factor * (upper_bound_loss + lower_bound_loss).mean()
        return loss
