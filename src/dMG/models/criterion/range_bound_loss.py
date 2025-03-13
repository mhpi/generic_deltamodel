from typing import Any, Dict, Optional

import torch


class RangeBoundLoss(torch.nn.Module):
    """Loss function that penalizes values outside of a specified range.

    Parameters
    ----------
    config : dict
        The configuration dictionary.
    target : torch.Tensor
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
        device: Optional[str] = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = 'Range Bound Loss'
        self.config = config
        self.device = device
        self.lb = config.get('lb', 0.9)
        self.ub = config.get('ub', 1.1)
        self.scale_factor = config.get('loss_factor', 1.0)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        n_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the range-bound loss.
        
        Loss function that penalizes values outside of a specified range. Loss is calculated as the sum of the individual average losses for each batch in the prediction tensor.
        
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
        upper_bound_loss = torch.relu(y_pred - self.ub)
        lower_bound_loss = torch.relu(self.lb - y_pred)

        # Batch mean loss across all predictions
        loss = self.scale_factor * (upper_bound_loss + lower_bound_loss)
        loss = loss.mean(dim=1).sum()

        return loss
