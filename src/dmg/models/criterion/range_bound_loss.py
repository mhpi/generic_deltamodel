from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class RangeBoundLoss(BaseCriterion):
    """Loss function that penalizes values outside of a specified range.

    Adapted from Tadd Bindas.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - lb: Lower bound for the loss. Default is 0.9.

        - ub: Upper bound for the loss. Default is 1.1.

        - loss_factor: Scaling factor for the loss. Default is 1.0.
    """
    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs,
    ) -> None:
        super().__init__(config, device)
        self.name = 'Range-bound Loss'
        self.config = config
        self.device = device

        self.lb = kwargs.get('lb', config.get('lb', 0.9))
        self.ub = kwargs.get('ub', config.get('ub', 1.1))
        self.loss_factor = kwargs.get('loss_factor', config.get('loss_factor', 1.0))

    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the range-bound loss.
        
        Loss function that penalizes values outside of a specified range. Loss
        is calculated as the sum of the individual average losses for each batch
        in the prediction tensor.
        
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
            The range-bound loss.
        """
        prediction, target = self._format(y_pred, y_obs)

        # Calculate the deviation from the bounds
        upper_bound_loss = torch.relu(prediction - self.ub)
        lower_bound_loss = torch.relu(self.lb - prediction)

        # Batch mean loss across all predictions
        loss = self.loss_factor * (upper_bound_loss + lower_bound_loss)
        loss = loss.mean(dim=1).sum()

        return loss
