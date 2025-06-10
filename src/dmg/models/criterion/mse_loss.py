from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class MSELoss(BaseCriterion):
    """Mean Squared Error (MSE) loss function.
    
    The MSE is calculated as:
        p: predicted value,
        t: target value,
        MSE = mean((p - t)^2)

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments for interface compatibility, not used.
    """
    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__(config, device)
        self.name = 'MSE Loss'
        self.config = config
        self.device = device

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
        prediction, target = self._format(y_pred, y_obs)


        if len(target) > 0:
            mask = ~torch.isnan(target)
            p_sub = prediction[mask]
            t_sub = target[mask]

            loss = torch.mean((p_sub - t_sub)**2)
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss
