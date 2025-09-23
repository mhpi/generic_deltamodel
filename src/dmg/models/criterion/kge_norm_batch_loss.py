from typing import Any, Optional

import torch

from dmg.models.criterion.base import BaseCriterion


class KgeNormBatchLoss(BaseCriterion):
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
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments.

        - eps: Stability term to prevent division by zero. Default is 0.1.
    """
    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: int,
    ) -> None:
        super().__init__(config, device)
        self.name = 'Batch NKGE Loss'
        self.config = config
        self.device = device

        self.eps = kwargs.get('eps', config.get('eps', 0.1))

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
            The loss value.
        """
        prediction, target = self._format(y_pred, y_obs)

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
