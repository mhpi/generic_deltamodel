from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseCriterion(torch.nn.Module, ABC):
    """Base class for loss functions extended from PyTorch Module.
    
    All loss functions should inherit from this class, which enforces minimum
    requirements for loss functions used within dMG.
    
    Parameters
    ----------
    config
        Configuration dictionary.
    device
        The device to run loss function on.
    **kwargs
        Additional arguments for loss computation, maintains loss function
        interchangeability. Not always used.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.
        
        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.
        **kwargs
            Additional arguments for loss computation, maintains loss function
            interchangeability. Not always used.
            
        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        pass
