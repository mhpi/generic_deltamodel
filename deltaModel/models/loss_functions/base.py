from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from numpy.typing import NDArray
import numpy as np

import torch


class BaseLossFunction(torch.nn.Module, ABC):
    """Base class for loss functions extended from PyTorch Module.
    
    All loss functions should inherit from this class, which enforces minimum
    requirements for loss functions used within dMG.
    
    Parameters
    ----------
    target : np.ndarray
        The target data array.
    config : dict
        The configuration dictionary.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.
    """
    def __init__(
        self,
        target: NDArray[np.float32],
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

    @abstractmethod
    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        pass
