from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray


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
        config: dict[str, Any],
        device: Optional[str] = 'cpu',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

    def _format(
        self,
        y_pred: Union[NDArray, torch.Tensor],
        y_obs: Union[NDArray, torch.Tensor],
    ) -> None:
        """Check input tensors for dimensionality and type.
        
        Parameters
        ----------
        y_pred
            Tensor of predicted target data.
        y_obs
            Tensor of target observation data.

        Returns
        -------
        y_pred
            Formatted tensor of predicted target data.
        y_obs
            Formatted tensor of target observation data.
        """
        # Check type
        if isinstance(y_pred, np.ndarray):
            prediction = torch.from_numpy(y_pred).to(self.device)
        elif isinstance(y_pred, torch.Tensor):
            prediction = y_pred.to(self.device)
        else:
            raise ValueError("y_pred must be a numpy array or torch tensor.")

        if isinstance(y_obs, np.ndarray):
            target = torch.from_numpy(y_obs).to(self.device)
        elif isinstance(y_obs, torch.Tensor):
            target = y_obs.to(self.device)
        else:
            raise ValueError("y_obs must be a numpy array or torch tensor.")

        # Check dimensionality -> [n_timesteps, n_samples]
        if prediction.ndim > 2:
            prediction = prediction.squeeze()
        if target.ndim > 2:
            target = target.squeeze()

        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError("Input tensors must have 2 dimensions.")
        if prediction.shape != target.shape:
            raise ValueError("Input tensors must have the same shape.")

        return prediction, target

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
