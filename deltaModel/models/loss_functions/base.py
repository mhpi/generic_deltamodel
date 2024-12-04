from abc import ABC, abstractmethod

import torch


class BaseLossFunction(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss."""
        pass
