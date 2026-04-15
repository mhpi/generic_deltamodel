from abc import ABC, abstractmethod

import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class BaseSampler(Dataset, ABC):
    """Base class for data samplers extended from PyTorch Dataset.

    All data samplers should inherit from this class to enforce minimum
    requirements for use within dMG.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    @abstractmethod
    def get_training_sample(self, dataset: dict, *args, **kwargs) -> dict:
        """Generate a training batch from the dataset."""

    @abstractmethod
    def get_validation_sample(self, dataset: dict, *args, **kwargs) -> dict:
        """Generate a batch for validation/evaluation."""

    def to_tensor(self, data: NDArray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.

        Parameters
        ----------
        data : numpy.ndarray
            The input data to convert.

        Returns
        -------
        torch.Tensor
            The data as a PyTorch tensor.
        """
        return torch.tensor(
            data,
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )

    def validate_config(self):
        """Validate the configuration dictionary to ensure required keys."""
        if not hasattr(self, 'config') or self.config is None:
            raise AttributeError(
                "Subclass must set self.config before calling validate_config."
            )
        required_keys = ["dtype", "device"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
