from abc import ABC, abstractmethod

import torch


class BaseTrainer(ABC):
    @abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer as named in config."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Entry point for training loop."""
        pass

    @abstractmethod
    def test(self) -> None:
        """Run testing loop and save results."""

    @abstractmethod
    def calc_metrics(self) -> None:
        """Calculate metrics for the model."""
        pass
