from abc import ABC, abstractmethod
from typing import Any, Optional

import torch


class BaseTrainer(ABC):
    """
    Abstract base class for trainers, providing a structure for implementing
    training, testing, inference, and metric calculation workflows.

    Attributes
    ----------
    config
        Configuration dictionary.
    model
        Model to be trained or evaluated.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = None

    @abstractmethod
    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer as named in the config.
        
        Returns
        -------
        torch.optim.Optimizer
            The initialized optimizer.
        """
        raise NotImplementedError("Derived classes must implement `init_optimizer` method.")

    @abstractmethod
    def train(self) -> None:
        """Entry point for the training loop."""
        raise NotImplementedError("Derived classes must implement `train` method.")

    @abstractmethod
    def evaluate(self) -> None:
        """Run testing loop and save results."""
        raise NotImplementedError("Derived classes must implement `evaluate` method.")

    @abstractmethod
    def inference(self) -> None:
        """Run testing loop and save results."""
        raise NotImplementedError("Derived classes must implement `evaluate` method.")

    @abstractmethod
    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """Calculate metrics for the model.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        raise NotImplementedError("Derived classes must implement the `calc_metrics` method.")

    def save_model(self, epoch: int) -> None:
        """Save the model at a specified epoch.
        
        Parameters
        ----------
        epoch
            The epoch number to save the model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot save model.")
        model_path = self.config.get("save_path", "./model.pth")
        torch.save(self.model.state_dict(), f"{model_path}_epoch_{epoch}.pth")
        print(f"Model saved to {model_path}_epoch_{epoch}.pth")

    def load_model(self, checkpoint_path: str) -> None:
        """Load model from a checkpoint.
        
        Parameters
        ----------
        checkpoint_path
            Path to the model checkpoint.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot load model.")
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")

    def log_config(self) -> None:
        """Log the configuration details."""
        print("Trainer Configuration:")
        for key, value in self.config.items():
            print(f"{key}: {value}")

    def validate_config(self) -> None:
        """Ensure the configuration contains required keys."""
        required_keys = ["train", "test", "delta_model"]
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ValueError(f"Configuration is missing required keys: {missing_keys}")
