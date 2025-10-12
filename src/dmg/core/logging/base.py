from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Abstract base class for dMG loggers."""

    def __init__(self, config=None):
        self.config = config or {}

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """Log scalar metrics (e.g., loss, accuracy)."""
        pass

    @abstractmethod
    def log_config(self, config: dict):
        """Log experiment configuration."""
        pass

    @abstractmethod
    def finalize(self):
        """Clean up resources (e.g., close files or sessions)."""
        pass
