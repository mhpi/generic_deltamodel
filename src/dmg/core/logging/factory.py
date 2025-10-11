from .base import BaseLogger
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandbLogger


def get_exp_logger(name: str = "none", **kwargs) -> BaseLogger:
    """Factory function to get the appropriate logger based on the name."""
    name = (name or "none").lower()

    if name == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif name == "wandb":
        return WandbLogger(**kwargs)
    elif name == "none":
        return NullLogger()
    else:
        raise ValueError(f"Unknown logger backend: {name}")


class NullLogger(BaseLogger):
    """A no-op logger (useful for disabling logging)."""

    def log_metrics(self, metrics, step=None):
        """No-op metrics logging."""
        pass

    def log_config(self, config):
        """No-op config logging."""
        pass

    def finalize(self):
        """No-op finalize method."""
        pass
