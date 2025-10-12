from .base import BaseLogger


class WandbLogger(BaseLogger):
    """Logger for Weights & Biases (wandb)."""

    def __init__(self, project='dmg', config=None, **kwargs):
        super().__init__(config)
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "Weights & Biases not installed. Try `uv pip install wandb`."
            ) from e

        self.wandb = wandb
        self.run = wandb.init(project=project, config=config, **kwargs)

        config = config or {}
        self.log_config(config)

    def log_metrics(self, metrics, step=None):
        """Log metrics to wandb."""
        self.wandb.log(metrics, step=step)

    def log_config(self, config):
        """Log configuration to wandb."""
        self.wandb.config.update(config, allow_val_change=True)

    def finalize(self):
        """Finalize the wandb run."""
        self.wandb.finish()
