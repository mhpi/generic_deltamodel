from .base import BaseLogger


class TensorBoardLogger(BaseLogger):
    """Logger for TensorBoard."""

    def __init__(self, log_dir='runs', config=None):
        super().__init__(config)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoard not installed. Please run `uv pip install tensorboard`."
            ) from e

        self.writer = SummaryWriter(log_dir=log_dir)

        config = config or {}
        self.log_config(config)

    def log_metrics(self, metrics, step=None):
        """Log metrics to TensorBoard."""
        for key, val in metrics.items():
            self.writer.add_scalar(key, val, step)

    def log_config(self, config):
        """Log configuration to TensorBoard."""
        # Flatten the config for better readability in TensorBoard's text view
        flat_config = self._flatten_dict(config)
        text = "\n".join(f"{k}: {v}" for k, v in flat_config.items())
        # Using a markdown-formatted code block for better rendering
        self.writer.add_text("Configuration", f"```\n{text}\n```")

    def finalize(self):
        """Finalize the TensorBoard writer."""
        self.writer.close()

    def _flatten_dict(self, d, parent_key='', sep='.'):
        """Flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
