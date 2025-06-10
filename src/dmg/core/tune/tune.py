import logging
import os

import torch
from ray import tune

from dmg.core.utils.factory import import_data_loader, import_trainer
from dmg.models.model_handler import ModelHandler as dModel

log = logging.getLogger(__name__)


class RayTrainable(tune.Trainable):
    """Trainer class for Ray Tune."""
    def setup(self, config):
        """Setup the trainer."""
        self.config = config
        self.epoch = 0

        # Load model
        self.model = dModel(config, verbose=self.config['verbose'])

        # Load data
        data_loader_cls = import_data_loader(self.config['data_loader'])
        self.data_loader = data_loader_cls(self.config)

        # Load trainer
        trainer_cls = import_trainer(config['trainer'])
        self.trainer = trainer_cls(
            self.config,
            self.model,
            train_dataset=self.data_loader.train_dataset,
            eval_dataset=self.data_loader.eval_dataset,
            dataset=self.data_loader.dataset,
        )

    def step(self):
        """Train one epoch and return loss."""
        self.trainer.train_one_epoch()
        return {"loss": self.trainer.total_loss}

    def save_checkpoint(self, checkpoint_dir):
        """Save torch model to checkpoint path."""
        path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), path)
        return path

    def load_checkpoint(self, checkpoint_path):
        """Load torch model from checkpoint path."""
        self.model.load_state_dict(torch.load(checkpoint_path))
