import logging
from typing import Any, Optional

import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.utils.factory import import_data_sampler
from dmg.core.utils.utils import save_outputs
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class MsTrainer(BaseTrainer):
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.
    
    Parameters
    ----------
    config
        Configuration settings for the model and experiment.
    model
        Learnable model object. If not provided, a new model is initialized.
    train_dataset
        Training dataset dictionary.
    eval_dataset
        Testing/inference dataset dictionary.
    dataset
        Inference dataset dictionary.
    optimizer
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    NOTE: Training method and sampler implementation coming at a later date.
    """
    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a model state optimizer."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")

    def init_scheduler(self) -> None:
        """Initialize a learning rate scheduler for the optimizer."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")

    def train(self) -> None:
        """Entry point for training loop."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")

    def evaluate(self) -> None:
        """Run model evaluation and return both metrics and model outputs."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch.
        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['simulation']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Forward loop
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Save predictions
        log.info("Saving model outputs")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)

        return self.predictions

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.
        
        Parameters
        ----------
        batch_list
            List of dictionaries containing batch data.
        target_key
            Key to extract from each batch dictionary.
        """
        data = {}
        try:
            if target_key:
                return torch.cat([x[target_key] for x in batch_list], dim=1).numpy()

            for key in batch_list[0].keys():
                if len(batch_list[0][key].shape) == 3:
                    dim = 1
                else:
                    dim = 0
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data

        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray,
    ) -> None:
        """Forward loop used in model evaluation and inference.
        
        Parameters
        ----------
        data
            Dictionary containing model input data.
        batch_start
            Start indices for each batch.
        batch_end
            End indices for each batch.
        """
        # Track predictions accross batches
        batch_predictions = []

        for i in tqdm.tqdm(range(len(batch_start)), desc='Forwarding', leave=False, dynamic_ncols=True):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[i],
                batch_end[i],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch predictions
            model_name = self.config['delta_model']['phy_model']['model'][0]
            prediction = {
                key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()
            }
            batch_predictions.append(prediction)
        return batch_predictions


    def calc_metrics(self) -> None:
        """Calculate and save model performance metrics."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")
    def _log_epoch_stats(self) -> None:
        """Log statistics after each epoch."""
        raise NotImplementedError("Method not implemented. Multiscale training"/
                                  " will be enabled at a later date.")
