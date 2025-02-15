import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm

from core.utils import save_outputs
from core.utils.factory import import_data_sampler
from models.model_handler import ModelHandler
from trainers.base import BaseTrainer

log = logging.getLogger(__name__)


class MsTrainer(BaseTrainer):
    """Generic, unified trainer for neural networks and differentiable models.

    Inspired by the Hugging Face Trainer class.
    
    Retrieves and formats data, initializes optimizers/schedulers/loss functions,
    and runs training and testing/inference loops.
    
    Parameters
    ----------
    config : dict
        Configuration settings for the model and experiment.
    model : torch.nn.Module, optional
        Learnable model object. If not provided, a new model is initialized.
    train_dataset : dict, optional
        Training dataset dictionary.
    eval_dataset : dict, optional
        Testing/inference dataset dictionary.
    inf_dataset : dict, optional
        Inference dataset dictionary.
    loss_func : torch.nn.Module, optional
        Loss function object. If not provided, a new loss function is initialized.
    optimizer : torch.optim.Optimizer, optional
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler : torch.nn.Module, optional
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    verbose : bool, optional
        Whether to print verbose output. Default is False.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    NOTE: Training method and sampler implementation coming at a later date.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        inf_dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        self.config = config
        self.model = model or ModelHandler(config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inf_dataset = inf_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a model state optimizer."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")

    def init_scheduler(self) -> None:
        """Initialize a learning rate scheduler for the optimizer."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")
    
    def load_states(self) -> None:
        """Load model, optimizer, and scheduler states from a checkpoint."""
        path = self.config['model_path']
        for file in os.listdir(path):
            if 'train_state' and str(self.start_epoch-1) in file:
                checkpoint = torch.load(os.path.join(path, file))
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                continue
        
        # raise FileNotFoundError(f"No checkpoint for epoch {self.start_epoch-1}.") ## TODO: Fix resume model training

        # Restore random states
        torch.set_rng_state(checkpoint['random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    def train(self) -> None:
        """Entry point for training loop."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")

    def evaluate(self) -> None:
        """Run model evaluation and return both metrics and model outputs."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch.
        n_samples = self.inf_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['predict']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Forward loop
        log.info(f"Begin forward on {len(batch_start)} batches...")
        for i in tqdm.tqdm(range(len(batch_start)), desc='Inference', leave=False, dynamic_ncols=True):
            self.current_batch = i

            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                self.inf_dataset,
                batch_start[i],
                batch_end[i],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch predictions
            model_name = self.config['dpl_model']['phy_model']['model'][0]
            prediction = {key: tensor.cpu().detach() for key, tensor in prediction[model_name].items()}
            batch_predictions.append(prediction)
        
        # Save predictions
        log.info("Saving model results")
        save_outputs(self.config, batch_predictions)
        self.predictions = self._batch_data(batch_predictions)
        
        return self.predictions
    
    def _batch_data(
        self,
        batch_list: List[Dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary."""
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
        
        except Exception as e:
            raise ValueError(f"Error concatenating batch data: {e}")

    def evaluation_loop(self) -> None:
        """Inference loop used in .evaluate() and .predict() methods."""
        return NotImplementedError
    
    def calc_metrics(
        self,
        batch_predictions: List[Dict[str, torch.Tensor]],
        observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: Dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log statistics after each epoch."""
        raise NotImplementedError("Method not implemented. Multiscale training will be enabled at a later date.")
