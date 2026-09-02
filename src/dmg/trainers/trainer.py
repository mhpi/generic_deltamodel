import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from numpy.typing import NDArray

from dmg.core.calc.metrics import Metrics
from dmg.core.data import create_training_grid
from dmg.core.utils.factory import import_data_sampler, load_criterion
from dmg.core.utils.utils import save_outputs, save_train_state
from dmg.models.model_handler import ModelHandler
from dmg.trainers.base import BaseTrainer

log = logging.getLogger('trainer')


# try:
#     from ray import tune
#     from ray.air import Checkpoint
# except ImportError:
#     log.warning('Ray Tune is not installed or is misconfigured. Tuning will be disabled.')


class Trainer(BaseTrainer):
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
    loss_func
        Loss function object. If not provided, a new loss function is initialized.
    optimizer
        Optimizer object for learning model states. If not provided, a new
        optimizer is initialized.
    scheduler
        Learning rate scheduler. If not provided, a new scheduler is initialized.
    write_out
        Whether to save model outputs and metrics to disk.
    verbose
        Whether to print verbose output.

    TODO: Incorporate support for validation loss and early stopping in
    training loop. This will also enable using ReduceLROnPlateau scheduler.
    """

    def __init__(
        self,
        config: dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.nn.Module] = None,
        write_out: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__(config, model)
        self.model = self.model or ModelHandler(config, verbose=verbose)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.dataset = dataset
        self.optimizer = optimizer  # overrides base None if caller supplies one
        self.scheduler = scheduler
        self.write_out = write_out
        self.verbose = verbose
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        self.is_in_train = False
        self.exp_logger = None

        if 'train' in config['mode']:
            if not self.train_dataset:
                raise ValueError("'train_dataset' required for training mode.")

            log.info("Initializing experiment")
            self.epochs = self.config['train']['epochs']

            # Loss function
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['train']['loss_function'],
                device=config['device'],
            )
            self.model.loss_func = self.loss_func

            # Optimizer and learning rate scheduler
            self.optimizer = optimizer or self.init_optimizer()
            if config['train']['lr_scheduler']:
                self.use_scheduler = True
                self.scheduler = scheduler or self.init_scheduler()
            else:
                self.use_scheduler = False

            # Resume model training by loading prior states.
            self.start_epoch = self.config['train']['start_epoch'] + 1
            if self.start_epoch > 1:
                self.load_states()

            self._init_loggers()
            self._init_loss_tracking()

    def _init_loss_tracking(self) -> None:
        """Initialize loss history lists and CSV log file."""
        self.train_loss_history: list[float] = []
        self.loss_component_history: dict[str, list[float]] = {}

        if self.write_out:
            self.plot_dir = self.config['plot_dir']

            self.csv_log_file = os.path.join(
                self.config['output_dir'], 'training_log.csv'
            )
            with open(self.csv_log_file, 'w') as f:
                f.write('epoch,batch,loss,time_s,gpu_mem_mb\n')

    def init_optimizer(self) -> torch.optim.Optimizer:
        """Initialize a state optimizer.

        Adding additional optimizers is possible by extending the optimizer_dict.

        Returns
        -------
        torch.optim.Optimizer
            Initialized optimizer object.
        """
        name = self.config['train']['optimizer']['name']
        learning_rate = self.config['train']['lr']
        optimizer_dict = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'Adadelta': torch.optim.Adadelta,
            'RMSprop': torch.optim.RMSprop,
        }

        # Fetch optimizer class
        cls = optimizer_dict.get(name)
        if cls is None:
            raise ValueError(
                f"Optimizer '{name}' not recognized. "
                f"Available options are: {list(optimizer_dict.keys())}",
            )

        # Forward any extra optimizer settings (momentum, weight_decay, betas).
        opt_kwargs = {
            k: v for k, v in self.config['train']['optimizer'].items() if k != 'name'
        }

        # Initialize
        try:
            self.optimizer = cls(
                self.model.get_parameters(),
                lr=learning_rate,
                **opt_kwargs,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing optimizer: {e}") from e
        return self.optimizer

    def init_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Initialize a learning rate scheduler for the optimizer.

        torch.optim.lr_scheduler.LRScheduler
            Initialized learning rate scheduler object.
        """
        params = self.config['train']['lr_scheduler'].copy()
        name = params.pop('name')
        scheduler_dict = {
            'StepLR': torch.optim.lr_scheduler.StepLR,
            'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR,
            # 'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
        }

        # Fetch scheduler class
        cls = scheduler_dict.get(name)
        if cls is None:
            raise ValueError(
                f"Scheduler '{name}' not recognized. "
                f"Available options are: {list(scheduler_dict.keys())}",
            )

        # Initialize
        try:
            self.scheduler = cls(
                self.optimizer,
                **params,
            )
        except RuntimeError as e:
            raise RuntimeError(f"Error initializing scheduler: {e}") from e
        return self.scheduler

    def load_states(self) -> None:
        """
        Load optimizer, scheduler, and RNG states from the checkpoint for
        ``start_epoch - 1`` so training can resume from that epoch.

        The matching checkpoint file is written by
        :func:`dmg.core.utils.utils.save_train_state` as
        ``trainer_state_ep{N}.pt`` in ``self.config['model_dir']``.
        """
        path = self.config.get('pretrained_model_dir') or self.config['model_dir']
        prev_epoch = self.start_epoch - 1
        target = os.path.join(path, f'trainer_state_ep{prev_epoch}.pt')
        if not os.path.exists(target):
            raise FileNotFoundError(
                f"No checkpoint trainer_state_ep{prev_epoch}.pt in {path} for epoch {prev_epoch}.",
            )
        log.info(
            f"Loading trainer states --> Resuming training from epoch {self.start_epoch}",
        )
        checkpoint = torch.load(target)

        # Restore optimizer / scheduler.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore RNG state so minibatch sampling stays reproducible across resume.
        torch.set_rng_state(checkpoint['random_state'])
        if torch.cuda.is_available() and 'cuda_random_state' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_random_state'])

    def train(self) -> None:
        """Train the model."""
        self.is_in_train = True

        # Setup a training grid (number of samples, minibatches, and timesteps)
        n_samples, n_minibatch, n_timesteps = create_training_grid(
            self.train_dataset['xc_nn_norm'],
            self.config,
        )

        log.info(
            f"Training model: Beginning {self.start_epoch} of {self.epochs} epochs",
        )

        # Training loop
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Disable garbage collection during the epoch for performance.
            gc.collect()
            gc.disable()

            self.train_one_epoch(
                epoch,
                n_samples,
                n_minibatch,
                n_timesteps,
            )

            gc.enable()
            gc.collect()

        self.exp_logger.finalize()

    def _plot_loss_curves(self) -> None:
        """Generate and save training loss plots (linear and log scale)."""
        if not self.train_loss_history:
            return

        epochs = range(1, len(self.train_loss_history) + 1)
        save_path = Path(self.plot_dir) / 'loss_plot.png'

        multi_model = len(self.loss_component_history) > 1

        for log_scale, suffix in [(False, ''), (True, '_log')]:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                epochs,
                self.train_loss_history,
                label='Total Loss' if multi_model else None,
                color='blue',
                linewidth=1.5,
            )

            if multi_model:
                colors = ['orange', 'green', 'red', 'purple', 'brown']
                for i, (name, losses) in enumerate(self.loss_component_history.items()):
                    ax.plot(
                        epochs,
                        losses,
                        label=name,
                        color=colors[i % len(colors)],
                        linewidth=1.5,
                        linestyle='--',
                    )

            title = 'Training Loss'
            if log_scale:
                ax.set_yscale('log')
                title += ' (Log Scale)'
                ax.grid(True, which='both', ls='--', linewidth=0.5, alpha=0.7)
            else:
                ax.grid(True, ls='--', linewidth=0.5, alpha=0.7)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            if multi_model:
                ax.legend(loc='upper right', fontsize=10)
            fig.tight_layout()

            out = save_path.with_stem(f"{save_path.stem}{suffix}")
            fig.savefig(out, dpi=150)
            plt.close(fig)

        log.info(f"Loss plots saved to {self.plot_dir}")

    def train_one_epoch(self, epoch, n_samples, n_minibatch, n_timesteps) -> None:
        """Train model for one epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        n_samples
            Number of samples in the training dataset.
        n_minibatch
            Number of minibatches in the training dataset.
        n_timesteps
            Number of timesteps in the training dataset.
        """
        start_time = time.perf_counter()

        self.current_epoch = epoch
        self.total_loss = 0.0

        if hasattr(self.model, 'loss_dict'):
            for key in self.model.loss_dict:
                self.model.loss_dict[key] = 0.0

        prog_bar = tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        # Iterate through epoch in minibatches.
        for mb in prog_bar:
            self.current_batch = mb
            batch_start = time.perf_counter()

            dataset_sample = self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps,
            )

            # Forward pass through model.
            _ = self.model(dataset_sample)
            loss = self.model.calc_loss(dataset_sample)

            loss.backward()

            # Defensive: skip optimizer step when the loss or any grad is
            # non-finite. Otherwise a single bad batch poisons the optimizer
            # accumulator (Adadelta in particular) and every subsequent batch
            # produces NaN. Common in physics-coupled losses with extreme
            # parameter regions.
            batch_loss = loss.item()
            loss_finite = batch_loss == batch_loss and batch_loss not in (
                float('inf'),
                float('-inf'),
            )
            if loss_finite:
                # Optional gradient clipping (default: off when grad_clip <= 0).
                # Helps cap damage from rare large-gradient outliers.
                max_norm = float(
                    self.config['train'].get('grad_clip')
                    # `grad_threshold` is the older spelling; still honored.
                    or self.config['train'].get('grad_threshold', 0.0),
                )
                if max_norm > 0:
                    # Pull params from the optimizer's own param_groups -- this
                    # guarantees alignment with what the optimizer will step,
                    # and avoids touching `ModelHandler.get_parameters` (which
                    # has a side effect: it assigns `self.parameters = []`,
                    # shadowing the inherited nn.Module method).
                    clip_params = [
                        p for g in self.optimizer.param_groups for p in g['params']
                    ]
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=max_norm)
                self.optimizer.step()
                self._consecutive_bad_batches = 0
                self.total_loss += batch_loss
            else:
                self._consecutive_bad_batches = (
                    getattr(self, '_consecutive_bad_batches', 0) + 1
                )
                log.warning(
                    f"Non-finite loss at epoch {epoch} batch {mb}; "
                    f"skipping optimizer step "
                    f"(consecutive bad batches: {self._consecutive_bad_batches})",
                )
                bad_limit = int(self.config['train'].get('max_bad_batches', 20))
                if self._consecutive_bad_batches >= bad_limit:
                    raise RuntimeError(
                        f"Aborting training: {self._consecutive_bad_batches} "
                        f"consecutive non-finite-loss batches at epoch {epoch}. "
                        f"Last good checkpoint should be near epoch {epoch - 1}.",
                    )
            self.optimizer.zero_grad()

            if self.write_out:
                batch_elapsed = time.perf_counter() - batch_start
                mem = 0
                if self.config['device'] != 'cpu':
                    mem = int(
                        torch.cuda.memory_reserved(device=self.config['device']) * 1e-6
                    )
                with open(self.csv_log_file, 'a') as f:
                    f.write(
                        f"{epoch},{mb},{batch_loss:.6f},{batch_elapsed:.2f},{mem}\n"
                    )

            if self.verbose:
                tqdm.tqdm.write(f"Epoch {epoch}, batch {mb} | loss: {loss.item()}")

        if self.use_scheduler:
            self.scheduler.step()

        if self.verbose:
            log.info(
                f"\n ---- \n Epoch {epoch} | Total Loss {self.total_loss} "
                f"| Avg Loss {self.total_loss / n_minibatch:.6f} \n ---- \n",
            )
        self._log_epoch_stats(epoch, self.model.loss_dict, n_minibatch, start_time)

        # Save model and trainer states.
        if (epoch % self.config['train']['save_epoch'] == 0) and self.write_out:
            self.model.save_model(epoch)
            save_train_state(
                self.config['model_dir'],
                epoch=epoch,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                clear_prior=True,
            )

            # if self.config['do_tune']:
            #     # Create temporary checkpoint if needed
            #     chkpt = None
            #     if epoch % self.calc_metricsconfig['tune']['save_epoch'] == 0:
            #         with tempfile.TemporaryDirectory() as temp_dir:
            #             model_path = os.path.join(temp_dir, "model_ep{epoch}.pt")
            #             torch.save(self.model.state_dict(), model_path)
            #             chkpt = Checkpoint.from_directory(temp_dir)

            #     # Report to Ray Tune
            #     tune.report(loss=self.total_loss, checkpoint=chkpt)

    def _apply_denorm(
        self,
        predictions: dict[str, np.ndarray],
        dataset: dict,
    ) -> dict[str, np.ndarray]:
        """Apply denormalization to the target key in batched predictions.

        Converts ML model predictions from normalized space to physical
        units (mm/day or the configured output unit). Physics model
        predictions are already in physical units and pass through unchanged.

        Parameters
        ----------
        predictions
            Batched predictions dict (key -> numpy array).
        dataset
            Dataset dict, may contain a 'denorm_fn' key.

        Returns
        -------
        dict[str, np.ndarray]
            Predictions with the target key denormalized.
        """
        denorm_fn = dataset.get('denorm_fn')
        if denorm_fn is None:
            return predictions

        target_name = self.config['train']['target'][0]
        if target_name not in predictions:
            return predictions

        pred = predictions[target_name]
        needs_squeeze = pred.ndim == 2
        if needs_squeeze:
            pred = np.expand_dims(pred, 2)
        pred = denorm_fn(pred)
        if needs_squeeze:
            pred = pred.squeeze(2)
        predictions[target_name] = pred
        return predictions

    def evaluate(self) -> None:
        """Run model evaluation and return both metrics and model outputs."""
        self.is_in_train = False

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.eval_dataset['target']

        # Get start and end indices for each batch
        n_samples = self.eval_dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(
            self.eval_dataset,
            batch_start,
            batch_end,
        )

        # Batch, denormalize, save, and compute metrics
        log.info("Saving model outputs + Calculating metrics")
        self.model.save_states()
        self.predictions = self._batch_data(batch_predictions)
        self.predictions = self._apply_denorm(self.predictions, self.eval_dataset)

        save_outputs(self.config, batch_predictions, observations)
        self._save_denormed_target(self.predictions)

        # Convert observations to output unit before computing metrics so they
        # match the (potentially denormalized) predictions.
        obs_np = observations.cpu().numpy()
        obs_convert_fn = self.eval_dataset.get('obs_convert_fn')
        if obs_convert_fn is not None:
            obs_np = obs_convert_fn(obs_np)

        # Align pred/obs time axes (handles both full-window and post-warm-up
        # model conventions; see Trainer._align_for_metrics docstring).
        self.predictions, obs_np = self._align_for_metrics(self.predictions, obs_np)

        # Calculate metrics
        self.calc_metrics(self.predictions, obs_np)

    def inference(self) -> None:
        """Run batch model inference and save model outputs."""
        self.is_in_train = False

        # Track overall predictions
        batch_predictions = []

        # Get start and end indices for each batch
        n_samples = self.dataset['xc_nn_norm'].shape[1]
        batch_start = np.arange(0, n_samples, self.config['sim']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Inference: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.dataset, batch_start, batch_end)

        # Batch, denormalize, and save
        log.info("Saving model outputs")
        self.model.save_states()
        self.predictions = self._batch_data(batch_predictions)
        self.predictions = self._apply_denorm(self.predictions, self.dataset)

        save_outputs(self.config, batch_predictions)
        self._save_denormed_target(self.predictions)

        return self.predictions

    def _save_denormed_target(
        self,
        predictions: dict[str, np.ndarray],
    ) -> None:
        """Overwrite the saved target prediction file with denormalized values.

        Only writes if denormalization was applied (i.e., the prediction
        values differ from what save_outputs wrote).
        """
        target_name = self.config['train']['target'][0]
        if target_name in predictions:
            np.save(
                os.path.join(self.config['sim_dir'], f'{target_name}.npy'),
                predictions[target_name],
            )

    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> None:
        """Merge batch data into a single dictionary.

        Parameters
        ----------
        batch_list
            List of dictionaries from each forward batch containing inputs and
            model predictions.
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
                data[key] = (
                    torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
                )
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

        prog_bar = tqdm.tqdm(
            range(len(batch_start)),
            desc='Forwarding',
            leave=False,
            dynamic_ncols=True,
        )

        for mb in prog_bar:
            # Select a batch of data
            dataset_sample = self.sampler.get_validation_sample(
                data,
                batch_start[mb],
                batch_end[mb],
            )

            prediction = self.model(dataset_sample, eval=True)

            # Save the batch predictions
            prediction = {
                key: tensor.detach().cpu()
                for key, tensor in prediction[self.model.models[0]].items()
                if tensor is not None
            }
            batch_predictions.append(prediction)
        return batch_predictions

    def calc_metrics(
        self,
        predictions: dict[str, np.ndarray],
        observations: np.ndarray,
    ) -> None:
        """Calculate and save model performance metrics.

        Parameters
        ----------
        predictions
            Batched (and denormalized) predictions dict.
        observations
            Target variable observation data as a numpy array, already
            converted to match the output unit of predictions AND already
            aligned to the prediction time axis (warm-up handled upstream
            in ``_align_for_metrics``).
        """
        target_name = self.config['train']['target'][0]
        pred = predictions[target_name]
        if pred.ndim == 2:
            pred = np.expand_dims(pred, 2)
        target = np.expand_dims(observations[:, :, 0], 2)

        if pred.shape != target.shape:
            raise ValueError(
                f"calc_metrics: pred shape {pred.shape} does not match "
                f"target shape {target.shape}. Models should return "
                f"post-warm-up output; use Trainer._align_for_metrics() to "
                f"reconcile legacy full-window models against post-warm-up targets."
            )

        # Compute metrics
        metrics = Metrics(
            np.swapaxes(pred.squeeze(), 1, 0),
            np.swapaxes(target.squeeze(), 1, 0),
        )

        # Save all metrics and aggregated statistics.
        metrics.dump_metrics(self.config['output_dir'])
        metrics.print_summary()

    def _align_for_metrics(
        self,
        predictions: dict[str, np.ndarray],
        observations: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Reconcile prediction and target time axes before metric scoring.

        The dMG model registry contains two conventions for what
        ``Model.forward()`` returns over a test window of length ``T``:

        - **post-warm-up**: returns ``T - warmup`` days (older Hbv_1_1p,
          most physics-based models).
        - **full-window**: returns the full ``T`` days, with the first
          ``warmup`` rows being spin-up that should not be scored
          (most pure-LSTM and newer Hbv variants).

        This helper detects which convention the active model uses (by
        comparing pred and target lengths) and strips warm-up symmetrically
        so ``calc_metrics`` receives matched-shape arrays. Going forward, new
        models should prefer the post-warm-up convention.

        Parameters
        ----------
        predictions
            Batched (and denormalized) predictions dict; values are
            ``(T_pred, N, C)`` arrays.
        observations
            Target observation array of shape ``(T_obs, N, num_targets)``.

        Returns
        -------
        Tuple of ``(predictions, observations)`` with their first axes aligned.
        """
        target_name = self.config['train']['target'][0]
        warmup = int(self.config['model'].get('warmup', 0))
        pred = predictions[target_name]
        T_pred = pred.shape[0]
        T_obs = observations.shape[0]

        if T_pred == T_obs:
            # Both full-window (or both already post-warm-up); strip warm-up
            # symmetrically from both.
            if warmup > 0:
                predictions = {
                    k: (v[warmup:] if v.shape[0] == T_obs else v)
                    for k, v in predictions.items()
                }
                observations = observations[warmup:]
        elif T_pred == T_obs - warmup:
            # Pred is already post-warm-up; strip target only.
            observations = observations[warmup:]
        elif T_pred - warmup == T_obs:
            # Target was already stripped; strip pred too.
            predictions = {
                k: (v[warmup:] if v.shape[0] == T_pred else v)
                for k, v in predictions.items()
            }
        else:
            raise ValueError(
                f"_align_for_metrics: cannot align pred (T={T_pred}) and "
                f"target (T={T_obs}) with warmup={warmup}. Expected pred to "
                f"be post-warm-up (T_obs - warmup) or full-window (T_obs)."
            )
        return predictions, observations

    def _log_epoch_stats(
        self,
        epoch: int,
        loss_dict: dict[str, float],
        n_minibatch: int,
        start_time: float,
    ) -> None:
        """Log statistics after each epoch.

        Parameters
        ----------
        epoch
            Current epoch number.
        loss_dict
            Dictionary containing loss values.
        n_minibatch
            Number of minibatches.
        start_time
            Start time of the epoch.
        """
        avg_loss_dict = {key: value / n_minibatch for key, value in loss_dict.items()}
        avg_total_loss = self.total_loss / n_minibatch
        loss_str = ", ".join(
            f"{key}: {value:.6f}" for key, value in avg_loss_dict.items()
        )
        elapsed = time.perf_counter() - start_time
        mem_aloc = 0

        if self.config['device'] != 'cpu':
            mem_aloc = int(
                torch.cuda.memory_reserved(device=self.config['device']) * 0.000001,
            )

        log.info(
            f"Loss after epoch {epoch}: {loss_str} \n"
            f"~ Runtime {elapsed:.2f} s, {mem_aloc} Mb reserved GPU memory",
        )

        # Track loss history
        self.train_loss_history.append(avg_total_loss)
        for model_name, loss_val in avg_loss_dict.items():
            if model_name not in self.loss_component_history:
                self.loss_component_history[model_name] = []
            self.loss_component_history[model_name].append(loss_val)

        # For experiment loggers: create a single dictionary of metrics to log
        metrics_to_log = {
            'Loss/train_total': avg_total_loss,
        }
        for model_name, loss_val in avg_loss_dict.items():
            metrics_to_log[f'Loss/{model_name}'] = loss_val

        if self.use_scheduler:
            metrics_to_log['learning_rate'] = self.scheduler.get_last_lr()[0]

        # Loop through all active loggers and log the metrics
        self.exp_logger.log_metrics(metrics_to_log, step=epoch)

        # Update loss plots
        if self.write_out:
            self._plot_loss_curves()
