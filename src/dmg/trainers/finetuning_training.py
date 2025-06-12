import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from dMG.core.data.samplers.finetune_sampler import Finetuneampler
from dMG.trainers.base import BaseTrainer
from dMG.core.utils.utils import save_outputs
from dMG.core.calc.metrics import Metrics
from dMG.core.utils.factory import import_data_sampler, load_criterion
from dMG.core.data import create_training_grid
import os

log = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    def __init__(
        self,
        config: Dict[str, Any],
        model: torch.nn.Module = None,
        train_dataset: Optional[dict] = None,
        eval_dataset: Optional[dict] = None,
        dataset: Optional[dict] = None,
        loss_func: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.nn.Module] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        """Initialize the FineTune trainer.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary
        model : torch.nn.Module, optional
            The model to train/test
        train_dataset : dict, optional
            Training dataset
        eval_dataset : dict, optional
            Evaluation dataset
        dataset : dict, optional
            Complete dataset if not split
        loss_func : torch.nn.Module, optional
            Loss function
        optimizer : torch.nn.Module, optional
            Optimizer
        verbose : bool, optional
            Whether to print detailed output
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = eval_dataset
        self.dataset = dataset
        self.device = config['device']
        self.verbose = verbose
        self.is_in_train = False
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.sampler = import_data_sampler(config['data_sampler'])(config)
        
        # Determine model type based on config or model output
        self.model_type = self._determine_model_type()
        
        if 'train' in config['mode']:
            # log.info(f"Initializing loss function and optimizer for {self.model_type} model")
            
            self.loss_func = loss_func or load_criterion(
                self.train_dataset['target'],
                config['loss_function'],
                device=config['device'],
            )

            self.model.loss_func = self.loss_func
            self.optimizer = optimizer or self.init_optimizer()
            self.start_epoch = self.config['train'].get('start_epoch', 0) + 1
            
            # Add scheduler for learning rate adjustment
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('lr_patience', 5),
                factor=self.config.get('lr_factor', 0.1)
            )

    def _determine_model_type(self) -> str:
        """Determine if this is an HBV model or direct output model."""
        return 'hbv'
        # Check config first
        if 'delta_model' in self.config and 'phy_model' in self.config['delta_model']:
            if 'Hbv' in self.config['delta_model']['phy_model'].get('model', ''):
                return 'hbv'
        
        # Default to direct output
        return 'direct'

    def init_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = self.config['train']['optimizer']
        learning_rate = self.config['delta_model']['nn_model']['learning_rate']

        optimizer_dict = {
            'Adadelta': torch.optim.Adadelta,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
        }

        optimizer_cls = optimizer_dict.get(optimizer_name)
        if optimizer_cls is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not recognized. "
                        f"Available options are: {list(optimizer_dict.keys())}")

        # Collect trainable parameters using get_parameters
        trainable_params = self.model.get_parameters()

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found. Check model initialization and parameter freezing.")

        try:
            optimizer = optimizer_cls(
                trainable_params,
                lr=learning_rate
            )
            return optimizer
        except Exception as e:
            raise ValueError(f"Error initializing optimizer: {e}")

    def train(self) -> None:
        """Training loop implementation."""
        log.info(f"Training model: Beginning {self.start_epoch} of {self.config['train']['epochs']} epochs")
        
        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_file = open(os.path.join(results_dir, "results.txt"), 'a')
        results_file.write(f"\nTrain start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        use_amp = self.config.get('use_amp', False)
        if use_amp and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        
        try:
            for epoch in range(self.start_epoch, self.config['train']['epochs'] + 1):
                train_loss, val_loss = [], []
                epoch_time = time.time()

                # Training phase
                self.model.train()
                
                # Debug: Check actual data dimensions
                log.info(f"Dataset shapes:")
                for key, value in self.train_dataset.items():
                    if hasattr(value, 'shape'):
                        log.info(f"  {key}: {value.shape}")
                
                # Get grid dimensions for training - but fix the dimension issue
                n_samples, n_minibatch, n_timesteps = create_training_grid(
                    self.train_dataset['xc_nn_norm'],
                    self.config
                )
                
                # Recalculate minibatches based on correct basin count
                batch_size = self.config['train']['batch_size']
                n_minibatch = max(1, (n_samples * self.config.get('train_samples_per_epoch', 1)) // batch_size)
                
                log.info(f"Corrected training grid - basins: {n_samples}, timesteps: {n_timesteps}, batches: {n_minibatch}")

                for i in range(1, n_minibatch + 1):
                    self.optimizer.zero_grad()
                    
                    batch_data = self.sampler.get_training_sample(
                        self.train_dataset,
                        n_samples,
                        n_timesteps
                    )
                    
                    # Use AMP context manager if available
                    cm = torch.cuda.amp.autocast() if use_amp and torch.cuda.is_available() else nullcontext()
                    with cm:
                        outputs = self.model(batch_data)
                        target = batch_data['target']
                        
                        # Handle different model output types
                        if self.model_type == 'hbv':
                            # HBV model with nested output structure
                            hbv_output = outputs['Hbv_1_1p']
                            if isinstance(hbv_output, dict) and 'streamflow' in hbv_output:
                                model_output = hbv_output['streamflow']
                            else:
                                model_output = hbv_output
                        else:
                            # Direct output model (e.g., soil moisture)
                            model_output = outputs['Hbv_1_1p']

                        # print(model_output["streamflow"])
                        loss = self.loss_func(model_output,target)
                        
                        if torch.isnan(loss):
                            continue
                            
                        train_loss.append(loss.item())

                    if use_amp and torch.cuda.is_available():
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()

                # Log statistics and save model
                self._log_epoch_stats(epoch, train_loss, val_loss, epoch_time, results_file)
                
                if epoch % self.config['train']['save_epoch'] == 0:
                    self.model.save_model(epoch)

        except Exception as e:
            log.error(f"Training error: {str(e)}")
            raise
        finally:
            results_file.close()
            log.info("Training complete")

    def evaluate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced testing method that handles both variable basin counts and full time periods."""
        log.info("Starting model testing with full time period prediction...")
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get total number of basins and time steps
        total_basins = self.test_dataset['xc_nn_norm'].shape[0]
        total_timesteps = self.test_dataset['target'].shape[0]
        log.info(f"Total basins: {total_basins}, Total timesteps: {total_timesteps}")
        
        # Calculate parameters
        warm_up = self.config['delta_model']['phy_model']['warm_up']
        rho = self.config['delta_model']['rho']
        
        # Define prediction interval (can be adjusted for overlap)
        stride = rho  # Non-overlapping windows
        
        # Calculate number of time windows needed
        effective_timesteps = total_timesteps - warm_up
        n_windows = (effective_timesteps + stride - 1) // stride
        log.info(f"Predicting {n_windows} time windows to cover {effective_timesteps} effective timesteps")
        
        # Use batch size from config or adjust if needed
        batch_size = min(self.config['test'].get('batch_size', 25), total_basins)
        n_batches = (total_basins + batch_size - 1) // batch_size
        
        # Create an array to store all predictions
        all_predictions = np.zeros((effective_timesteps, total_basins, 1))
        
        # Disable gradient computation
        with torch.no_grad():
            # First loop through time windows
            for window in range(n_windows):
                # Calculate starting time index for this window
                time_start = warm_up + window * stride
                
                # Ensure we don't exceed dataset bounds
                if time_start >= total_timesteps:
                    break
                    
                # Calculate ending time index for this window
                time_end = min(time_start + rho, total_timesteps)
                actual_window_size = time_end - time_start
                
                # Inner loop through basin batches
                window_predictions = []
                
                for i in range(n_batches):
                    # Calculate basin indices for this batch
                    basin_start = i * batch_size
                    basin_end = min(basin_start + batch_size, total_basins)
                    
                    try:
                        # Create a custom function to get time-windowed validation sample
                        dataset_sample = self._get_time_window_sample(
                            self.test_dataset,
                            basin_start,
                            basin_end,
                            time_start - warm_up,  # Include warm-up period before prediction window
                            time_end
                        )
                        
                        # Forward pass
                        prediction = self.model(dataset_sample, eval=True)
                        
                        # Extract predictions based on model type
                        batch_pred = self._extract_prediction(prediction, warm_up)
                        
                        if batch_pred is not None:
                            window_predictions.append(batch_pred)
                            
                    except Exception as e:
                        log.error(f"Error processing batch {i+1} in window {window+1}: {str(e)}")
                        continue
                        
                # Combine basin predictions for this time window
                if window_predictions:
                    try:
                        # Concatenate basin predictions
                        window_pred_full = np.concatenate(window_predictions, axis=1)
                        
                        # Store in the appropriate slice of the full predictions array
                        window_offset = window * stride
                        end_offset = min(window_offset + actual_window_size, effective_timesteps)
                        pred_length = min(window_pred_full.shape[0], end_offset - window_offset)
                        
                        all_predictions[window_offset:window_offset+pred_length, :, :] = window_pred_full[:pred_length, :, :]
                        
                    except Exception as e:
                        log.error(f"Error combining predictions for window {window+1}: {str(e)}")
                else:
                    log.warning(f"No predictions generated for window {window+1}")
        
        # Get observations (removing warm-up period)
        observations = self.test_dataset['target'][warm_up:].cpu().numpy()
        
        # Ensure shapes match
        if all_predictions.shape[0] > observations.shape[0]:
            all_predictions = all_predictions[:observations.shape[0], :, :]
        elif observations.shape[0] > all_predictions.shape[0]:
            observations = observations[:all_predictions.shape[0], :, :]
        
        # Calculate metrics and save outputs
        log.info(f"Final prediction shape: {all_predictions.shape}, observation shape: {observations.shape}")
        if all_predictions.size > 0:
            output_dir = self.config['validation_path']
         
            log.info(f"Saving results to directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save raw predictions and observations
            np.save(os.path.join(output_dir, 'predictions.npy'), all_predictions)
            np.save(os.path.join(output_dir, 'observations.npy'), observations)
            log.info(f"Saved raw predictions and observations to {output_dir}")
            
            # Calculate metrics using the appropriate Metrics class
            self.calc_metrics(all_predictions, observations)
            return all_predictions, observations
        else:
            log.warning("No predictions were generated during testing")
            return None, None

    def _extract_prediction(self, prediction: dict, warm_up: int) -> Optional[np.ndarray]:
        """Extract prediction tensor based on model type."""
        if not isinstance(prediction, dict):
            log.warning(f"Unexpected prediction type: {type(prediction)}")
            return None
        
        if self.model_type == 'hbv':
            # HBV model with nested structure
            if 'Hbv_1_1p' in prediction and isinstance(prediction['Hbv_1_1p'], dict):
                if 'flow_sim' in prediction['Hbv_1_1p']:
                    batch_pred = prediction['Hbv_1_1p']['flow_sim'].cpu().numpy()
                else:
                    log.warning(f"Could not find flow_sim in HBV output. Keys: {prediction['Hbv_1_1p'].keys()}")
                    return None
            else:
                log.warning(f"Could not find Hbv_1_1p in prediction. Keys: {prediction.keys()}")
                return None
        else:
            # Direct output model (e.g., soil moisture)
            if 'Hbv_1_1p' in prediction:
                batch_pred = prediction['Hbv_1_1p'].cpu().numpy()
            else:
                log.warning(f"Could not find prediction output. Keys: {prediction.keys()}")
                return None
        
        # Remove warm-up period if present
        if batch_pred.shape[0] > warm_up:
            batch_pred = batch_pred[warm_up:, :, :]
        
        return batch_pred

    def calc_metrics(self, predictions: np.ndarray, observations: np.ndarray) -> None:
        """Calculate comprehensive metrics using the appropriate Metrics class."""
        try:
            # Ensure no NaN values in predictions
            predictions = np.nan_to_num(predictions, nan=0.0)
            
            # Format predictions and observations for the Metrics class
            # Metrics expects shape [basins, time]
            pred_formatted = np.swapaxes(predictions.squeeze(), 0, 1)
            obs_formatted = np.swapaxes(observations.squeeze(), 0, 1)
            
            log.info(f"Calculating comprehensive metrics")
            log.info(f"Formatted shapes - predictions: {pred_formatted.shape}, observations: {obs_formatted.shape}")
            
            # Use appropriate metrics class
            try:
                # Try soil moisture metrics first if it might be soil moisture data
                if self.model_type == 'direct':
                    from dMG.core.calc.metrics_soilMoisture import Metrics
                    log.info("Using soil moisture specific metrics")
                else:
                    metrics = Metrics(pred_formatted, obs_formatted)
            except ImportError:
                # Fall back to standard metrics
                log.info("Using standard metrics")
                metrics = Metrics(pred_formatted, obs_formatted)
            
            # Create Metrics object and save
            metrics = Metrics(pred_formatted, obs_formatted)
            metrics.dump_metrics(self.config['validation_path'])
            
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            import traceback
            log.error(traceback.format_exc())
    
    def inference(self) -> None:
        """Run batch model inference - required by BaseTrainer."""
        raise NotImplementedError

   

    def _get_time_window_sample(self, dataset, basin_start, basin_end, time_start, time_end):
        """Custom method to get a validation sample for a specific time window."""
        # Calculate dimensions
        batch_size = basin_end - basin_start
        seq_len = time_end - time_start
        
        batch_data = {}
        
        # Process NN data (dimensions: [basins, time, features])
        if 'xc_nn_norm' in dataset:
            batch_data['xc_nn_norm'] = dataset['xc_nn_norm'][basin_start:basin_end, time_start:time_start+seq_len].to(self.device)
            log.debug(f"Time window xc_nn_norm shape: {batch_data['xc_nn_norm'].shape}")
        
        # Process physics data (dimensions: [time, basins, features])
        if 'x_phy' in dataset:
            batch_data['x_phy'] = dataset['x_phy'][time_start:time_start+seq_len, basin_start:basin_end].to(self.device)
            log.debug(f"Time window x_phy shape: {batch_data['x_phy'].shape}")
        
        # Process static attributes
        if 'c_phy' in dataset:
            batch_data['c_phy'] = dataset['c_phy'][basin_start:basin_end].to(self.device)
        
        if 'c_nn' in dataset:
            batch_data['c_nn'] = dataset['c_nn'][basin_start:basin_end].to(self.device)
        
        # Process target data
        if 'target' in dataset:
            batch_data['target'] = dataset['target'][time_start:time_start+seq_len, basin_start:basin_end].to(self.device)
        
        return batch_data

    def _log_epoch_stats(
        self,
        epoch: int,
        train_loss: List[float],
        val_loss: Optional[List[float]],
        epoch_time: float,
        results_file
    ) -> None:
        """Log training statistics."""
        train_loss_avg = np.mean(train_loss)
        val_loss_avg = np.mean(val_loss) if val_loss else None
        
        self.epoch_train_loss_list.append(train_loss_avg)
        if val_loss_avg:
            self.epoch_val_loss_list.append(val_loss_avg)

        log_msg = (
            f"Epoch {epoch}: train_loss={train_loss_avg:.3f}"
            f"{f', val_loss={val_loss_avg:.3f}' if val_loss_avg else ''}"
            f" ({time.time() - epoch_time:.2f}s)"
        )
        
        log.info(log_msg)
        results_file.write(log_msg + "\n")
        results_file.flush()
        
        self._save_loss_data(epoch, train_loss_avg, val_loss_avg)

    def _save_loss_data(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        """Save loss data to file and plot."""
        results_dir = self.config.get('save_path', 'results')
        os.makedirs(results_dir, exist_ok=True)
        loss_data = os.path.join(results_dir, "loss_data.csv")
        
        with open(loss_data, 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss if val_loss else ''}\n")


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()