#Old working verison of code 
import logging
import time
from typing import Any, Dict, List, Optional
from contextlib import nullcontext  # Use built-in contextlib.nullcontext

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from core.data.data_samplers.finetune_sampler import FinetuneDataSampler

from trainers.base import BaseTrainer
from core.utils import save_outputs
from core.calc.metrics import Metrics
from models.loss_functions import get_loss_func
from core.data import create_training_grid
import os

log = logging.getLogger(__name__)

class FineTuneTrainer(BaseTrainer):
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
        self.sampler = FinetuneDataSampler(config)
        
        if 'train' in config['mode']:
            log.info(f"Initializing loss function and optimizer")
            # self.loss_func = RMSELoss()
            # self.model.loss_func = self.loss_func
            self.loss_func = loss_func or get_loss_func(
                self.train_dataset['target'].to('cpu'),
                config['loss_function'],
                config['device'],
            )

            self.model.loss_func = self.loss_func
            self.optimizer = optimizer or self.create_optimizer()
            self.start_epoch = self.config['train'].get('start_epoch', 0) + 1
            
            # Add scheduler for learning rate adjustment
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('lr_patience', 5),
                factor=self.config.get('lr_factor', 0.1)
            )

            
            # Create necessary directories for outputs
            # if 'validation_path' not in self.config:
            #     if 'test_mode' in self.config and self.config['test_mode']['type'] == 'spatial' and 'current_holdout_index' in self.config['test_mode']:
            #         holdout_idx = self.config['test_mode']['current_holdout_index']
            #         # Create a dedicated output directory for this specific holdout
            #         holdout_dir = os.path.join(self.config['save_path'], f'spatial_holdout_{holdout_idx}')
            #         self.config['validation_path'] = os.path.join(holdout_dir, 'validation')
            #         self.config['testing_path'] = os.path.join(holdout_dir, 'testing')
            #         self.config['out_path'] = holdout_dir
            #     else:
            #         # Default paths if not spatial holdout testing
            #         self.config['validation_path'] = os.path.join(self.config['save_path'], 'validation')
            #         self.config['testing_path'] = os.path.join(self.config['save_path'], 'testing')
            #         self.config['out_path'] = self.config['save_path']
                
            # # Create output directories
            # os.makedirs(self.config['validation_path'], exist_ok=True)
            # os.makedirs(self.config['testing_path'], exist_ok=True)
    def create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = self.config['train']['optimizer']
        learning_rate = self.config['dpl_model']['nn_model']['learning_rate']

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
        # log.info(f"cliping grad {self.config.get('clip_grad')}")
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
                data_shape = self.train_dataset['xc_nn_norm'].shape
               # Get grid dimensions for training
                n_timesteps, n_minibatch, n_samples = create_training_grid(
                    self.train_dataset['xc_nn_norm'],
                    self.config
                )
                log.info(f"Training grid - samples: {n_samples}, batches: {n_minibatch}, timesteps: {n_timesteps}")

                for i in range(1, n_minibatch + 1):
                    self.optimizer.zero_grad()
                    
                    batch_data = self.sampler.get_training_sample(self.train_dataset,
                                                                        n_samples,
                                                                        n_timesteps
                                                                    )
                    
                    # Use AMP context manager if available
                    cm = torch.cuda.amp.autocast() if use_amp and torch.cuda.is_available() else nullcontext()
                    with cm:
                        outputs = self.model(batch_data)

                        target = batch_data['target']
                        hbv_output = outputs['HBV_1_1p']
                            
                        # if not isinstance(hbv_output, dict) or 'flow_sim' not in hbv_output:
                        #     raise ValueError(f"Invalid HBV output structure. Got {type(hbv_output)}")
                        # print(batch_data)

                        loss = self.loss_func(hbv_output['flow_sim'],target,batch_data['batch_sample'])
                        
                        if torch.isnan(loss):
                            continue
                            
                        train_loss.append(loss.item())

                    if use_amp and torch.cuda.is_available():
                        scaler.scale(loss).backward()
                        # if self.config.get('clip_grad'):
                        #     scaler.unscale_(self.optimizer)
                        #     torch.nn.utils.clip_grad_norm_(
                        #         self.model.parameters(), 
                        #         self.config['clip_grad']
                        #     )
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        # if self.config.get('clip_grad'):
                        #     torch.nn.utils.clip_grad_norm_(
                        #         self.model.parameters(), 
                        #         self.config['clip_grad']
                        #     )
                        self.optimizer.step()

                # Validation phase
                # if self.config.get('do_eval', True):
                #     val_loss = self._validate()
                    
                # Log statistics and save model
                self._log_epoch_stats(epoch, train_loss, val_loss, epoch_time, results_file)
                
                if epoch % self.config['train']['save_epoch'] == 0:
                    self.model.save_model(epoch)

                # Adjust learning rate
                # if val_loss:
                #     self.scheduler.step(np.mean(val_loss))

                # clear_gpu_memory()

        except Exception as e:
            log.error(f"Training error: {str(e)}")
            # clear_gpu_memory()
            raise
        finally:
            results_file.close()
            log.info("Training complete")



    def test(self):
        """Enhanced testing method that handles both variable basin counts and full time periods."""
        log.info("Starting model testing with full time period prediction...")
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Get total number of basins and time steps
        total_basins = self.test_dataset['xc_nn_norm'].shape[0]
        total_timesteps = self.test_dataset['target'].shape[0]
        log.info(f"Total basins: {total_basins}, Total timesteps: {total_timesteps}")
        
        # Calculate parameters
        warm_up = self.config['dpl_model']['phy_model']['warm_up']
        rho = self.config['dpl_model']['rho']
        seq_len = warm_up + rho
        
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
                
                # log.info(f"Processing time window {window+1}/{n_windows} (timesteps {time_start} to {time_end})")
                
                # Inner loop through basin batches
                window_predictions = []
                
                for i in range(n_batches):
                    # Calculate basin indices for this batch
                    basin_start = i * batch_size
                    basin_end = min(basin_start + batch_size, total_basins)
                    
                    # log.info(f"  Processing basin batch {i+1}/{n_batches} (basins {basin_start} to {basin_end-1})")
                    
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
                        
                        # Extract predictions (only take the part after warm-up)
                        if isinstance(prediction, dict):
                            if 'HBV_1_1p' in prediction and 'flow_sim' in prediction['HBV_1_1p']:
                                # Get the prediction and remove the warm-up period
                                batch_pred = prediction['HBV_1_1p']['flow_sim'].cpu().numpy()
                                if batch_pred.shape[0] > warm_up:
                                    batch_pred = batch_pred[warm_up:, :, :]
                                
                                # Store the batch prediction
                                window_predictions.append(batch_pred)
                            else:
                                log.warning(f"Could not find prediction. Output keys: {prediction.keys()}")
                                continue
                        else:
                            log.warning(f"Unexpected prediction type: {type(prediction)}")
                            continue
                            
                    except Exception as e:
                        log.error(f"Error processing batch {i+1} in window {window+1}: {str(e)}")
                        import traceback
                        log.error(traceback.format_exc())
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
                        
                        # log.info(f"  Added predictions for window {window+1} to positions {window_offset} through {window_offset+pred_length-1}")
                        
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
            
            # Calculate metrics using the Metrics class
            self.calc_metrics(all_predictions, observations)
            return all_predictions, observations
        else:
            log.warning("No predictions were generated during testing")
            return none, none
        
       

    def calc_metrics(self, predictions: np.ndarray, observations: np.ndarray) -> None:
        """Calculate comprehensive metrics using the Metrics class.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions with shape [time, basins, 1]
        observations : np.ndarray
            Target observations with shape [time, basins, 1]
        """
        try:
            # Ensure no NaN values in predictions
            predictions = np.nan_to_num(predictions, nan=0.0)
            
            # Format predictions and observations for the Metrics class
            # Metrics expects shape [basins, time]
            pred_formatted = np.swapaxes(predictions.squeeze(), 0, 1)
            obs_formatted = np.swapaxes(observations.squeeze(), 0, 1)
            
            log.info(f"Calculating comprehensive metrics")
            log.info(f"Formatted shapes - predictions: {pred_formatted.shape}, observations: {obs_formatted.shape}")
            
            # Create Metrics object
            metrics = Metrics(pred_formatted, obs_formatted)
            
            # Save metrics to the specified output directory
            metrics.dump_metrics(self.config['validation_path'])
            
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            import traceback
            log.error(traceback.format_exc())
            
            # Fall back to basic metrics calculation if the comprehensive method fails
            self._calculate_basic_metrics(predictions, observations)

    def _calculate_basic_metrics(self, predictions: np.ndarray, observations: np.ndarray) -> None:
        """Calculate basic metrics as a fallback.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions with shape [time, basins, 1]
        observations : np.ndarray
            Target observations with shape [time, basins, 1]
        output_dir : str
            Directory to save metrics files
        """
        log.info("Falling back to basic metrics calculation")
        
        # Flatten predictions and observations for metrics
        pred_flat = predictions.flatten()
        obs_flat = observations.flatten()
        
        # Handle NaN values
        mask = ~np.isnan(obs_flat) & ~np.isnan(pred_flat)
        pred_flat = pred_flat[mask]
        obs_flat = obs_flat[mask]
        
        # Correlation
        correlation = np.corrcoef(pred_flat, obs_flat)[0, 1]
        log.info(f"Correlation Coefficient: {correlation:.4f}")
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(pred_flat - obs_flat))
        log.info(f"Mean Absolute Error: {mae:.4f}")
        
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((pred_flat - obs_flat)**2))
        log.info(f"Root Mean Squared Error: {rmse:.4f}")
        
        # Nash-Sutcliffe Efficiency (NSE)
        obs_mean = np.mean(obs_flat)
        nse = 1 - (np.sum((obs_flat - pred_flat)**2) / np.sum((obs_flat - obs_mean)**2))
        log.info(f"Nash-Sutcliffe Efficiency: {nse:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(self.config['validation_path'], 'test_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Correlation Coefficient: {correlation:.4f}\n")
            f.write(f"Mean Absolute Error: {mae:.4f}\n")
            f.write(f"Root Mean Squared Error: {rmse:.4f}\n")
            f.write(f"Nash-Sutcliffe Efficiency: {nse:.4f}\n")
        
        log.info(f"Basic metrics saved to {metrics_file}")

    def _get_time_window_sample(self, dataset, basin_start, basin_end, time_start, time_end):
        """Custom method to get a validation sample for a specific time window."""
        # log.info(f"Creating time window sample for basins {basin_start} to {basin_end-1}, time {time_start} to {time_end}")
        
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
  
        
    def _get_batch_data(self, n_samples: int, n_timesteps: int) -> Dict[str, torch.Tensor]:
        """Get a batch of training data."""
        batch_data = {
            key: value.float().to(self.device)
            for key, value in self.sampler.get_training_sample(
                self.train_dataset,
                n_samples,
                n_timesteps
            ).items()
            if torch.is_tensor(value)
        }
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
            
   
# class NSELoss(torch.nn.Module):
#     """Nash-Sutcliffe Efficiency Loss."""
#     def forward(self, pred_dict: Dict[str, Any], target: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate NSE loss.
        
#         Parameters
#         ----------
#         pred_dict : Dict[str, Any]
#             Model predictions with HBV['flow_sim'] tensor
#         target : torch.Tensor
#             Target values [time, batch, 1]
            
#         Returns
#         -------
#         torch.Tensor
#             NSE loss value
#         """
#         # Extract prediction tensor
#         pred = pred_dict['HBV']['flow_sim']
        
#         # Create mask for valid values
#         mask = torch.isfinite(target)
        
#         # Handle broadcasting if shapes differ
#         if mask.shape != pred.shape:
#             mask = mask.expand_as(pred)
        
#         # Apply mask
#         pred_flat = pred.reshape(-1)[mask.reshape(-1)]
#         target_flat = target.reshape(-1)[mask.reshape(-1)]
        
#         if pred_flat.numel() == 0:
#             return torch.tensor(0.0, device=pred.device)
        
#         # Calculate NSE
#         mean = torch.mean(target_flat)
#         numerator = torch.sum((pred_flat - target_flat) ** 2)
#         denominator = torch.sum((target_flat - mean) ** 2)
        
#         # Add small epsilon to avoid division by zero
#         return 1 - numerator / (denominator + 1e-6)

# class RMSELoss(torch.nn.Module):
#     """Root Mean Square Error Loss for HBV output."""
    
#     def forward(self, pred_dict: Dict[str, Any], target: torch.Tensor) -> torch.Tensor:
#         """
#         Calculate RMSE loss between HBV flow simulation and target.
        
#         Parameters
#         ----------
#         pred_dict : Dict[str, Any]
#             Model output dictionary containing {'HBV': {'flow_sim': tensor}}
#         target : torch.Tensor
#             Target values [time, batch, 1]
            
#         Returns
#         -------
#         torch.Tensor
#             Non-negative RMSE loss value
#         """
#         try:
#             # Input validation
#             if not isinstance(pred_dict, dict) or 'HBV' not in pred_dict:
#                 raise ValueError(f"Invalid prediction structure. Got {type(pred_dict)}")
            
#             hbv_output = pred_dict['HBV']
#             if not isinstance(hbv_output, dict) or 'flow_sim' not in hbv_output:
#                 raise ValueError(f"Invalid HBV output structure. Got {type(hbv_output)}")
            
#             pred = hbv_output['flow_sim']
            
#             # Debug info
#             log.debug(f"Loss calculation shapes - pred: {pred.shape}, target: {target.shape}")
#             log.debug(f"Pred stats - min: {pred.min():.3f}, max: {pred.max():.3f}, mean: {pred.mean():.3f}")
#             log.debug(f"Target stats - min: {target.min():.3f}, max: {target.max():.3f}, mean: {target.mean():.3f}")
            
#             # Move to same device if needed
#             if pred.device != target.device:
#                 target = target.to(pred.device)
            
#             # Ensure matching shapes
#             if pred.shape != target.shape:
#                 if len(pred.shape) == 3 and len(target.shape) == 3:
#                     if pred.shape[0] == target.shape[1] and pred.shape[1] == target.shape[0]:
#                         # Transpose target if dimensions are swapped
#                         target = target.transpose(0, 1)
#                     elif pred.shape[-1] != target.shape[-1]:
#                         # Handle singleton dimensions
#                         if pred.shape[-1] == 1:
#                             target = target.squeeze(-1)
#                         elif target.shape[-1] == 1:
#                             pred = pred.squeeze(-1)
                
#                 if pred.shape != target.shape:
#                     raise ValueError(f"Shape mismatch after adjustments. Pred: {pred.shape}, Target: {target.shape}")
            
#             # Create and apply mask for valid values
#             mask = torch.isfinite(target)
#             if mask.shape != pred.shape:
#                 mask = mask.expand_as(pred)
            
#             pred_masked = pred[mask]
#             target_masked = target[mask]
            
#             if pred_masked.numel() == 0:
#                 log.warning("No valid elements for loss calculation")
#                 return torch.tensor(0.0, device=pred.device)
            
#             # Calculate RMSE
#             squared_diff = (pred_masked - target_masked) ** 2
#             squared_diff = torch.clamp(squared_diff, min=0.0)  # Ensure non-negative
#             mse = torch.mean(squared_diff)
#             rmse = torch.sqrt(mse + 1e-8)  # Small epsilon to prevent sqrt(0)
            
#             # Validate output
#             if torch.isnan(rmse) or torch.isinf(rmse):
#                 log.warning(f"Invalid loss value: {rmse}")
#                 return torch.tensor(0.0, device=pred.device)
            
#             return rmse
            
#         except Exception as e:
#             log.error(f"Error in loss calculation: {str(e)}")
#             raise

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()