import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy.typing import NDArray

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
                # log.info(f"Dataset shapes:")
                # for key, value in self.train_dataset.items():
                #     if hasattr(value, 'shape'):
                #         log.info(f"  {key}: {value.shape}")
                
                # Get grid dimensions for training - but fix the dimension issue
                n_samples, n_minibatch, n_timesteps = create_training_grid(
                    self.train_dataset['xc_nn_norm'],
                    self.config
                )
                                
                
                log.info(f"training grid - basins: {n_samples}, timesteps: {n_timesteps}, mini_batches: {n_minibatch}")

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

                        # print(model_output.shape)
                        # print(target.shape)
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
        """Run model evaluation and return both metrics and model outputs."""
        self.is_in_train = False

        # Track overall predictions and observations
        batch_predictions = []
        observations = self.test_dataset['target']

        # CORRECTED: Use shape[1] for TIME dimension like the working version
        n_samples = self.test_dataset['xc_nn_norm'].shape[1]  # 731 timesteps, not basins
        batch_start = np.arange(0, n_samples, self.config['test']['batch_size'])
        batch_end = np.append(batch_start[1:], n_samples)

        # Model forward
        log.info(f"Validating Model: Forwarding {len(batch_start)} batches")
        batch_predictions = self._forward_loop(self.test_dataset, batch_start, batch_end)

        # Save predictions and calculate metrics
        log.info("Saving model outputs + Calculating metrics")
        save_outputs(self.config, batch_predictions, observations)
        self.predictions = self._batch_data(batch_predictions)

        # Calculate metrics
        self.calc_metrics(batch_predictions, observations)

        return batch_predictions, observations


    def _process_predictions(self, batch_predictions: list) -> np.ndarray:
        """Convert list of batch predictions to a single numpy array."""
        if not batch_predictions:
            raise ValueError("No batch predictions to process")
        
        # Extract arrays from each batch
        arrays = []
        for i, batch in enumerate(batch_predictions):
            if isinstance(batch, dict):
                # Get the prediction tensor from the dictionary
                if 'prediction' in batch:
                    array = batch['prediction']
                else:
                    # Take the first value if 'prediction' key doesn't exist
                    array = next(iter(batch.values()))
                
                # Convert to numpy if needed
                if isinstance(array, torch.Tensor):
                    array = array.cpu().detach().numpy()
                
                arrays.append(array)
                log.debug(f"Batch {i} array shape: {array.shape}")
            else:
                log.warning(f"Unexpected batch type: {type(batch)}")
                continue
        
        if not arrays:
            raise ValueError("No valid arrays found in batch predictions")
        
        # Concatenate arrays
        # Assuming arrays are shaped like [time, batch_basins, features]
        # We want to concatenate along the basin dimension (axis=1)
        try:
            if len(arrays[0].shape) == 3:
                # [time, batch_basins, features] -> concatenate along axis=1
                combined = np.concatenate(arrays, axis=1)
            elif len(arrays[0].shape) == 2:
                # [time, batch_basins] -> concatenate along axis=1
                combined = np.concatenate(arrays, axis=1)
            else:
                # Fallback: concatenate along axis=0
                combined = np.concatenate(arrays, axis=0)
            
            log.info(f"Combined array shape: {combined.shape}")
            return combined
            
        except Exception as e:
            log.error(f"Error concatenating arrays: {e}")
            # Debug: print shapes of first few arrays
            for i, arr in enumerate(arrays[:3]):
                log.error(f"Array {i} shape: {arr.shape}")
            raise    
    
    def _forward_loop(
        self,
        data: dict[str, torch.Tensor],
        batch_start: NDArray,
        batch_end: NDArray
    ) -> list:
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
        # Track predictions across batches
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

            # Save the batch predictions - CORRECTED to match working version
            model_name = self.config['delta_model']['phy_model']['model'][0]
            
            # Check the structure and handle appropriately
            model_output = prediction[model_name]
            
            if isinstance(model_output, dict):
                # Working version expects this to be a dict of tensors
                prediction_dict = {
                    key: tensor.cpu().detach() for key, tensor in model_output.items()
                }
            elif isinstance(model_output, torch.Tensor):
                # If it's a tensor, check if it contains a dict (single element)
                if model_output.numel() == 1:
                    try:
                        extracted = model_output.item()
                        if isinstance(extracted, dict):
                            prediction_dict = {
                                key: tensor.cpu().detach() if isinstance(tensor, torch.Tensor) else tensor
                                for key, tensor in extracted.items()
                            }
                        else:
                            # Fallback: create dict with single prediction
                            prediction_dict = {'prediction': model_output.cpu().detach()}
                    except:
                        prediction_dict = {'prediction': model_output.cpu().detach()}
                else:
                    # Multi-element tensor: treat as prediction
                    prediction_dict = {'prediction': model_output.cpu().detach()}
            else:
                log.warning(f"Unexpected model output type: {type(model_output)}")
                prediction_dict = {'prediction': model_output}
                
            batch_predictions.append(prediction_dict)
            
        return batch_predictions


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

    # def calc_metrics(self, predictions: np.ndarray, observations: np.ndarray) -> None:
    #     """Calculate comprehensive metrics using the appropriate Metrics class."""
    #     try:
    #         # Ensure no NaN values in predictions
    #         predictions = np.nan_to_num(predictions, nan=0.0)
            
    #         # Format predictions and observations for the Metrics class
    #         # Metrics expects shape [basins, time]
    #         pred_formatted = np.swapaxes(predictions.squeeze(), 0, 1)
    #         obs_formatted = np.swapaxes(observations.squeeze(), 0, 1)
            
    #         log.info(f"Calculating comprehensive metrics")
    #         log.info(f"Formatted shapes - predictions: {pred_formatted.shape}, observations: {obs_formatted.shape}")
            
    #         # Use appropriate metrics class
    #         try:
    #             # Try soil moisture metrics first if it might be soil moisture data
    #             if self.model_type == 'direct':
    #                 from dMG.core.calc.metrics_soilMoisture import Metrics
    #                 log.info("Using soil moisture specific metrics")
    #             else:
    #                 metrics = Metrics(pred_formatted, obs_formatted)
    #         except ImportError:
    #             # Fall back to standard metrics
    #             log.info("Using standard metrics")
    #             metrics = Metrics(pred_formatted, obs_formatted)
            
    #         # Create Metrics object and save
    #         metrics = Metrics(pred_formatted, obs_formatted)
    #         metrics.dump_metrics(self.config['out_path'])
            
    #     except Exception as e:
    #         log.error(f"Error calculating metrics: {str(e)}")
    #         import traceback
    #         log.error(traceback.format_exc())
    
    def inference(self) -> None:
        """Run batch model inference - required by BaseTrainer."""
        raise NotImplementedError



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

            
    def _batch_data(
        self,
        batch_list: list[dict[str, torch.Tensor]],
        target_key: str = None,
    ) -> dict:
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
                    dim = 1  # Concatenate along time dimension for 3D tensors
                else:
                    dim = 0  # Concatenate along first dimension for 2D tensors
                data[key] = torch.cat([d[key] for d in batch_list], dim=dim).cpu().numpy()
            return data

        except ValueError as e:
            raise ValueError(f"Error concatenating batch data: {e}") from e

    def calc_metrics(
        self,
        batch_predictions: list[dict[str, torch.Tensor]], 
        observations: torch.Tensor,
    ) -> None:
        """Calculate and save model performance metrics - FIXED VERSION.

        Parameters
        ----------
        batch_predictions
            List of dictionaries containing model predictions.
        observations
            Target variable observation data.
        """
        try:
            # Check what keys are available
            available_keys = list(batch_predictions[0].keys()) if batch_predictions else []
            log.info(f"Available prediction keys: {available_keys}")
            
            # Get target name from config, but fall back to first available key
            target_name = self.config['train']['target'][0]
            
            if target_name in available_keys:
                log.info(f"Using target key: '{target_name}'")
                predictions = self._batch_data(batch_predictions, target_name)
            else:
                # Use first available key as fallback
                fallback_key = available_keys[0] if available_keys else None
                if fallback_key:
                    log.warning(f"Target key '{target_name}' not found, using '{fallback_key}' instead")
                    predictions = self._batch_data(batch_predictions, fallback_key)
                else:
                    log.error("No keys available in batch predictions")
                    return
            
            # Process target like the working version
            if isinstance(observations, torch.Tensor):
                target = np.expand_dims(observations[:, :, 0].cpu().numpy(), 2)
            else:
                target = np.expand_dims(observations[:, :, 0], 2)

            log.info(f"Before warmup alignment - predictions: {predictions.shape}, target: {target.shape}")

            # FIXED: Handle warmup period mismatch properly
            warm_up = self.config['delta_model']['phy_model'].get('warm_up', 0)
            
            # Check if there's a time dimension mismatch that suggests warmup handling issue
            time_diff = target.shape[0] - predictions.shape[0]
            
            if time_diff == warm_up:
                # Target includes warmup, predictions don't - remove warmup from target
                log.info(f"Removing {warm_up} warmup steps from target only")
                target = target[warm_up:, :]
            elif time_diff == -warm_up:
                # Predictions include warmup, target doesn't - remove warmup from predictions  
                log.info(f"Removing {warm_up} warmup steps from predictions only")
                predictions = predictions[warm_up:, :]
            elif time_diff == 0:
                # Same size - remove warmup from both if specified
                if warm_up > 0:
                    log.info(f"Removing {warm_up} warmup steps from both")
                    target = target[warm_up:, :]
                    predictions = predictions[warm_up:, :]
            else:
                # Other mismatch - try to align by taking the minimum
                min_time = min(target.shape[0], predictions.shape[0])
                log.warning(f"Time mismatch not matching warmup. Aligning to minimum: {min_time}")
                target = target[:min_time, :]
                predictions = predictions[:min_time, :]

            log.info(f"Final shapes for metrics - predictions: {predictions.shape}, target: {target.shape}")
            
            # Additional shape validation
            if predictions.shape[0] != target.shape[0]:
                log.error(f"Time dimension still mismatched: predictions {predictions.shape[0]} vs target {target.shape[0]}")
                return
                
            if predictions.shape[1] != target.shape[1]:
                log.error(f"Basin dimension mismatch: predictions {predictions.shape[1]} vs target {target.shape[1]}")
                return

            # Squeeze and check dimensions before swapaxes
            pred_squeezed = predictions.squeeze()
            target_squeezed = target.squeeze()
            
            log.info(f"After squeeze - predictions: {pred_squeezed.shape}, target: {target_squeezed.shape}")
            
            # Only proceed if we have valid 2D arrays
            if len(pred_squeezed.shape) < 2 or len(target_squeezed.shape) < 2:
                log.error(f"Invalid dimensions after squeeze: pred {pred_squeezed.shape}, target {target_squeezed.shape}")
                return

            # Compute metrics using the exact same format as working version
            metrics = Metrics(
                np.swapaxes(pred_squeezed, 1, 0),  # [basins, time]
                np.swapaxes(target_squeezed, 1, 0),       # [basins, time]
            )

            # Save all metrics and aggregated statistics
            metrics.dump_metrics(self.config['out_path'])
            log.info("Metrics calculation completed successfully")
            
        except Exception as e:
            log.error(f"Error calculating metrics: {str(e)}")
            import traceback
            log.error(traceback.format_exc())

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()