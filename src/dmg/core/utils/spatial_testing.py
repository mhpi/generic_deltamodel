"""
Spatial testing utilities for running holdout validation experiments.
"""
import logging
import os
import numpy as np
from typing import List, Tuple
from omegaconf import DictConfig
from dmg.core.utils.factory import import_data_loader, import_trainer
from dmg.core.calc.metrics import Metrics
import torch

log = logging.getLogger(__name__)


def run_spatial_testing(config: DictConfig, model) -> None:
    """Execute spatial testing across all holdout indices."""
    holdout_indices = config['test']['holdout_indexs']
    extent = config['test']['extent']
    
    log.info(f"Running spatial testing with {len(holdout_indices)} holdouts")
    
    # Setup aggregated results directory in the main testing folder
    base_output_dir = config.get('out_path', '.')
    agg_results_dir = os.path.join(base_output_dir, f'spatial_aggregated_{extent}')
    os.makedirs(agg_results_dir, exist_ok=True)
    
    all_predictions, all_targets = [], []
    
    for holdout_idx in holdout_indices:
        log.info(f"Processing spatial test with holdout index: {holdout_idx}")
        
        # Create holdout-specific config
        current_config = config.copy() if isinstance(config, dict) else dict(config)
        current_config['test']['current_holdout_index'] = holdout_idx
        
        # Setup directories - just testing folder within each holdout
        holdout_dir = os.path.join(base_output_dir, f'spatial_holdout_{holdout_idx}_{extent}')
        testing_dir = os.path.join(holdout_dir, 'testing')
        
        current_config['testing_path'] = testing_dir
        current_config['out_path'] = testing_dir  # Output directly to testing folder
        
        # Ensure model_path is set for compatibility
        if 'model_path' not in current_config:
            current_config['model_path'] = config.get('model_path', holdout_dir)
        
        os.makedirs(testing_dir, exist_ok=True)
        
        # Reinitialize model for each holdout to prevent data leakage
        if 'train' in config['mode']:
            from dmg.models.model_handler import ModelHandler as dModel
            holdout_model = dModel(current_config, verbose=True)
        else:
            holdout_model = model
        
        # Initialize data loader
        data_loader_cls = import_data_loader(config['data_loader'])
        data_loader = data_loader_cls(
            current_config, 
            test_split=True, 
            overwrite=False,
            holdout_index=holdout_idx
        )
        
        # Initialize trainer
        trainer_cls = import_trainer(config['trainer'])
        trainer = trainer_cls(
            config=current_config,
            model=holdout_model,
            train_dataset=data_loader.train_dataset,
            eval_dataset=data_loader.eval_dataset,
            verbose=True
        )
        
        # Execute mode and collect results
        pred, target = None, None
        mode = config['mode']
        
        if mode == 'train':
            trainer.train()
        elif mode == 'test':
            pred, target = trainer.evaluate()
        elif mode == 'train_test':
            trainer.train()
            pred, target = trainer.evaluate()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Collect results
        if pred is not None and target is not None:
            all_predictions.append(pred)
            all_targets.append(target)
        
        # Save holdout metadata
        _save_holdout_metadata(holdout_dir, holdout_idx, config, extent)
    
    # Aggregate results
    if all_predictions and all_targets:
        log.info("Aggregating results from all holdouts...")
        _aggregate_spatial_results(all_predictions, all_targets, agg_results_dir, config)
    else:
        log.warning("No predictions collected from spatial testing")


def _save_holdout_metadata(holdout_dir: str, holdout_idx: int, config: DictConfig, extent: str) -> None:
    """Save metadata about the holdout experiment."""
    metadata_path = os.path.join(holdout_dir, 'holdout_info.txt')
    
    with open(metadata_path, 'w') as f:
        f.write(f"Holdout Index: {holdout_idx}\n")
        
        if extent == 'PUR':
            region_info = config['test']['huc_regions'][holdout_idx]
            f.write(f"HUC Regions: {region_info}\n")
        elif extent == 'PUB':
            pub_id = config['test']['PUB_ids'][holdout_idx]
            f.write(f"PUB ID: {pub_id}\n")
        
        if 'holdout_basins' in config['test']:
            f.write(f"Basins: {config['test']['holdout_basins']}\n")


def _aggregate_spatial_results(predictions: List, targets: List, 
                              output_dir: str, config: DictConfig) -> None:
    """Aggregate results from multiple spatial holdouts."""
    log.info(f"Aggregating results from {len(predictions)} holdouts")
    
    try:
        # Handle different types of predictions
        processed_predictions = []
        processed_targets = []
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            log.info(f"Processing holdout {i}")
            
            # Handle predictions - could be list of dicts or numpy array
            if isinstance(pred, list):
                # It's a list of batch predictions (dictionaries)
                log.info(f"Holdout {i}: Converting list of {len(pred)} batch predictions to numpy array")
                
                # Extract prediction key - assume first available key
                if pred and isinstance(pred[0], dict):
                    available_keys = list(pred[0].keys())
                    target_key = available_keys[0] if available_keys else 'prediction'
                    log.info(f"Using prediction key: '{target_key}'")
                    
                    # Concatenate batch predictions
                    try:
                        # Try concatenating along timestep dimension (dim=0)
                        batch_tensors = [torch.tensor(batch[target_key]) if not isinstance(batch[target_key], torch.Tensor) 
                                       else batch[target_key] for batch in pred]
                        pred_array = torch.cat(batch_tensors, dim=0).numpy()
                    except Exception as e:
                        log.warning(f"Failed to concatenate predictions for holdout {i}: {e}")
                        # Fallback: try different concatenation strategy
                        try:
                            pred_array = np.concatenate([batch[target_key].numpy() if isinstance(batch[target_key], torch.Tensor) 
                                                       else batch[target_key] for batch in pred], axis=0)
                        except Exception as e2:
                            log.error(f"Could not process predictions for holdout {i}: {e2}")
                            continue
                else:
                    log.warning(f"Unexpected prediction format for holdout {i}: {type(pred)}")
                    continue
                    
            elif isinstance(pred, np.ndarray):
                # Already a numpy array
                pred_array = pred
            elif isinstance(pred, torch.Tensor):
                # Convert tensor to numpy
                pred_array = pred.numpy()
            else:
                log.warning(f"Unknown prediction type for holdout {i}: {type(pred)}")
                continue
            
            # Handle targets
            if isinstance(target, torch.Tensor):
                target_array = target.cpu().numpy()
            elif isinstance(target, np.ndarray):
                target_array = target
            else:
                log.warning(f"Unknown target type for holdout {i}: {type(target)}")
                continue
            
            # Ensure targets have the right shape
            if len(target_array.shape) == 3 and target_array.shape[2] == 1:
                target_array = target_array[:, :, 0]  # Remove last dimension if it's 1
            
            log.info(f"Holdout {i} - Prediction shape: {pred_array.shape}, Target shape: {target_array.shape}")
            
            processed_predictions.append(pred_array)
            processed_targets.append(target_array)
        
        if not processed_predictions:
            log.error("No valid predictions to aggregate")
            return
        
        # Log shapes for debugging
        pred_shapes = [p.shape for p in processed_predictions]
        target_shapes = [t.shape for t in processed_targets]
        log.info(f"Processed prediction shapes: {pred_shapes}")
        log.info(f"Processed target shapes: {target_shapes}")
        
        # Ensure consistent shapes across holdouts
        # Find minimum dimensions
        min_time_steps = min(p.shape[0] for p in processed_predictions)
        
        # Check if all have same number of time steps
        if not all(p.shape[0] == processed_predictions[0].shape[0] for p in processed_predictions):
            log.warning(f"Inconsistent time steps across holdouts. Trimming to minimum: {min_time_steps}")
            processed_predictions = [p[:min_time_steps] for p in processed_predictions]
            processed_targets = [t[:min_time_steps] for t in processed_targets]
        
        # Concatenate along basin dimension (axis 1 for 2D arrays)
        if len(processed_predictions[0].shape) == 2:
            # 2D arrays: [time, basins]
            all_predictions = np.concatenate(processed_predictions, axis=1)
            all_targets = np.concatenate(processed_targets, axis=1)
        elif len(processed_predictions[0].shape) == 3:
            # 3D arrays: [time, basins, features]
            all_predictions = np.concatenate(processed_predictions, axis=1)
            all_targets = np.concatenate(processed_targets, axis=1)
        else:
            log.error(f"Unexpected prediction array shape: {processed_predictions[0].shape}")
            return
        
        log.info(f"Aggregated predictions shape: {all_predictions.shape}")
        log.info(f"Aggregated targets shape: {all_targets.shape}")
        
        # Save aggregated arrays
        np.save(os.path.join(output_dir, 'aggregated_predictions.npy'), all_predictions)
        np.save(os.path.join(output_dir, 'aggregated_targets.npy'), all_targets)
        
        # Calculate and save metrics
        pred_formatted = np.swapaxes(all_predictions.squeeze(), 0, 1)  # [basins, time]
        target_formatted = np.swapaxes(all_targets.squeeze(), 0, 1)    # [basins, time]
        
        log.info(f"Formatted for metrics - predictions: {pred_formatted.shape}, targets: {target_formatted.shape}")
        
        log.info("Calculating metrics for aggregated results")
        metrics = Metrics(pred_formatted, target_formatted)
        metrics.dump_metrics(output_dir)
        
        log.info("Spatial aggregation completed successfully")
        
    except Exception as e:
        log.error(f"Error in spatial result aggregation: {str(e)}")
        import traceback
        log.error(traceback.format_exc())