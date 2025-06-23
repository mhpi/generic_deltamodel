"""Spatial testing utilities for running holdout validation experiments."""
import logging
import os
import traceback

import numpy as np
import torch
from omegaconf import DictConfig

from dmg.core.calc.metrics import Metrics
from dmg.core.utils.factory import import_data_loader, import_trainer

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
            holdout_index=holdout_idx,
        )
        
        # Initialize trainer
        trainer_cls = import_trainer(config['trainer'])
        trainer = trainer_cls(
            config=current_config,
            model=holdout_model,
            train_dataset=data_loader.train_dataset,
            eval_dataset=data_loader.eval_dataset,
            verbose=True,
        )
        
        # Execute mode and collect results
        pred, target = None, None
        mode = config['mode']
        
        if mode == 'train':
            trainer.train()
        elif mode == 'test':
            trainer.evaluate()
            pred, target = _extract_predictions_and_targets(trainer, config)
        elif mode == 'train_test':
            trainer.train()
            trainer.evaluate()
            pred, target = _extract_predictions_and_targets(trainer, config)
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


def _aggregate_spatial_results(
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    output_dir: str,
    config: DictConfig,
) -> None:
    """Aggregate results from multiple spatial holdouts."""
    log.info(f"Aggregating results from {len(predictions)} holdouts")
    
    try:
        # Log shapes for debugging
        pred_shapes = [p.shape for p in predictions]
        target_shapes = [t.shape for t in targets]
        log.info(f"Prediction shapes: {pred_shapes}")
        log.info(f"Target shapes: {target_shapes}")
        
        # Ensure consistent time dimensions
        min_time_steps = min(p.shape[0] for p in predictions)
        trimmed_predictions = [p[:min_time_steps] for p in predictions]
        trimmed_targets = [t[:min_time_steps] for t in targets]
        
        # Concatenate along basin dimension (axis 1)
        all_predictions = np.concatenate(trimmed_predictions, axis=1)
        all_targets = np.concatenate(trimmed_targets, axis=1)
        
        log.info(f"Aggregated predictions shape: {all_predictions.shape}")
        log.info(f"Aggregated targets shape: {all_targets.shape}")
        
        # Save aggregated arrays
        np.save(os.path.join(output_dir, 'aggregated_predictions.npy'), all_predictions)
        np.save(os.path.join(output_dir, 'aggregated_targets.npy'), all_targets)
        
        # Calculate and save metrics
        pred_formatted = np.swapaxes(all_predictions.squeeze(), 0, 1)
        target_formatted = np.swapaxes(all_targets.squeeze(), 0, 1)
        
        log.info("Calculating metrics for aggregated results")
        metrics = Metrics(pred_formatted, target_formatted)
        metrics.dump_metrics(output_dir)
        
    except Exception as e:
        log.error(traceback.format_exc())
        raise RuntimeError(f"Error in spatial result aggregation: {str(e)}") from e


def _extract_predictions_and_targets(
    trainer: torch.nn.Module,
    config: DictConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract predictions and targets from trainer after evaluation.
    
    Parameters
    ----------
    trainer
        The trainer object after evaluation has been completed
    config
        Configuration dictionary
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (predictions, targets) as numpy arrays
    """
    try:
        # Get the target variable name
        target_name = config['train']['target'][0]
        
        # Extract predictions - trainer.predictions is a dict from _batch_data
        if hasattr(trainer, 'predictions') and trainer.predictions is not None:
            if isinstance(trainer.predictions, dict) and target_name in trainer.predictions:
                pred = trainer.predictions[target_name]
            elif isinstance(trainer.predictions, np.ndarray):
                pred = trainer.predictions
            else:
                log.error(f"Unexpected predictions format: {type(trainer.predictions)}")
                return None, None
        else:
            log.error("No predictions found in trainer")
            return None, None
        
        # Extract targets from eval dataset
        target = trainer.eval_dataset['target']
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # Apply warmup removal if needed (same as in trainer.calc_metrics)
        warmup = config['delta_model']['phy_model']['warm_up']
        if warmup > 0:
            target = target[warmup:, :, :]
        
        # Ensure we have the right target variable (first one if multiple)
        if len(target.shape) == 3 and target.shape[2] > 0:
            target = np.expand_dims(target[:, :, 0], 2)
        
        log.info(f"Extracted predictions shape: {pred.shape}")
        log.info(f"Extracted targets shape: {target.shape}")
        
        return pred, target
        
    except Exception as e:
        log.error(traceback.format_exc())
        raise RuntimeError(f"Error extracting predictions and targets: {e}") from e
