"""
Spatial testing utilities for running holdout validation experiments.
"""
import logging
import os
import numpy as np
from typing import List, Tuple
from omegaconf import DictConfig
from dMG.core.utils.factory import import_data_loader, import_trainer
from dMG.core.calc.metrics import Metrics

log = logging.getLogger(__name__)


def run_spatial_testing(config: DictConfig, model) -> None:
    """Execute spatial testing across all holdout indices."""
    holdout_indices = config['test_mode']['holdout_indexs']
    extent = config['test_mode']['extent']
    
    log.info(f"Running spatial testing with {len(holdout_indices)} holdouts")
    
    # Setup aggregated results directory
    agg_results_dir = os.path.join(config['validation_path'], f'spatial_aggregated_{extent}')
    os.makedirs(agg_results_dir, exist_ok=True)
    
    all_predictions, all_targets = [], []
    
    for holdout_idx in holdout_indices:
        log.info(f"Processing spatial test with holdout index: {holdout_idx}")
        
        # Create holdout-specific config
        current_config = config.copy() if isinstance(config, dict) else dict(config)
        current_config['test_mode']['current_holdout_index'] = holdout_idx
        
        # Setup directories
        holdout_dir = os.path.join(config['validation_path'], f'spatial_holdout_{holdout_idx}_{extent}')
        current_config['validation_path'] = os.path.join(holdout_dir, 'validation')
        current_config['testing_path'] = os.path.join(holdout_dir, 'testing')
        current_config['out_path'] = holdout_dir
        
        os.makedirs(current_config['validation_path'], exist_ok=True)
        
        # Reinitialize model for each holdout to prevent data leakage
        if 'train' in config['mode']:
            from dMG.models.model_handler import ModelHandler as dModel
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
            pred, target = trainer.test()
        elif mode == 'train_test':
            trainer.train()
            pred, target = trainer.test()
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
            region_info = config['test_mode']['huc_regions'][holdout_idx]
            f.write(f"HUC Regions: {region_info}\n")
        elif extent == 'PUB':
            pub_id = config['test_mode']['PUB_ids'][holdout_idx]
            f.write(f"PUB ID: {pub_id}\n")
        
        if 'holdout_basins' in config['test_mode']:
            f.write(f"Basins: {config['test_mode']['holdout_basins']}\n")


def _aggregate_spatial_results(predictions: List[np.ndarray], targets: List[np.ndarray], 
                              output_dir: str, config: DictConfig) -> None:
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
        log.error(f"Error in spatial result aggregation: {str(e)}")
        import traceback
        log.error(traceback.format_exc())

