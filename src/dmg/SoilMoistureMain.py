import logging
import matplotlib
import os
import sys
import time
log = logging.getLogger(__name__)
import hydra
import torch
from omegaconf import DictConfig
import sys
import os
from core.calc.metrics_soilMoisture import Metrics
import numpy as np

# Import core components
from core.utils import (
    initialize_config,
    print_config,
    set_randomseed
)
# import matplotlib.pyplot as plt
from core.utils.module_loaders import get_data_loader, get_trainer
from models.model_handler2 import ModelHandler as dModel
from trainers.finetuning_noHBV import FineTuneTrainer
from core.data.data_loaders.finetune_loader import FineTuneDataLoader

@hydra.main(version_base='1.3', config_path='conf/', config_name='soilMoisturePUB')
def main(config: DictConfig) -> None:
    """Main entry point for model training/testing.
    
    Parameters
    ----------
    config : DictConfig
        Hydra configuration object
    """
    try:
        start_time = time.perf_counter()
        # Basic setup
        config = initialize_config(config)
        set_randomseed(config['random_seed'])
        
        test_mode = config['test_mode']['type']
        log.info(f"RUNNING MODE: {config['mode']}, TESTING TYPE: {test_mode}")

        print_config(config)

        # Load model
        log.info("Initializing model...") 
        model = dModel(config, verbose=True)
        
        # Get the data loader class
        DataLoaderClass = get_data_loader(config['data_loader'])
        
        if test_mode == 'temporal':
            # Load dataset for temporal testing
            log.info("Loading dataset for temporal testing...")
            data_loader = DataLoaderClass(config, test_split=True, overwrite=False)

            # Initialize trainer
            log.info("Initializing trainer...")
            trainer = FineTuneTrainer(
                config=config,
                model=model,
                train_dataset=data_loader.train_dataset,
                eval_dataset=data_loader.eval_dataset,
                verbose=True
            )

            mode = config['mode']
            if mode == 'train':
                trainer.train()
            elif mode == 'test':
                trainer.test()
            elif mode == 'train_test':
                trainer.train()
                trainer.test()
            else:
                raise ValueError(f"Invalid mode: {mode}")
                
        elif test_mode == 'spatial':
            # For spatial testing, loop through holdout indices
            holdout_indices = config['test_mode']['holdout_indexs']
            all_predictions, all_targets = [], []
            extent = config['test_mode']['extent']
            # Create directory for aggregated results
            agg_results_dir = os.path.join(config['validation_path'], f'spatial_aggregated_{extent}')
            # print(agg_results_dir)
            os.makedirs(agg_results_dir, exist_ok=True)
            
            for holdout_idx in holdout_indices:
                log.info(f"Processing spatial test with holdout index: {holdout_idx}")
                
                # Create a copy of config to modify for this specific holdout
                current_config = config.copy() if isinstance(config, dict) else dict(config)
                
                # Update the config with the current holdout index
                current_config['test_mode']['current_holdout_index'] = holdout_idx
                # Create a specific output directory for this holdout index
                holdout_dir = os.path.join(config['validation_path'], f'spatial_holdout_{holdout_idx}_{extent}')
                # print(holdout_dir)
                current_config['validation_path'] = os.path.join(holdout_dir, 'validation')
                current_config['testing_path'] = os.path.join(holdout_dir, 'testing')
                current_config['out_path'] = holdout_dir
                
                os.makedirs(current_config['validation_path'], exist_ok=True)
                # os.makedirs(current_config['testing_path'], exist_ok=True)
                
                # Initialize data loader with the current holdout index
                data_loader = DataLoaderClass(
                    current_config, 
                    test_split=True, 
                    overwrite=False,
                    holdout_index=holdout_idx
                )
                
                # Initialize trainer for this holdout
                log.info(f"Initializing trainer for spatial test index {holdout_idx}...")
                trainer = FineTuneTrainer(
                    config=current_config,
                    model=model,
                    train_dataset=data_loader.train_dataset,
                    eval_dataset=data_loader.eval_dataset,
                    verbose=True
                )
                
                # Execute requested mode
                mode = config['mode']
                pred, target = None, None
                
                if mode == 'train':
                    trainer.train()
                elif mode == 'test':
                    pred, target = trainer.test()
                elif mode == 'train_test':
                    trainer.train()
                    pred, target = trainer.test()
                else:
                    raise ValueError(f"Invalid mode: {mode}")
                
                # Collect predictions and targets for aggregation
                if pred is not None and target is not None:
                    all_predictions.append(pred)
                    all_targets.append(target)
                    
                    # Save a summary file with metadata about this holdout
                    with open(os.path.join(holdout_dir, 'holdout_info.txt'), 'w') as f:
                        extent = config['test_mode']['extent']
                        if extent == 'PUR':
                            region_info = config['test_mode']['huc_regions'][holdout_idx]
                            f.write(f"Holdout Index: {holdout_idx}\n")
                            f.write(f"HUC Regions: {region_info}\n")
                            if 'holdout_basins' in config['test_mode']:
                                f.write(f"Basins: {config['test_mode']['holdout_basins']}\n")
                        elif extent == 'PUB':
                            pub_id = config['test_mode']['PUB_ids'][holdout_idx]
                            f.write(f"Holdout Index: {holdout_idx}\n")
                            f.write(f"PUB ID: {pub_id}\n")
                            if 'holdout_basins' in config['test_mode']:
                                f.write(f"Basins: {config['test_mode']['holdout_basins']}\n")

            # Aggregate results from multiple holdouts
            if all_predictions and all_targets:
                log.info("Aggregating results from all holdouts...")
                aggregate_spatial_results(all_predictions, all_targets, agg_results_dir, config)
            else:
                raise ValueError(f"Invalid testing type: {test_mode}")

    except KeyboardInterrupt:
        log.info("Process interrupted by user")
    except Exception as e:
        log.error(f"Error occurred: {str(e)}")
        raise e
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed | "
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        )


def aggregate_spatial_results(predictions, targets, output_dir, config):
    """Aggregate results from multiple spatial holdouts.
    
    Parameters
    ----------
    predictions : list
        List of prediction arrays from each holdout
    targets : list
        List of target arrays from each holdout
    output_dir : str
        Directory to save aggregated results
    config : dict
        Configuration dictionary
    """
    log.info(f"Aggregating results from {len(predictions)} holdouts")
    
    try:
        # First, determine the proper shapes to concatenate
        # Predictions and targets from different holdouts may have varying basin counts
        # but should have the same time dimension
        
        # List to track shapes for debugging
        pred_shapes = [p.shape for p in predictions]
        target_shapes = [t.shape for t in targets]
        log.info(f"Prediction shapes: {pred_shapes}")
        log.info(f"Target shapes: {target_shapes}")
        
        # Concatenate along the basin dimension (axis 1)
        # First need to ensure all arrays have the same time dimension
        min_time_steps = min(p.shape[0] for p in predictions)
        
        # Trim to the same time length if needed
        trimmed_predictions = [p[:min_time_steps] for p in predictions]
        trimmed_targets = [t[:min_time_steps] for t in targets]
        
        # Concatenate along basin dimension
        all_predictions = np.concatenate(trimmed_predictions, axis=1)
        all_targets = np.concatenate(trimmed_targets, axis=1)
        
        log.info(f"Aggregated predictions shape: {all_predictions.shape}")
        log.info(f"Aggregated targets shape: {all_targets.shape}")
        
        # Save concatenated arrays
        np.save(os.path.join(output_dir, 'aggregated_predictions.npy'), all_predictions)
        np.save(os.path.join(output_dir, 'aggregated_targets.npy'), all_targets)
        
        # Calculate metrics on the aggregated results
        # Need to format to [basins, time] for the Metrics class
        pred_formatted = np.swapaxes(all_predictions.squeeze(), 0, 1)
        target_formatted = np.swapaxes(all_targets.squeeze(), 0, 1)
        
        # Calculate metrics
        log.info("Calculating metrics for aggregated results")
        metrics = Metrics(pred_formatted, target_formatted)
        
        # Save metrics
        metrics.dump_metrics(output_dir)
        
        # Generate a detailed report
        create_aggregated_report(output_dir, config, metrics)
        
    except Exception as e:
        log.error(f"Error in spatial result aggregation: {str(e)}")
        import traceback
        log.error(traceback.format_exc())

def create_aggregated_report(output_dir, config, metrics):
    """Create a detailed report of the aggregated results.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the report
    config : dict
        Configuration dictionary
    metrics : Metrics
        Metrics object with calculated metrics
    """
    report_path = os.path.join(output_dir, 'aggregated_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=== Aggregated Spatial Testing Report ===\n\n")
        
        # Configuration details
        f.write("Configuration:\n")
        f.write(f"- Testing Type: {config['test_mode']['type']}\n")
        f.write(f"- Spatial Extent: {config['test_mode']['extent']}\n")
        f.write(f"- Holdout Indices: {config['test_mode']['holdout_indexs']}\n")
        f.write(f"- Test Period: {config['test']['start_time']} to {config['test']['end_time']}\n\n")
        
        # Model details
        f.write("Model:\n")
        f.write(f"- Phy Model: {config['dpl_model']['phy_model']['model']}\n")
        f.write(f"- NN Model: {config['dpl_model']['nn_model']['model']}\n")
        f.write(f"- Warm-up: {config['dpl_model']['phy_model']['warm_up']}\n\n")
        
        # Key metrics
        stats = metrics.calc_stats()
        f.write("Key Performance Metrics (Median, Mean, Std):\n")
        
        # NSE (Nash-Sutcliffe Efficiency)
        if 'nse' in stats:
            f.write(f"- NSE: {stats['nse']['median']:.4f}, {stats['nse']['mean']:.4f}, {stats['nse']['std']:.4f}\n")
        
        # KGE (Kling-Gupta Efficiency)
        if 'kge' in stats:
            f.write(f"- KGE: {stats['kge']['median']:.4f}, {stats['kge']['mean']:.4f}, {stats['kge']['std']:.4f}\n")
        
        # Correlation
        if 'corr' in stats:
            f.write(f"- Correlation: {stats['corr']['median']:.4f}, {stats['corr']['mean']:.4f}, {stats['corr']['std']:.4f}\n")
        
        # RMSE
        if 'rmse' in stats:
            f.write(f"- RMSE: {stats['rmse']['median']:.4f}, {stats['rmse']['mean']:.4f}, {stats['rmse']['std']:.4f}\n")
        
        # Bias
        if 'bias' in stats:
            f.write(f"- Bias: {stats['bias']['median']:.4f}, {stats['bias']['mean']:.4f}, {stats['bias']['std']:.4f}\n")
        
        # Flow metrics
        if 'flv' in stats and 'fhv' in stats:
            f.write(f"- Low Flow Bias (FLV): {stats['flv']['median']:.4f}, {stats['flv']['mean']:.4f}, {stats['flv']['std']:.4f}\n")
            f.write(f"- High Flow Bias (FHV): {stats['fhv']['median']:.4f}, {stats['fhv']['mean']:.4f}, {stats['fhv']['std']:.4f}\n")
        
    log.info(f"Aggregated report created at {report_path}")


if __name__ == "__main__":
    main()