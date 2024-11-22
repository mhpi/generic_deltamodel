""" Main script for running differentiable model experiments."""
import logging
import time
import os
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig
import hydra
from core.calc.stat import metrics
from core.data.dataset_loading import get_dataset_dict
from core.utils import initialize_config, print_config, set_randomseed
from models.model_handler_dev import ModelHandler as dModel

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="conf/",
    config_name="config_dev",
)
def main(config: DictConfig) -> None:
    try:
        start_time = time.perf_counter()

        ### Initializations ###
        config = initialize_config(config)
        set_randomseed(config['random_seed'])

        log.info(f"RUNNING MODE: {config['mode']}")
        print_config(config)

        ### Load differentiable models ###
        model_paths = [
            '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/HBV_1_1p_/NseLossBatch_/dynamic_para/parBETA_parBETAET_parK0_/HBV_1_1p_model_Ep50.pt',
            '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/PRMS_/NseLossBatch_/dynamic_para/alpha_scx_cgw_resmax_k1_k2_/PRMS_model_Ep50.pt'
            '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/SACSMA_with_snow_/NseLossBatch_/dynamic_para/pctim_smax_f1_f2_kuz_rexp_f3_f4_pfree_klzp_klzs_parCWH_/SACSMA_with_snow_model_Ep50.pt'
        ]
        model_list= []

        for i in range(len(model_paths)):
            model_list = dModel(config, verbose=True)
        models = [dModel.load_from_checkpoint(path).to(config['device']) for path in model_paths]

        ### Process datasets ###
        log.info("Processing datasets...")
        train_dataset = get_dataset_dict(config, train=True)
        test_dataset = get_dataset_dict(config, train=False)

        ### Forward Models on Train Dataset ###
        log.info("Running models on train dataset...")
        train_preds = []
        for model in models:
            preds = model(train_dataset['input'])
            train_preds.append(preds.detach())  # Save model predictions

        # Stack and compute the best-performing model per basin
        train_preds = torch.stack(train_preds, dim=0)  # Shape: [num_models, n_train_samples, n_basins]
        train_obs = train_dataset['target'][:, :, 0]  # Observed flow, Shape: [n_train_samples, n_basins]

        # Compute NSE for each model on each basin
        train_nse = torch.zeros(train_preds.shape[0], train_preds.shape[2])  # Shape: [num_models, n_basins]
        for i, preds in enumerate(train_preds):
            train_nse[i, :] = 1 - torch.sum((preds - train_obs) ** 2, dim=0) / torch.sum(
                (train_obs - train_obs.mean(dim=0)) ** 2, dim=0
            )

        # Compute softmax weights across models for each basin
        softmax_weights = torch.softmax(train_nse, dim=0)  # Shape: [num_models, n_basins]

        ### Forward Models on Test Dataset ###
        log.info("Running models on test dataset...")
        test_preds = []
        for model in models:
            preds = model(test_dataset['input'])
            test_preds.append(preds.detach())

        # Stack test predictions
        test_preds = torch.stack(test_preds, dim=0)  # Shape: [num_models, n_test_samples, n_basins]

        # Apply softmax weights to select best-performing model per basin
        selected_preds = (softmax_weights.unsqueeze(1) * test_preds).sum(dim=0)  # Shape: [n_test_samples, n_basins]

        ### Calculate Metrics ###
        log.info("Calculating metrics...")
        test_obs = test_dataset['target'][:, :, 0]  # Observed flow, Shape: [n_test_samples, n_basins]
        metrics_dict = metrics(
            np.swapaxes(selected_preds.numpy(), 1, 0),
            np.swapaxes(test_obs.numpy(), 1, 0)
        )

        save_path = config['results']['save_path']
        os.makedirs(save_path, exist_ok=True)

        # Save metrics
        metric_df = pd.DataFrame(
            [[np.nanmedian(metrics_dict[key]), np.nanstd(metrics_dict[key]), np.nanmean(metrics_dict[key])]
             for key in metrics_dict.keys()],
            index=metrics_dict.keys(),
            columns=['median', 'STD', 'mean']
        )
        metric_df.to_csv(os.path.join(save_path, "metrics.csv"))
        log.info(f"Metrics saved to {save_path}")

        # Save predictions and observations
        np.save(os.path.join(save_path, "test_preds.npy"), selected_preds.numpy())
        np.save(os.path.join(save_path, "test_obs.npy"), test_obs.numpy())

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    except Exception as e:
        log.error(f"Error: {e}")
        raise e

    finally:
        print("Cleaning up...")
        torch.cuda.empty_cache()

        total_time = time.perf_counter() - start_time
        log.info(
            f"| {config['mode']} completed |"
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        )


if __name__ == "__main__":
    main()
