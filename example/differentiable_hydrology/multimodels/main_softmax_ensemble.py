""" Main script for running differentiable model experiments."""
import logging
import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
from core.calc.stat import metrics
from core.data.dataset_loading import get_dataset_dict
from core.utils import initialize_config, print_config, set_randomseed
from omegaconf import DictConfig

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

        ### Stat to use to quantify model performance ###
        stat_name = 'NSE'

        ## Save Path ##
        # save_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/ensemble_softmax/' + stat_name + '/HBV_1_1p_PRMS_SACSMA_with_snow_/NSE_loss_batch'
        # save_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/ensemble_softmax/' + stat_name + '/PRMS_SACSMA_with_snow_/NSE_loss_batch'
        save_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/ensemble_softmax/' + stat_name + '/HBV_1_1p_SACSMA_with_snow_/NSE_loss_batch'
        # save_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/ensemble_softmax/' + stat_name + '/HBV_1_1p_PRMS_/NSE_loss_batch'

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        ### Load train and test predictions ###
        log.info("Loading predictions...")
        data_paths = [
            '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/HBV_1_1p_/NseLossBatch_/dynamic_para/parBETA_parBETAET_parK0_/test1989_1999/',
            # '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/PRMS_/NseLossBatch_/dynamic_para/alpha_scx_cgw_resmax_k1_k2_/test1989_1999/',
            '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/SACSMA_with_snow_/NseLossBatch_/dynamic_para/pctim_smax_f1_f2_kuz_rexp_f3_f4_pfree_klzp_klzs_parCWH_/test1989_1999/'
        ]


        train_preds = []  # List of predictions for the training period
        test_preds = []  # List of predictions for the testing period
        warm_up = config['dpl_model']['phy_model']['warm_up']

        for data_path in data_paths:
            train_preds.append(torch.from_numpy(np.load(os.path.join(data_path, "flow_sim_train_period.npy"))))
            test_preds.append(torch.from_numpy(np.load(os.path.join(data_path, "flow_sim.npy"))))

        train_preds = torch.stack(train_preds, dim=0).squeeze()  # Shape: [num_models, n_train_samples, n_basins]
        test_preds = torch.stack(test_preds, dim=0)    # Shape: [num_models, n_test_samples, n_basins]
        test_preds = test_preds.squeeze() 


        ### Compute softmax weights on training period ###
        log.info("Computing softmax weights based on training period...")
        train_obs = get_dataset_dict(config, train=True)['target'][warm_up:, :, 0]  # Shape: [n_train_samples, n_basins]
        train_obs = torch.from_numpy(train_obs)

        # Calculate statistics for each model on each basin for train data.
        train_metric = torch.zeros(train_preds.shape[0], train_preds.shape[2])  # Shape: [num_models, n_basins]

        for i, preds in enumerate(train_preds):
            train_metric[i,:] = torch.from_numpy(metrics(np.swapaxes(preds.squeeze().numpy(), 1, 0), np.swapaxes(train_obs.numpy(), 1, 0))[stat_name])

        # Compute softmax weights across models for each basin
        softmax_weights = torch.softmax(train_metric, dim=0)  # Shape: [num_models, n_basins]

        assert torch.allclose(softmax_weights.sum(dim=0), torch.ones_like(softmax_weights.sum(dim=0))), \
            "Softmax weights should sum to 1 across models for each basin."


        ### Apply weights to test period predictions ###
        log.info("Applying softmax weights to test period predictions...")
        selected_preds = (softmax_weights.unsqueeze(1) * test_preds).sum(dim=0)  # Shape: [n_test_samples, n_basins]


        ### Evaluate and Save Metrics ###
        log.info("Calculating metrics...")
        test_obs = get_dataset_dict(config, train=False)['target'][warm_up:, :, 0]  # Shape: [n_test_samples, n_basins]

        metrics_dict = metrics(
            np.swapaxes(selected_preds.numpy(), 1, 0),
            np.swapaxes(test_obs, 1, 0)
        )

        # Save metrics
        metric_df = pd.DataFrame(
            [[np.nanmedian(metrics_dict[key]), np.nanstd(metrics_dict[key]), np.nanmean(metrics_dict[key])]
             for key in metrics_dict.keys()],
            index=metrics_dict.keys(),
            columns=['median', 'STD', 'mean']
        )
        metric_df.to_csv(os.path.join(save_path, "metrics.csv"))
        log.info(f"Metrics saved to {save_path}")

        # Save selected predictions and observations
        np.save(os.path.join(save_path, "test_preds.npy"), selected_preds.numpy())
        np.save(os.path.join(save_path, "test_obs.npy"), test_obs)
        np.save(os.path.join(save_path, "softmax_weights.npy"), softmax_weights.numpy())


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
