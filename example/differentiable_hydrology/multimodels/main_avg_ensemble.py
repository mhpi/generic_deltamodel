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
# Dev imports
from models.model_handler_dev import ModelHandler as dModel
from omegaconf import DictConfig
from trainers.trainer_dev import Trainer

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
        
        ### Process datasets ###
        log.info("Processing datasets...")
        eval_dataset = get_dataset_dict(config, train=False)

        # Load test predictions
        hbv_flow_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/HBV_1_1p_/NseLossBatch_/dynamic_para/parBETA_parBETAET_parK0_/test1989_1999/flow_sim.npy'
        prms_flow_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/PRMS_/NseLossBatch_/dynamic_para/alpha_scx_cgw_resmax_k1_k2_/test1989_1999/flow_sim.npy'
        sacsma_flow_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/no_ensemble/LSTM_E50_R365_B100_H256_n16_noLogNorm_111111/SACSMA_with_snow_/NseLossBatch_/dynamic_para/pctim_smax_f1_f2_kuz_rexp_f3_f4_pfree_klzp_klzs_parCWH_/test1989_1999/flow_sim.npy'


        ## Save Path ##
        save_path = '/data/lgl5139/project_blue_eyes/generic_diffModel/results/camels_531/train_1999_2008/3_forcing/ensemble_avg/HBV_1_1p_PRMS_SACSMA_with_snow_/NSE_loss_batch'
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # Average the predictions for ensemble:
        test_preds = []
        for flow_path in [hbv_flow_path, prms_flow_path, sacsma_flow_path]:
            preds = np.load(flow_path)
            test_preds.append(torch.from_numpy(preds))

        # Stack and average predictions
        test_preds = torch.stack(test_preds, dim=0)  # Shape: [num_models, 3652, 531]
        predictions = test_preds.mean(dim=0)        # Shape: [3652, 531]

        # Compile flow predictions and corresponding observations
        flow_obs = eval_dataset['target'][:, :, 0]

        if config['dpl_model']['phy_model']['warm_up_states']:
            warm_up = config['dpl_model']['phy_model']['warm_up']
            flow_obs = flow_obs[warm_up:, :]

        preds_list, obs_list, name_list = [], [], []
        preds_list.append(predictions.numpy())
        obs_list.append(np.expand_dims(flow_obs, 2))
        name_list.append('flow')

        # Calculate statistics
        stat_dicts = [
            metrics(np.swapaxes(pred.squeeze(), 1, 0), np.swapaxes(obs.squeeze(), 1, 0))
            for pred, obs in zip(preds_list, obs_list)
        ]

        # Save metrics
        for stat_dict, name in zip(stat_dicts, name_list):
            metric_df = pd.DataFrame(
                [[np.nanmedian(stat_dict[key]), np.nanstd(stat_dict[key]), np.nanmean(stat_dict[key])]
                for key in stat_dict.keys()],
                index=stat_dict.keys(), columns=['median', 'STD', 'mean']
            )
            metric_df.to_csv(os.path.join(save_path, f'metrics_{name}.csv'))
            print('NSE:', np.nanmedian(stat_dict['NSE']))


        file_name = 'flow_sim.npy'       
        np.save(os.path.join(save_path, file_name), predictions.numpy)

        file_name = 'flow_obs.npy'       
        np.save(os.path.join(save_path, file_name), flow_obs)

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
