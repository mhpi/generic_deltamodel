"""
Vanilla testing/validation for differentiable models & multimodel ensembles.
"""
import os
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import tqdm

from conf.config import Config
from core.calc.stat import stat_error
from core.data import take_sample_test
from core.utils import save_outputs
from models.model_handler import ModelHandler

log = logging.getLogger(__name__)



class TestModel:
    """
    High-level testing handler; retrieves and formats testing data,
    initializes all individual models, and tests a trained model.
    """
    def __init__(self, config: Config):
        # self.start = time.time()
        self.config = config

        # Initializing collection of dPL hydrology models.
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])

    def run(self) -> None:
        log.info(f"Testing model: {self.config['name']} | Collecting testing data")

        # Get dataset dictionary.
        self._get_data_dict()
        
        # Get model predictions and observation data.
        log.info(f"Testing on batches of {self.config['test_batch']} basins...")
        batched_preds_list = self._get_model_predictions()
        y_obs = self.dataset_dict['obs'][:, :, :]

        log.info(f"Saving model results.")
        save_outputs(self.config, batched_preds_list, y_obs)
            
        # Calculate model result statistics.
        self.calc_metrics(batched_preds_list, y_obs)


    def _get_model_predictions(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get predictions from a trained model.
        """
        batched_preds_list = []
        for i in tqdm.tqdm(range(len(self.iS)), leave=False, dynamic_ncols=True):
            dataset_dict_sample = take_sample_test(self.config,
                                                   self.dataset_dict,
                                                   self.iS[i],
                                                   self.iE[i])
            # Forward pass for hydrology models.
            hydro_preds = self.dplh_model_handler(dataset_dict_sample, eval=True)

            # Compile predictions from each batch.
            model_name = self.config['physics_model']['models'][0]
            batched_preds_list.append({key: tensor.cpu().detach() for key,
                                        tensor in hydro_preds[model_name].items()})
        return batched_preds_list

    def calc_metrics(self, batched_preds_list: List[Dict[str, torch.Tensor]],
                     y_obs: torch.Tensor) -> None:
        """ Calculate test metrics and save to csv. """
        preds_list = []
        obs_list = []
        name_list = []
        
        # Format streamflow predictions and observations.
        flow_preds = torch.cat([d['flow_sim'] for d in batched_preds_list], dim=1)
        flow_obs = y_obs[:, :, self.config['target'].index('00060_Mean')]

        # Remove warmup days for dHBV1.1p.
        if ('hbv_capillary' in self.config['physics_model']['models']) and \
        (self.config['hbvcap_no_warm']) and (self.config['ensemble_type'] == 'none'):
            pass
        else:
            flow_obs = flow_obs[self.config['warm_up']:, :]
        # flow_preds = flow_preds[self.config['warm_up']:, :, :]

        preds_list.append(flow_preds.numpy())
        obs_list.append(np.expand_dims(flow_obs, 2))
        name_list.append('flow')

        # Swap axes for shape [basins, days]
        statDictLst = [
            stat_error(np.swapaxes(x.squeeze(), 1, 0), np.swapaxes(y.squeeze(), 1, 0))
            for (x, y) in zip(preds_list, obs_list)
        ]

        # Calculate statistics on model results/performance.
        for stat, name in zip(statDictLst, name_list):
            count = 0
            mdstd = np.zeros([len(stat), 3])
            for key in stat.keys():
                median = np.nanmedian(stat[key])
                STD = np.nanstd(stat[key])
                mean = np.nanmean(stat[key])
                k = np.array([[median, STD, mean]])
                mdstd[count] = k
                count += 1
            mdstd = pd.DataFrame(
                mdstd, index=stat.keys(), columns=['median', 'STD', 'mean']
            )
            # Save statistics to CSV
            mdstd.to_csv((os.path.join(self.config['testing_dir'], 'mdstd_' + name + '.csv')))
