"""
Test a differentiable model with CONUS MERIT data.
"""
import logging
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import tqdm

from conf.config import Config
from core.calc.stat import stat_error
from core.data import take_sample_test_merit
from core.data.conus_merit_processor import get_data_dict
from core.utils import save_outputs
from models.model_handler import ModelHandler
from models.multimodels.model_average import model_average

from models.differentiable_model import dPLHydroModel


log = logging.getLogger(__name__)



class TestModel:
    """
    High-level handler for testing models on CONUS MERIT data; retrieves and 
    formats test data, initializes model, and tests a trained model
    and runs training.
    """
    def __init__(self, config: Config):
        # self.start = time.time()
        self.config = config

        # Initializing dPLHydro model
        self.dplh_model_handler = ModelHandler(self.config).to(self.config['device'])

    def run(self, experiment_tracker) -> None:
        log.info(f"Testing model: {self.config['name']} | Collecting testing data")

        # Get dataset dictionary
        self._get_data_dict()
        
        # Get model predictions and observation data.
        log.info(f"Testing on batches of {self.config['test_batch']} basins...")
        batched_preds_list = self._get_model_predictions()
        y_obs = self.dataset_dict['obs'][:, :, :]

        log.info(f"Saving model results.")
        save_outputs(self.config, batched_preds_list, y_obs)
            
        # Calculate model result statistics.
        self.calc_metrics(batched_preds_list, y_obs)

    def _get_data_dict(self) -> None:
        """
        Get dictionary of input data.

        iS, iE: arrays of start and end pairs of basin indicies for batching.
        """
        # Load forcings + attributes.
        self.dataset_dict, self.config = get_data_dict(self.config, train=False)

        # Get basin batch start and end indicies.
        self._get_batch_bounds()

    def _get_model_predictions(self) -> List[Dict[str, torch.Tensor]]:
        """
        Get predictions from a trained model.
        """
        batched_preds_list = []
        model_name = self.config['hydro_models'][0]
        for i in tqdm.tqdm(range(len(self.iS)), leave=False, dynamic_ncols=True):
            dataset_dict_sample = take_sample_test_merit(self.config,
                                                          self.dataset_dict,
                                                          self.iS[i],
                                                          self.iE[i],
                                                          )
            # Forward pass for hydrology models.
            hydro_preds = self.dplh_model_handler(dataset_dict_sample, eval=True)

            batched_preds_list.append({key: tensor.cpu().detach() for key,
                                        tensor in hydro_preds[model_name].items()})
        return batched_preds_list

    def calc_metrics(self, batched_preds_list: List[Dict[str, torch.Tensor]],
                     y_obs: torch.Tensor) -> None:
        """
        Calculate test metrics and save to csv.

        TODO: You could streamline this up a little more.
        """
        preds_list = []
        obs_list = []
        name_list = []
        
        # Format streamflow predictions and observations, and remove warm up days.
        target_id = self.config['target'].index('00060_Mean')
        flow_preds = torch.cat([d['flow_sim'] for d in batched_preds_list], dim=1)
        flow_obs = y_obs[self.config['warm_up']:, :, target_id]

        preds_list.append(flow_preds.numpy())
        obs_list.append(np.expand_dims(flow_obs, 2))
        name_list.append('flow')

        #######################
        # if SAVE_DATA:
        #     ## Added to save prediction and observation data:
        #     np.save(OUT_DATA_SAVE_PATH + 'sacsma_dyn_sf_pred.npy',preds_list)
        #     np.save(OUT_DATA_SAVE_PATH + 'sacsma_sf_obs.npy',obs_list)
        #######################

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

    def _get_batch_bounds(self) -> None: 
        """
        Get basin batch bounds for testing.

        :return iS, iE: arrays of start and end pairs of basin indicies for
            batching.
        """
        nmerit_list = []
        merit_id = 0
        area_info = self.dataset_dict['area_info']
        gage_key = self.dataset_dict['gage_key']

        for gage in gage_key:
            merit_id = merit_id + len(area_info[gage]['COMID'])
            nmerit_list.append(merit_id)

        iS = []
        prev_n_merit = 0
        merit_interval = self.config['test_batch']

        for gage_idx, n_merit in enumerate(nmerit_list):
            if (n_merit - prev_n_merit) >= merit_interval:
                iS.append(gage_idx)
                prev_n_merit= nmerit_list[gage_idx]

        self.ngrid = len(gage_key)
        self.iS = np.array(iS)
        self.iE = np.append(self.iS[1:], self.ngrid)
