"""
Vanilla testing/validation for differentiable models & multimodel ensembles.
"""
import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import tqdm
from core.calc.stat import stat_error
from core.data import get_validation_sample
from core.utils import save_outputs
from models.model_handler import ModelHandler
from torch.nn import Module

log = logging.getLogger(__name__)



class TestModel:
    """Generic testing for differentiable models.
    
    Retrieves and formats testing data,
    initializes all individual models, and tests a trained model or ensemble.

    Parameters
    ----------
    config : dict
        Configuration settings for the model and experiment.
    model : Module, optional
        dPL (differentiable parameter learning) model object.
    dataset : dict, optional
        Dataset dictionary containing forcings and attributes
    """
    def __init__(self, config: Dict[str, Any], model: Module = None,
                 dataset: dict = None) -> None:
        self.config = config

        if model is None:
            self.model = ModelHandler(config)
        else:
            self.model = model

        if dataset is None:
            log.info(f"Loading test data")
            self.dataset = get_dataset_dict(config, train=False)
        else:
            self.dataset = dataset

    def run(self) -> None:
        """Run testing loop."""
        log.info(f"Testing model: {self.config['name']}")
        
        # Get model predictions and observation data.
        log.info(f"Testing on batches of {self.config['test_batch']} basins...")
        batch_predictions = self._get_batch_predictions()
        observations = self.dataset_dict['target'][:, :, :]

        log.info(f"Saving model results.")
        save_outputs(self.config, batch_predictions, observations)
            
        # Calculate model performance statistics.
        self.calc_metrics(batch_predictions, observations)

    def _get_batch_predictions(self) -> List[Dict[str, torch.Tensor]]:
        """Get model predictions for each batch of test data."""
        prediction_list = []
        for i in tqdm.tqdm(range(len(self.iS)), leave=False, dynamic_ncols=True):
            dataset_sample = get_validation_sample(self.config,
                                                   self.dataset_dict,
                                                   self.iS[i],
                                                   self.iE[i])
            # Model forward pass
            prediction = self.model(dataset_sample, eval=True)

            # Compile predictions from each batch.
            model_name = self.config['phy_model']['model'][0]
            prediction_list.append({key: tensor.cpu().detach() for key,
                                        tensor in prediction[model_name].items()})
        return prediction_list

    def calc_metrics(self, batch_predictions: List[Dict[str, torch.Tensor]],
                     observations: torch.Tensor) -> None:
        """Calculate test metrics and save to csv.
        
        Parameters
        ----------
        batch_predictions : List[Dict[str, torch.Tensor]]
            List of predictions for each model, for each batch.
        observations : torch.Tensor
            Observation data for comparison.
        """
        preds_list = []
        obs_list = []
        name_list = []
        
        # Format streamflow predictions and observations.
        flow_preds = torch.cat([d['flow_sim'] for d in batch_predictions], dim=1)
        flow_obs = observations[:, :, self.config['target'].index('00060_Mean')]

        # Remove warmup days for dHBV1.1p.
        if ('HBV1_1p' in self.config['phy_model']['model']) and \
        (self.config['phy_model']['use_warmup_mode']) and (self.config['ensemble_type'] == 'none'):
            pass
        else:
            flow_obs = flow_obs[self.config['phy_model']['warm_up']:, :]

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
