import logging
import os
from typing import Any, Optional, Union

import torch
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path


from dmg.core.utils import save_model
from dmg.core.calc import Metrics

from dmg.models.phy_models.hbv_2_mts import Hbv_2_mts
from dmg.models.neural_networks import LstmMlpModel, LstmMlp2Model, StackLstmMlpModel
from dmg.models.delta_models.mts_dpl_model import MtsDplModel as DplModel


log = logging.getLogger(__name__)


class MtsModelHandler(torch.nn.Module):
    """Streamlines handling of differentiable models and multimodel ensembles.

    This interface additionally acts as a link to the CSDMS BMI, enabling
    compatibility with the NOAA-OWP NextGen framework.

    Features
    - Model initialization (new or from a checkpoint)
    - Loss calculation
    - Forwarding for single/multi-model setups
    - (Planned) Multimodel ensembles/loss and multi-GPU compute

    Parameters
    ----------
    config
        Configuration settings for the model.
    device
        Device to run the model on.
    verbose
        Whether to print verbose output.

    NOTE: this is a temporary implementation for testing the MTS HBV2 model and
        will later be merged with the main model handler.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[str] = None,
        verbose=False,
    ) -> None:
        super().__init__()
        self.config = config
        self.name = 'Differentiable Model Handler'
        self.model_path = config['model_path']
        self.verbose = verbose
        self.target_name = config['train']['target'][0]
        self.models = self.list_models()
        self.loss_func = None

        if device is None:
            self.device = (
                torch.device(config['gpu_id'])
                if config['device'] != 'cpu'
                else torch.device('cpu')
            )
        else:
            self.device = torch.device(device)

        self._init_models()

    def _init_models(self):
        config = self.config
        device = self.device

        phy_model = Hbv_2_mts(
            low_freq_config=config['delta_model']['phy_model']['low_freq_model'],
            high_freq_config=config['delta_model']['phy_model']['high_freq_model'],
            device=torch.device(device),
        )
        low_freq_nn_model = LstmMlpModel(
            nx1=len(config['delta_model']['nn_model']['low_freq_model']['forcings'])
            + len(config['delta_model']['nn_model']['low_freq_model']['attributes']),
            ny1=phy_model.low_freq_model.learnable_param_count1,
            hiddeninv1=config['delta_model']['nn_model']['low_freq_model'][
                'lstm_hidden_size'
            ],
            nx2=len(config['delta_model']['nn_model']['low_freq_model']['attributes']),
            ny2=phy_model.low_freq_model.learnable_param_count2,
            hiddeninv2=config['delta_model']['nn_model']['low_freq_model'][
                'mlp_hidden_size'
            ],
            dr1=config['delta_model']['nn_model']['low_freq_model']['lstm_dropout'],
            dr2=config['delta_model']['nn_model']['low_freq_model']['mlp_dropout'],
            sub_batch_size=config['delta_model']['nn_model']['sub_batch_size'],
            device=torch.device(device),
        )
        high_freq_nn_model = LstmMlp2Model(
            nx1=len(config['delta_model']['nn_model']['high_freq_model']['forcings'])
            + len(config['delta_model']['nn_model']['high_freq_model']['attributes']),
            ny1=phy_model.high_freq_model.learnable_param_count1,
            hiddeninv1=config['delta_model']['nn_model']['high_freq_model'][
                'lstm_hidden_size'
            ],
            nx2=len(config['delta_model']['nn_model']['high_freq_model']['attributes']),
            ny2=phy_model.high_freq_model.learnable_param_count2,
            hiddeninv2=config['delta_model']['nn_model']['high_freq_model'][
                'mlp_hidden_size'
            ],
            nx3=len(
                config['delta_model']['nn_model']['high_freq_model']['attributes2']
            ),
            ny3=phy_model.high_freq_model.learnable_param_count3,
            hiddeninv3=config['delta_model']['nn_model']['high_freq_model'][
                'mlp2_hidden_size'
            ],
            dr1=config['delta_model']['nn_model']['high_freq_model']['lstm_dropout'],
            dr2=config['delta_model']['nn_model']['high_freq_model']['mlp_dropout'],
            dr3=config['delta_model']['nn_model']['high_freq_model']['mlp2_dropout'],
            sub_batch_size=config['delta_model']['nn_model']['sub_batch_size'],
            device=torch.device(device),
        )
        nn_model = StackLstmMlpModel(low_freq_nn_model, high_freq_nn_model)
        dpl_model = DplModel(
            phy_model=phy_model,
            nn_model=nn_model,
            config=config['delta_model'],
            device=torch.device(device),
        )
        self.dpl_model = dpl_model
        self.train_warmup = phy_model.train_warmup

    def list_models(self) -> list[str]:
        """List of models specified in the configuration.

        Returns
        -------
        list[str]
            List of model names.
        """
        models = self.config['delta_model']['phy_model']['name']
        return models

    def load_model(self, epoch: int = 0):
        """Load a specific model from a checkpoint.

        Parameters
        ----------
        epoch
            Epoch to load the model from.
        """
        name = self.models[0]

        if epoch == 0:
            # Leave model uninitialized for training.
            if self.verbose:
                log.info(f"Created new model: {name}")
        else:
            # Initialize model from checkpoint state dict.
            path = os.path.join(self.model_path, f"d{str(name)}_Ep{str(epoch)}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found for model {name}.",
                )

            self.dpl_model.load_state_dict(
                torch.load(
                    path,
                    weights_only=True,
                    map_location=self.device,
                ),
            )
            self.dpl_model.to(self.device)

            if self.verbose:
                log.info(f"Loaded model: {name}, Ep {epoch}")

    def get_parameters(self):
        """Return all model parameters.

        Returns
        -------
        Iterable
            Model parameters.
        """
        return self.dpl_model.parameters()

    def forward(
        self,
        dataset_dict: dict[str, torch.Tensor],
        mode: str = 'train',
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mode : train, eval, simulate

        Returns
        -------
        torch.Tensor
            Model output for the target variable.
        """
        if mode == 'train':
            self.dpl_model.phy_model.set_mode(is_simulate=False)
            self.dpl_model.nn_model.set_mode(is_simulate=False)
            self.dpl_model.train()
            output = self.dpl_model(dataset_dict)
        elif mode == 'eval':
            self.dpl_model.phy_model.set_mode(is_simulate=False)
            self.dpl_model.nn_model.set_mode(is_simulate=False)
            self.dpl_model.eval()
            with torch.no_grad():
                output = self.dpl_model(dataset_dict)
        elif mode == 'simulate':
            self.dpl_model.phy_model.set_mode(is_simulate=True)
            self.dpl_model.nn_model.set_mode(is_simulate=True)
            self.dpl_model.eval()
            with torch.no_grad():
                output = self.dpl_model(dataset_dict)
        return output[self.target_name]  # (n_t, n_gauges, 1)

    def calc_loss(
        self, predictions: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate the loss between predictions and targets."""
        return self.loss_func(
            predictions[self.train_warmup :],  # (n_t, n_gauges, 1)
            batch['target'].swapaxes(0, 1)[self.train_warmup :],  # (n_t, n_gauges, 1)
            batch['batch_sample'],
        )

    def calc_valid_metrics(
        self, pred_dict: dict[int, torch.Tensor], obs_dict: dict[int, torch.Tensor]
    ) -> pd.DataFrame:
        """
        :param pred_dict: gage_idx: predictions (n_batches, window_size, 1).
        :param obs_dict: gage_idx: observations (n_batches, window_size, 1).
        :return:
        """
        valid_metrics = []
        for gage_idx in pred_dict.keys():
            pred = pred_dict[gage_idx][:, self.train_warmup :, 0].numpy().flatten()
            obs = obs_dict[gage_idx][:, self.train_warmup :, 0].numpy().flatten()
            model_dict = Metrics(pred=pred, target=obs).model_dump()
            model_dict.pop('pred', None)
            model_dict.pop('target', None)
            model_dict = {key: value[0] for key, value in model_dict.items()}
            valid_metrics.append(model_dict)
        valid_metrics = pd.DataFrame(valid_metrics)
        valid_metrics['gage_idx'] = pred_dict.keys()
        return valid_metrics

    def save_predictions(
        self,
        pred_dict: dict[int, torch.Tensor],
        gage_ids: np.ndarray,
        times: np.ndarray,
        filename: Union[Path, str],
    ):
        """Save predictions to a NetCDF file."""

        def save_nc_file(
            data: dict[str, np.ndarray],
            units: dict[str, str],
            gauges: np.ndarray,
            filename: Union[str, Path],
            times: np.ndarray = None,
            ds_attrs: dict = None,
        ) -> None:
            xr_dict = {}
            for key, value in data.items():
                if times is None:
                    xr_dict[key] = xr.DataArray(
                        value,
                        dims=("gauge"),
                        coords={"gauge": gauges},
                        name=key,
                        attrs={"units": units[key], "long_name": key},
                    )
                else:
                    xr_dict[key] = xr.DataArray(
                        value,
                        dims=("gauge", "time"),
                        coords={"gauge": gauges, "time": times},
                        name=key,
                        attrs={"units": units[key], "long_name": key},
                    )
            ds = xr.Dataset(xr_dict)
            ds.attrs = ds_attrs if ds_attrs is not None else {}
            ds.to_netcdf(
                filename,
                format="NETCDF4",
                engine="netcdf4",
                encoding={
                    var: {
                        "zlib": False,
                        "complevel": 0,
                        "shuffle": False,
                        "dtype": "float32",
                    }
                    for var in ds.data_vars
                },
            )

        data = np.concatenate(
            [value[:, :, 0] for value in pred_dict.values()], axis=0
        )  # (n_gauges, n_t)
        data = data[np.array(list(pred_dict.keys())).argsort()]  # sort by gage idx
        data = {self.target_name: data}
        units = {self.target_name: 'mm/hour'}
        times = pd.to_datetime(times, unit='s').astype(str).values
        save_nc_file(
            data=data, units=units, gauges=gage_ids, times=times, filename=filename
        )
        return

    def save_model(self, epoch: int) -> None:
        """Save the current model state."""
        save_model(
            config=self.config,
            model=self.dpl_model,
            model_name=self.models[0],
            epoch=epoch,
        )
        if self.verbose:
            log.info(f"All states saved for ep:{epoch}")
