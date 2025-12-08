from typing import Any, Optional

import torch
from tqdm import tqdm

from dmg.models.phy_models.hbv_2 import Hbv_2
from dmg.models.phy_models.hbv_2h import Hbv_2h


class Hbv_2_mts(torch.nn.Module):
    """HBV 2.0, multi timescale, distributed UH.

    Multi-component, multi-scale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins.

    Authors
    -------
    -   Wencong Yang
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        low_freq_config: Optional[dict[str, Any]] = None,
        high_freq_config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.low_freq_model = Hbv_2(low_freq_config, device=device)
        self.low_freq_model.initialize = True
        self.high_freq_model = Hbv_2h(high_freq_config, device=device)

        # learnable transfer
        self.state_transfer_model = torch.nn.ModuleDict(
            {
                name: torch.nn.Sequential(
                    torch.nn.Linear(
                        self.low_freq_model.nmul, self.high_freq_model.nmul
                    ),
                    torch.nn.ReLU(),
                )
                for name in self.high_freq_model.state_names
            }
        )
        # # identity transfer
        # self.state_transfer_model = torch.nn.ModuleDict({
        #     name: torch.nn.Identity()
        #     for name in self.high_freq_model.state_names
        # })

        self.train_spatial_chunk_size = high_freq_config['train_spatial_chunk_size']
        self.simulate_spatial_chunk_size = high_freq_config[
            'simulate_spatial_chunk_size'
        ]
        self.simulate_temporal_chunk_size = high_freq_config[
            'simulate_temporal_chunk_size'
        ]
        self.spatial_chunk_size = self.train_spatial_chunk_size
        self.simulate_mode = False

        self.train_warmup = high_freq_config[
            'train_warmup'
        ]  # warmup steps for routing during training

    def _forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Base forward."""
        low_freq_parameters, high_freq_parameters = parameters

        # transfer states
        low_freq_x_dict = {
            'x_phy': x_dict['x_phy_low_freq'],
            'ac_all': x_dict['ac_all'],
            'elev_all': x_dict['elev_all'],
            'muwts': x_dict.get('muwts', None),
        }
        low_freq_states = self.low_freq_model(
            low_freq_x_dict,
            low_freq_parameters,
        )
        states = self.state_transfer(low_freq_states)

        # transfer parameters
        phy_dy_params_dict, phy_static_params_dict, distr_params_dict = (
            self.param_transfer(
                low_freq_parameters,
                high_freq_parameters,
            )
        )

        # run the model
        x = x_dict['x_phy_high_freq']
        # Ac = x_dict['ac_all'].unsqueeze(-1).repeat(1, self.high_freq_model.nmul)
        # Elevation = x_dict['elev_all'].unsqueeze(-1).repeat(1, self.high_freq_model.nmul)
        Ac = x_dict['ac_all'].unsqueeze(-1).expand(-1, self.high_freq_model.nmul)
        Elevation = (
            x_dict['elev_all'].unsqueeze(-1).expand(-1, self.high_freq_model.nmul)
        )
        outlet_topo = x_dict['outlet_topo']
        areas = x_dict['areas']

        predictions = self.high_freq_model.PBM(
            forcing=x,
            Ac=Ac,
            Elevation=Elevation,
            states=tuple(states),
            phy_dy_params_dict=phy_dy_params_dict,
            phy_static_params_dict=phy_static_params_dict,
            outlet_topo=outlet_topo,
            areas=areas,
            distr_params_dict=distr_params_dict,
        )
        return predictions

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Foward supports spatial and temporal chunking.
        x_dict and parameters can be in cpu for simulation mode to save GPU memory.
        """
        device = self.device
        n_units = x_dict['areas'].shape[0]
        spatial_chunk_size = self.spatial_chunk_size
        temporal_chunk_size = self.simulate_temporal_chunk_size
        train_warmup = self.train_warmup

        if (not self.simulate_mode) and (n_units <= spatial_chunk_size):
            self.high_freq_model.use_distr_routing = False
            return self._forward(x_dict, parameters)

        ### chunked runoff generation for simulation mode or large training batches
        self.high_freq_model.use_distr_routing = False
        preds_list = []
        for i in tqdm(
            range(0, n_units, spatial_chunk_size), desc="Spatial runoff chunks"
        ):
            # print('i: ', i)
            end_idx = min(i + spatial_chunk_size, n_units)
            reach_idx = (x_dict['outlet_topo'] == 1).nonzero(as_tuple=False)
            idxs_in_chunk = (reach_idx[:, 1] >= i) & (reach_idx[:, 1] < end_idx)
            chunk_x_dict = {
                # 'xc_nn_norm_low_freq': x_dict['xc_nn_norm_low_freq'][:, i:end_idx],
                # 'xc_nn_norm_high_freq': x_dict['xc_nn_norm_high_freq'][:, i:end_idx],
                # 'c_nn_norm': x_dict['c_nn_norm'][i:end_idx],
                # 'rc_nn_norm': x_dict['rc_nn_norm'][idxs_in_chunk],
                'x_phy_low_freq': x_dict['x_phy_low_freq'][:, i:end_idx].to(device),
                'x_phy_high_freq': x_dict['x_phy_high_freq'][:, i:end_idx].to(device),
                'ac_all': x_dict['ac_all'][i:end_idx].to(device),
                'elev_all': x_dict['elev_all'][i:end_idx].to(device),
                'areas': x_dict['areas'][i:end_idx].to(device),
                'outlet_topo': x_dict['outlet_topo'][:, i:end_idx].to(device),
            }
            chunk_parameters = (
                [
                    parameters[0][0][:, i:end_idx].to(
                        device
                    ),  # low-freq dynamic phy params
                    parameters[0][1][i:end_idx].to(
                        device
                    ),  # low-freq static phy params
                ],
                [
                    parameters[1][0][:, i:end_idx].to(
                        device
                    ),  # high-freq dynamic phy params
                    parameters[1][1][i:end_idx].to(
                        device
                    ),  # high-freq static phy params
                    parameters[1][2][idxs_in_chunk].to(
                        device
                    ),  # high-freq distributed params
                ],
            )
            chunk_predictions = self._forward(chunk_x_dict, chunk_parameters)
            preds_list.append(chunk_predictions)
        predictions = self.concat_spatial_chunks(preds_list)
        runoff = predictions['Qs']
        high_freq_length = runoff.shape[0]

        ### chunked routing
        _, _, _, distr_params = self.high_freq_model.unpack_parameters(parameters[1])
        distr_params_dict = self.high_freq_model.descale_distr_parameters(distr_params)
        distr_params_dict = {
            key: value.to(device) for key, value in distr_params_dict.items()
        }
        outlet_topo = x_dict['outlet_topo'].to(device)
        areas = x_dict['areas'].to(device)
        # total_steps = (
        #     high_freq_length - train_warmup + temporal_chunk_size - 1
        # ) // temporal_chunk_size
        preds_list = []
        for t in tqdm(
            range(train_warmup, high_freq_length, temporal_chunk_size),
            desc="Temporal routing chunks",
        ):
            # print('t: ', t)
            end_t = min(t + temporal_chunk_size, high_freq_length)
            chunk_runoff = runoff[t - train_warmup : end_t]
            chunk_predictions = self.high_freq_model.distr_routing(
                Qs=chunk_runoff,
                distr_params_dict=distr_params_dict,
                outlet_topo=outlet_topo,
                areas=areas,
            )
            if t > train_warmup:  # remove routing warmup for all but first chunk
                chunk_predictions = {
                    key: value[train_warmup:]
                    for key, value in chunk_predictions.items()
                }
            preds_list.append(chunk_predictions)
        routing_predictions = self.concat_temporal_chunks(preds_list)
        predictions['streamflow'] = routing_predictions['Qs_rout']
        return predictions

    def concat_spatial_chunks(self, pred_list: list[dict[str, torch.Tensor]]):
        """Concatenate spatial chunk pedictions."""
        output = {}
        for key in pred_list[0].keys():
            if pred_list[0][key].ndim == 3:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=1
                )  # (window_size, n_units, nmul)
            else:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=0
                )  # (n_units, nmul) or (n_units,)
        return output

    def concat_temporal_chunks(self, pred_list: list[dict[str, torch.Tensor]]):
        """Concatenate temporal chunk predictions."""
        output = {}
        for key in pred_list[0].keys():
            if pred_list[0][key].ndim == 3:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=0
                )  # (window_size, n, nmul)
            else:
                output[key] = pred_list[0][key]  # (n_units, nmul) or (n_units,)
        return output

    def set_mode(self, is_simulate: bool):
        """Set simulate mode."""
        if is_simulate:
            self.spatial_chunk_size = self.simulate_spatial_chunk_size
            self.simulate_mode = True
        else:
            self.spatial_chunk_size = self.train_spatial_chunk_size
            self.simulate_mode = False

    def param_transfer(
        self,
        low_freq_parameters: list[torch.Tensor],
        high_freq_parameters: list[torch.Tensor],
    ):
        """Map low-frequency parameters to high-frequency parameters."""
        warmup_phy_dy_params, warmup_phy_static_params, warmup_routing_params = (
            self.low_freq_model.unpack_parameters(low_freq_parameters)
        )
        phy_dy_params, phy_static_params, routing_params, distr_params = (
            self.high_freq_model.unpack_parameters(high_freq_parameters)
        )
        # new dynamic params
        phy_dy_params_dict = self.high_freq_model.descale_phy_dy_parameters(
            phy_dy_params, dy_list=self.high_freq_model.dynamic_params
        )
        # keep warmup static params, add high-freq specific static params
        static_param_names = [
            param
            for param in self.high_freq_model.phy_param_names
            if param not in self.high_freq_model.dynamic_params
        ]
        warmup_static_param_names = [
            param
            for param in self.low_freq_model.phy_param_names
            if param not in self.low_freq_model.dynamic_params
        ]
        var_indexes = [
            i
            for i, param in enumerate(static_param_names)
            if param not in warmup_static_param_names
        ]
        phy_static_params_dict = self.high_freq_model.descale_phy_stat_parameters(
            torch.concat(
                [warmup_phy_static_params, phy_static_params[:, var_indexes]], dim=1
            ),
            stat_list=static_param_names,
        )
        # new distributed params
        distr_params_dict = self.high_freq_model.descale_distr_parameters(distr_params)
        # new routing params
        if self.high_freq_model.routing:
            self.high_freq_model.routing_param_dict = (
                self.high_freq_model.descale_rout_parameters(routing_params)
            )
        return phy_dy_params_dict, phy_static_params_dict, distr_params_dict

    def state_transfer(self, states: list[torch.Tensor]):
        """Map low-frequency states to high-frequency states."""
        states_dict = dict(zip(self.high_freq_model.state_names, states))
        return [
            self.state_transfer_model[key](states_dict[key])
            for key in states_dict.keys()
        ]
