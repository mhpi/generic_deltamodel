from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


class Hbv_2h(torch.nn.Module):
    """HBV 2.0 hourly, distributed UH.

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
            config: Optional[dict[str, Any]] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 2.0UH hourly'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = False
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.device = device

        self.dt = 1.0 / 24
        self.parameter_bounds = {
            'parBETA': [1.0, 6.0],
            'parFC': [50, 1000],
            'parK0': [0.05, 0.9],
            'parK1': [0.01, 0.5],
            'parK2': [0.001, 0.2],
            'parLP': [0.2, 1],
            'parPERC': [0, 10],
            'parUZL': [0, 100],
            'parTT': [-2.5, 2.5],
            'parCFMAX': [0.5, 10],
            'parCFR': [0, 0.1],
            'parCWH': [0, 0.2],
            'parBETAET': [0.3, 5],
            'parC': [0, 1],
            'parRT': [0, 20],
            'parAC': [0, 2500],
            # Infiltration parameters for hourly
            'parF0': [5.0 / self.dt, 120.0 / self.dt],  # dry (max) infiltration capacity, mm/day
            'parFMIN': [0.0, 1.0],  # wet (min) capacity ratio
            'parALPHA': [0.5, 5.0],  # shape of f(s); larger -> more thresholdy
        }
        self.routing_parameter_bounds = {
            'rout_a': [0, 5.0],
            'rout_b': [0, 12.0],
        }

        self.distr_parameter_bounds = {
            'rout_a': [0, 5.0],
            'rout_b': [0, 12.0],
            'rout_tau': [0, 48.0]
        }

        self.lenF = 72  # Length of unit hydrograph
        self.muwts = None
        self.use_distr_routing = True
        self.state_names = ['SNOWPACK', 'MELTWATER', 'SM', 'SUZ', 'SLZ']
        self.infiltration = True
        self.lag_uh = True
        if not self.infiltration:
            self.parameter_bounds.pop('parF0')
            self.parameter_bounds.pop('parFMIN')
            self.parameter_bounds.pop('parALPHA')
        if not self.lag_uh:
            self.distr_parameter_bounds.pop('rout_tau')

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get(self.__class__.__name__, self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = (len(self.phy_param_names) - len(self.dynamic_params)) * self.nmul \
                                      + len(self.routing_param_names)
        self.learnable_param_count3 = len(self.distr_parameter_bounds)
        self.learnable_param_count = self.learnable_param_count1 + self.learnable_param_count2 + self.learnable_param_count3

    def unpack_parameters(
            self,
            parameters: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.

        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        dy_param_count = len(self.dynamic_params)
        dif_count = phy_param_count - dy_param_count

        # Physical dynamic parameters
        phy_dy_params = parameters[0].view(
            parameters[0].shape[0],
            parameters[0].shape[1],
            dy_param_count,
            self.nmul,
        )

        # Physical static parameters
        phy_static_params = parameters[1][:, :dif_count * self.nmul].view(
            parameters[1].shape[0],
            dif_count,
            self.nmul,
        )

        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = parameters[1][:, dif_count * self.nmul:]

        # Distributed routing parameters
        distr_params = parameters[2]

        return phy_dy_params, phy_static_params, routing_params, distr_params

    def descale_phy_dy_parameters(
            self,
            phy_dy_params: torch.Tensor,
            dy_list: list,
    ) -> dict[str, torch.Tensor]:
        """Descale physical parameters.

        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_dy_params.size(0)
        n_grid = phy_dy_params.size(1)

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur.
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(dy_list):
            staPar = phy_dy_params[-1, :, i, :].unsqueeze(0).repeat([n_steps, 1, 1])

            dynPar = phy_dy_params[:, :, i, :]
            drmask = torch.bernoulli(pmat).detach_().to(self.device)

            comPar = dynPar * (1 - drmask) + staPar * drmask
            param_dict[name] = change_param_range(
                param=comPar,
                bounds=self.parameter_bounds[name],
            )
        return param_dict

    def descale_phy_stat_parameters(
            self,
            phy_stat_params: torch.Tensor,
            stat_list: list,
    ) -> dict[str, torch.Tensor]:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(stat_list):
            param = phy_stat_params[:, i, :]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.parameter_bounds[name],
            )
        return parameter_dict

    def descale_rout_parameters(
            self,
            routing_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def descale_distr_parameters(
            self,
            distr_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Descale distributed routing parameters.
        Parameters
        ----------
        distr_params
            Normalized distributed routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled distributed routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.distr_parameter_bounds.keys()):
            param = distr_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.distr_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: list[torch.Tensor],
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for HBV1.1p.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        Ac = x_dict['ac_all'].unsqueeze(-1).repeat(1, self.nmul)
        Elevation = x_dict['elev_all'].unsqueeze(-1).repeat(1, self.nmul)
        outlet_topo = x_dict['outlet_topo']
        areas = x_dict['areas']
        self.muwts = x_dict.get('muwts', None)

        # Unpack parameters.
        phy_dy_params, phy_static_params, routing_params, distr_params = self.unpack_parameters(parameters)

        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        n_grid = x.size(1)

        # Initialize model states.
        SNOWPACK = torch.zeros([n_grid, self.nmul],
                               dtype=torch.float32,
                               device=self.device) + 0.001
        MELTWATER = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        SM = torch.zeros([n_grid, self.nmul],
                         dtype=torch.float32,
                         device=self.device) + 0.001
        SUZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001
        SLZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001

        phy_dy_params_dict = self.descale_phy_dy_parameters(
            phy_dy_params,
            dy_list=self.dynamic_params,
        )

        phy_static_params_dict = self.descale_phy_stat_parameters(
            phy_static_params,
            stat_list=[param for param in self.phy_param_names if param not in self.dynamic_params],
        )

        distr_params_dict = self.descale_distr_parameters(
            distr_params
        )

        # Run the model for the remainder of simulation period.
        return self.PBM(
            x,
            Ac,
            Elevation,
            [SNOWPACK, MELTWATER, SM, SUZ, SLZ],
            phy_dy_params_dict,
            phy_static_params_dict,
            outlet_topo,
            areas,
            distr_params_dict
        )

    def PBM(
            self,
            forcing: torch.Tensor,
            Ac: torch.Tensor,
            Elevation: torch.Tensor,
            states: tuple,
            phy_dy_params_dict: dict,
            phy_static_params_dict: dict,
            outlet_topo: torch.Tensor,
            areas: torch.Tensor,
            distr_params_dict: dict
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Run the hourly HBV2 model forward. All output fluxes in mm/hour.

        Parameters
        ----------
        forcing
            Input forcing data.
        states
            Initial model states.
        full_param_dict
            Dictionary of model parameters.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        dt = self.dt

        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')] / dt  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')] / dt  # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        n_steps, n_grid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(n_steps, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001
        Q0_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q1_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q2_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001

        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        recharge_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        excs_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        evapfactor_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        tosoil_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PERC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SWE_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        capillary_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        ################## New: Save model states for all time steps. ##################
        SNOWPACK_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        MELTWATER_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SM_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SUZ_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SLZ_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        param_dict = {}
        for t in range(n_steps):
            ###################### New: numerical guardrail for long-sequence running ######################
            SNOWPACK = torch.clamp(SNOWPACK, min=0.0)
            MELTWATER = torch.clamp(MELTWATER, min=0.0)
            SM = torch.clamp(SM, min=self.nearzero)
            SUZ = torch.clamp(SUZ, min=self.nearzero)
            SLZ = torch.clamp(SLZ, min=self.nearzero)
            ################################################################################################

            # Get dynamic parameter values per timestep.
            for key in phy_dy_params_dict.keys():
                param_dict[key] = phy_dy_params_dict[key][t, :, :]
            for key in phy_static_params_dict.keys():
                param_dict[key] = phy_static_params_dict[key][:, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            parTT_new = (Elevation >= 2000).type(torch.float32) * 4.0 + (Elevation < 2000).type(torch.float32) * \
                        param_dict['parTT']
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT_new).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT_new).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW * dt
            melt = param_dict['parCFMAX'] * (Tm[t, :, :] - parTT_new)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt * dt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = param_dict['parCFR'] * param_dict['parCFMAX'] * (
                    parTT_new - Tm[t, :, :]
            )
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing * dt, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = (MELTWATER - (param_dict['parCWH'] * SNOWPACK)) / dt
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil * dt

            ###################### New: Hortonian Infiltration Excess ######################

            if self.infiltration:
                # Hortonian infiltration excess: infiltration capacity as a function of wetness
                W = RAIN + tosoil
                s = torch.clamp(SM / param_dict['parFC'], 0.0, 1.0 - 0.01)  # relative wetness, safe guard for pow and bf/fp16
                parFMIN = param_dict['parFMIN'] * param_dict['parF0']
                with torch.amp.autocast(device_type="cuda", enabled=False):  # torch.pow not stable with bf/fp16 when base ~ 0
                    fcap = parFMIN + (param_dict['parF0'] - parFMIN) * torch.pow(1.0 - s, param_dict['parALPHA'])
                I = torch.minimum(W, fcap)  # goes into soil
                IE = torch.clamp(W - fcap, min=0.0)  # Hortonian excess

                # Soil and evaporation using I
                soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
                soil_wetness = torch.clamp(soil_wetness, 0.0, 1.0)
                recharge = I * soil_wetness
                SM = SM + (I - recharge) * dt
            else:
                soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
                soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
                recharge = (RAIN + tosoil) * soil_wetness
                SM = SM + (RAIN + tosoil - recharge) * dt

            #################################################################################

            excess = (SM - param_dict['parFC']) / dt
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess * dt
            # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
            evapfactor = (SM / (param_dict['parLP'] * param_dict['parFC'])) ** param_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact * dt) / dt
            SM = torch.clamp(SM - ETact * dt, min=self.nearzero)

            # Capillary rise (HBV 1.1p mod) -------------------------------
            capillary = torch.min(SLZ, param_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / param_dict['parFC'], max=1.0)) * dt) / dt

            SM = torch.clamp(SM + capillary * dt, min=self.nearzero)
            SLZ = torch.clamp(SLZ - capillary * dt, min=self.nearzero)

            # Groundwater boxes -------------------------------
            SUZ = SUZ + (recharge + excess) * dt
            PERC = torch.min(SUZ, param_dict['parPERC'] * dt) / dt
            SUZ = SUZ - PERC * dt
            Q0 = param_dict['parK0'] * torch.clamp(SUZ - param_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0 * dt
            Q1 = param_dict['parK1'] * SUZ
            SUZ = SUZ - Q1 * dt
            SLZ = SLZ + PERC * dt

            LF = torch.clamp((Ac - param_dict['parAC']) / 1000, min=-1, max=1) * param_dict['parRT'] * (Ac < 2500) + \
                 torch.exp(torch.clamp(-(Ac - 2500) / 50, min=-10.0, max=0.0)) * param_dict['parRT'] * (Ac >= 2500)
            SLZ = torch.clamp(SLZ + LF * dt, min=0.0)

            Q2 = param_dict['parK2'] * SLZ
            SLZ = SLZ - Q2 * dt

            ###################### New: Add Hortonian Infiltration Excess ######################
            if self.infiltration:
                Qsimmu[t, :, :] = Q0 + Q1 + Q2 + IE
            else:
                Qsimmu[t, :, :] = Q0 + Q1 + Q2
            #####################################################################################
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            AET[t, :, :] = ETact
            SWE_sim[t, :, :] = SNOWPACK
            capillary_sim[t, :, :] = capillary

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC

            ####################### New: Save model states for all time steps. #######################
            SNOWPACK_sim[t, :, :] = SNOWPACK
            MELTWATER_sim[t, :, :] = MELTWATER
            SM_sim[t, :, :] = SM
            SUZ_sim[t, :, :] = SUZ
            SLZ_sim[t, :, :] = SLZ

        # Get the overall average
        # or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(n_steps, n_grid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = uh_gamma(
                self.routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
                self.routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=self.lenF,
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = uh_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = uh_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = uh_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout:
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(n_steps, n_grid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Baseflow index (BFI) calculation
            # BFI_sim = 100 * (torch.sum(Q2_rout, dim=0) / (
            #         torch.sum(Qs, dim=0) + self.nearzero))[:, 0]

            # Return all sim results.
            out_dict = {
                'Qs': Qs * dt,  # Routed Streamflow for units
                # 'srflow': Q0_rout * dt,  # Routed surface runoff
                # 'ssflow': Q1_rout * dt,  # Routed subsurface flow
                # 'gwflow': Q2_rout * dt,  # Routed groundwater flow
                # 'AET_hydro': AET.mean(-1, keepdim=True) * dt,  # Actual ET
                # 'PET_hydro': PETm.mean(-1, keepdim=True) * dt,  # Potential ET
                # 'SWE': SWE_sim.mean(-1, keepdim=True),  # Snow water equivalent
                # 'streamflow_no_rout': Qsim.unsqueeze(dim=2) * dt,  # Streamflow
                # 'srflow_no_rout': Q0_sim.mean(-1, keepdim=True) * dt,  # Surface runoff
                # 'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True) * dt,  # Subsurface flow
                # 'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True) * dt,  # Groundwater flow
                # 'recharge': recharge_sim.mean(-1, keepdim=True) * dt,  # Recharge
                # 'excs': excs_sim.mean(-1, keepdim=True) * dt,  # Excess stored water
                # 'evapfactor': evapfactor_sim.mean(-1, keepdim=True),  # Evaporation factor
                # 'tosoil': tosoil_sim.mean(-1, keepdim=True) * dt,  # Infiltration
                # 'percolation': PERC_sim.mean(-1, keepdim=True) * dt,  # Percolation
                # 'capillary': capillary_sim.mean(-1, keepdim=True) * dt,  # Capillary rise
                # 'BFI': BFI_sim * dt,  # Baseflow index
                # 'SNOWPACK': SNOWPACK_sim,  # New: SNOWPACK
                # 'MELTWATER': MELTWATER_sim,  # New: MELTWATER
                # 'SM': SM_sim,  # New: SM
                # 'SUZ': SUZ_sim,  # New: SUZ
                # 'SLZ': SLZ_sim  # New: SLZ
            }

            if not self.warm_up_states:
                for key in out_dict.keys():
                    if key != 'BFI':
                        out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]

            if self.use_distr_routing:
                distr_out_dict = self.distr_routing(
                    Qs=Qs * dt,
                    distr_params_dict=distr_params_dict,
                    outlet_topo=outlet_topo,
                    areas=areas
                )
                out_dict['streamflow'] = distr_out_dict['Qs_rout']  # Routed Streamflow for gages

            return out_dict

    # def distr_routing(self,
    #                 Qs: torch.Tensor,
    #                 distr_params_dict: dict,
    #                 outlet_topo: torch.Tensor,
    #                 areas: torch.Tensor):
    #     topo_f = outlet_topo.float()
    #     gage_counts = topo_f.sum(dim=1).clamp(min=1)  # shape (gage,)
    #     Qs_flat = Qs.squeeze(-1)  # (T, U)
    #     agg_sum = Qs_flat @ topo_f.T  # (T, G)
    #     Qout = agg_sum / gage_counts.unsqueeze(0)  # (T, G)
    #     return {'Qs_rout': Qout.unsqueeze(-1)}

    def distr_routing(self,
                      Qs: torch.Tensor,
                      distr_params_dict: dict,
                      outlet_topo: torch.Tensor,
                      areas: torch.Tensor):
        """
        :param Qs: (n_steps, n_units, 1)
        :param distr_params_dict: dict of (n_pairs, n_params)
        :param outlet_topo: (n_gages, n_units)
        :param areas: (n_units,)
        :return:
        """
        device = areas.device
        n_steps = Qs.size(0)
        max_lag = self.lenF

        # extract per-pair series
        Qs_weighted = (Qs * areas[None, :, None])  # area-weighted runoff, (n_steps, n_units, 1)
        reach_idx = (outlet_topo == 1).nonzero(as_tuple=False)
        pair_rows = reach_idx[:, 0].to(device).long()
        pair_cols = reach_idx[:, 1].to(device).long()
        Qs_pairs = Qs_weighted[:, pair_cols, :]  # (n_steps, n_pairs, 1)

        # routing via convolution
        UH = uh_gamma(
            distr_params_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
            distr_params_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
            lenF=max_lag,
        )
        if self.lag_uh:  # add a lag to the unit hydrograph
            UH = self.frac_shift1d(UH, distr_params_dict['rout_tau'])
        rf = Qs_pairs.permute([1, 2, 0]).contiguous()  # (n_pairs, 1, n_steps)
        UH = UH.permute([1, 2, 0]).contiguous()  # (n_pairs, 1, n_steps)
        Qs_lagged = uh_conv(rf, UH).squeeze(1).contiguous()  # (n_pairs, n_steps)

        # Group-sum: scatter_add_ along rows
        n_gages = int(outlet_topo.shape[0])
        Qs_rout = torch.zeros(n_gages, Qs_lagged.shape[1], device=Qs_lagged.device, dtype=Qs_lagged.dtype)
        Qs_rout.scatter_add_(0, pair_rows.view(-1, 1).expand(-1, Qs_lagged.shape[1]), Qs_lagged)  # (n_gages, n_steps)

        # Normalize by upstream area
        denom = (outlet_topo * areas[None, :]).sum(dim=1).unsqueeze(1).clamp(min=1e-6)
        Qs_rout = Qs_rout / denom
        Qs_rout = Qs_rout.T.unsqueeze(-1)  # (n_steps, n_gages, 1)

        # output
        output = {
            'Qs_rout': Qs_rout
        }
        return output

    @staticmethod
    def frac_shift1d(w, tau):
        """
        Differentiable fractional shift: return w(t - tau) by mixing k- and (k+1)-step shifts.
        For tau = k + f (0<=f<1): y[t] = (1-f)*w[t-k] + f*w[t-(k+1)]
        w:   [T,B,V]
        tau: [B,V]  (>=0 recommended)
        """
        T, B, V = w.shape
        device, dtype = w.device, w.dtype

        # Decompose tau = k + f
        tau = tau.view(1, B, V).to(dtype)
        k = torch.floor(tau)  # [1,B,V]
        f = (tau - k)  # [1,B,V]

        # Time indices 0..T-1
        t = torch.arange(T, device=device, dtype=dtype).view(T, 1, 1)  # [T,1,1]

        # Target indices for the two integer shifts
        i0 = t - k  # corresponds to shift by k
        i1 = t - (k + 1)  # corresponds to shift by k+1

        # Gather with clamp + explicit zeroing (true zero padding)
        i0c = i0.clamp(0, T - 1).long()
        i1c = i1.clamp(0, T - 1).long()

        w0 = torch.gather(w, 0, i0c)
        w1 = torch.gather(w, 0, i1c)

        mask0 = (i0 >= 0) & (i0 <= T - 1)
        mask1 = (i1 >= 0) & (i1 <= T - 1)
        w0 = w0 * mask0.to(dtype)
        w1 = w1 * mask1.to(dtype)

        # Linear blend: (1-f)*k-shift + f*(k+1)-shift
        y = (1.0 - f) * w0 + f * w1

        # Renormalize to unit mass per (B,V) -> may cause instability
        # y = y / y.sum(0).clamp_min(1e-6)
        return y  # [T,B,V]




