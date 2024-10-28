import torch
from models.pet_models.potet import get_potet
from core.calc.hydrograph import UH_gamma, UH_conv
import torch.nn.functional as F
from core.utils.utils import change_param_range
import numpy as np



class HBVMulTDET_WaterLoss(torch.nn.Module):
    """
    Multi-component HBV Model *2.0* Pytorch version (dynamic and static param
    capable) adapted from Yalan Song.

    Supports optional Evapotranspiration parameter ET and capillary rise mod.
    
    Modified from the original numpy version from Beck et al., 2020
    (http://www.gloh2o.org/hbv/), which runs the HBV-light hydrological model
    (Seibert, 2005).
    """
    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_WaterLoss, self).__init__()
        # TODO: recombine these to parameter sets and then dynamically manage
        # which parameters are added to a separated dynamic parameter set.
        # HBV dynamic vs static water loss terms.
        self.hbv_param_bound = dict(parBETA=[1.0, 6.0],
                                    parK0=[0.05, 0.9],
                                    parBETAET=[0.3, 5],
                                    )
        self.wl_param_bound = dict(parFC=[50, 1000],
                                   parK1=[0.01, 0.5],
                                   parK2=[0.001, 0.2],
                                   parLP=[0.2, 1],
                                   parPERC=[0, 10],
                                   parUZL=[0, 100],
                                   parTT=[-2.5, 2.5],
                                   parCFMAX=[0.5, 10],
                                   parCFR=[0, 0.1],
                                   parCWH=[0, 0.2],
                                   parC=[0, 1],
                                   parTRbound=[0,20],
                                   parAc=[0, 2500]
                                   )
        # All parameters
        self.parameters_bound = {**self.hbv_param_bound, **self.wl_param_bound}

        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]

    def forward(self, x_hydro_model, params_raw, waterloss_params_raw, ai_batch,
                ac_batch, idx_matrix, config, muwts=None, static_idx=-1, warm_up=0,
                init=False, routing=False, comprout=False, conv_params_hydro=None):
        nearzero = config['nearzero']
        nmul = config['nmul']

        # Initialization to warm-up states
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                initmodel = HBVMulTDET_WaterLoss().to(config['device'])
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(xinit,
                                                                      params_raw,
                                                                      waterloss_params_raw,
                                                                      ai_batch,
                                                                      ac_batch,
                                                                      idx_matrix,
                                                                      config,
                                                                      muwts=None,
                                                                      static_idx=warm_up - 1,
                                                                      warm_up=0,
                                                                      init=True,
                                                                      routing=False,
                                                                      comprout=False,
                                                                      conv_params_hydro=None)
        else:
            # Without warm-up, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            MELTWATER = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SM = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SUZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SLZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(config['device'])

        # HBV and water loss parameters
        params_dict_raw = dict()
        wl_params_dict_raw = dict()
        for num, param in enumerate(self.hbv_param_bound.keys()):
            params_dict_raw[param] = change_param_range(param=params_raw[:, :, num, :],
                                                             bounds=self.hbv_param_bound[param])
        for num, param in enumerate(self.wl_param_bound.keys()):
            wl_params_dict_raw[param] = change_param_range(
                param=waterloss_params_raw[:, num, :],
                bounds=self.wl_param_bound[param]
            )

        vars = config['observations']['var_t_nn']

        # Forcings
        P = x_hydro_model[warm_up:, :, vars.index('P')]  # Precipitation
        T = x_hydro_model[warm_up:, :, vars.index('Temp')]  # Mean air temp
        PET = x_hydro_model[warm_up:, :, vars.index('PET')]  # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, nmul)
        PETm = PET.unsqueeze(2).repeat(1, 1, nmul)
        

        # mu2 = wl_params_dict_raw['parAc'].shape[-1]
        ac_batch_torch = (torch.from_numpy(np.array(ac_batch)).to(config['device']))
        ac_batchm = ac_batch_torch.unsqueeze(-1).repeat(1,config['ann_opt']['nmul'])

        if init:
            # Run all static for warmup.
            dy_params = []

            Nstep, Ngrid = P.size()
            # Initialize time series of model variable.
            Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(config['device'])

        else:
            # Get list of dynamic parameters.
            # dy_params = config['dy_params']['HBV_waterLoss']
            dy_params = list(self.hbv_param_bound.keys())

            Nstep, _ = P.size()
            Ngrid = idx_matrix.shape[-1]

            # Initialize time series of model variables.
            Qsimmu = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])
            ET_sim = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])
            SWE_sim = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])
            
            # Output the box components of Q.
            Q0_sim = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])
            Q1_sim = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])
            Q2_sim = (torch.zeros((Nstep,Ngrid), dtype=torch.float32) + 0.001).to(config['device'])

            AET = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            recharge_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            excs_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            evapfactor_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            tosoil_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            PERC_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            SWE_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])
            capillary_sim = (torch.zeros(Nstep,Ngrid, dtype=torch.float32)).to(config['device'])

            ai_batch_torch = torch.from_numpy(np.array(ai_batch)).to(config['device'])
            idx_matrix_torch = torch.from_numpy(np.array(idx_matrix)).to(config['device'])

        # Init static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in dy_params: # and len(params_raw.shape) > 2:
                params_dict[key] = params_dict_raw[key][static_idx, :, :]

        # Add static water loss parameters
        params_dict.update(wl_params_dict_raw)

        for t in range(Nstep):
            # Get dynamic parameter values per timestep.
            for key in dy_params:
                params_dict[key] = params_dict_raw[key][warm_up + t, :, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= params_dict['parTT']).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < params_dict['parTT']).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = params_dict['parCFMAX'] * (Tm[t, :, :] - params_dict['parTT'])
            melt = torch.clamp(melt, min=0.0)
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = params_dict['parCFR'] * params_dict['parCFMAX'] * (
                params_dict['parTT'] - Tm[t, :, :]
                )
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (params_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation -------------------------------
            soil_wetness = (SM / params_dict['parFC']) ** params_dict['parBETA']
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge

            excess = SM - params_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
            evapfactor = (SM / (params_dict['parLP'] * params_dict['parFC'])) ** params_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=nearzero)  # SM != 0 for grad tracking.

            # Capillary rise mod -------------------------------
            capillary = torch.min(SLZ, params_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / params_dict['parFC'], max=1.0)))

            SM = torch.clamp(SM + capillary, min=nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=nearzero)

            # Groundwater boxes -------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, params_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = params_dict['parK0'] * torch.clamp(SUZ - params_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = params_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            
            regional_flow = torch.clamp((ac_batchm - params_dict['parAc']) / 1000, min = -1, max = 1)
            regional_flow = regional_flow * params_dict['parTRbound'] * (ac_batchm<2500) + \
            torch.exp(torch.clamp(-(ac_batchm-2500)/50, min = -10.0,max = 0.0)) * params_dict['parTRbound'] * (ac_batchm>=2500)

            SLZ = torch.clamp(SLZ + regional_flow, min=0.0)
            Q2 = params_dict['parK2'] * SLZ
            SLZ = SLZ - Q2

            if init:
                Qsimmu[t, :] = Q0 + Q1 + Q2

                # Get the overall average 
                # or weighted average using learned weights.
                if muwts is None:
                    Qsimavg = Qsimmu.mean(-1)
                else:
                    Qsimavg = (Qsimmu * muwts).sum(-1)
            else:
                regional_flow_out = torch.max(regional_flow, -SLZ)

                Qsimmu[t, :] = (((Q0 + Q1 + Q2).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                Q0_sim[t, :] = (((regional_flow_out).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                Q1_sim[t, :] = (((Q1).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                Q2_sim[t, :] = (((Q2).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                AET[t, :] = (((ETact).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                SWE_sim[t, :] = (((SNOWPACK).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                capillary_sim[t, :] = (((capillary).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)

                recharge_sim[t, :] = (((recharge).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                excs_sim[t, :] = (((excess).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                evapfactor_sim[t, :] = (((evapfactor).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                tosoil_sim[t, :] = (((tosoil).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)
                PERC_sim[t, :] = (((PERC).mean(-1) * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) * idx_matrix_torch).sum(0)

        # Run routing
        if routing:
            # Routing for all components or just the average.
            if init:
                if comprout:
                    # All components; reshape to [time, gages * num models]
                    Qsim = Qsimmu.view(Nstep, Ngrid * nmul)
                else:
                    # Average, then do routing.
                    Qsim = Qsimavg
            
                rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            else:
                rf = torch.unsqueeze(Qsimmu, -1).permute([1, 2, 0])  # [gages,vars,time]
                
            # Scale routing params
            temp_a0 = change_param_range(
                param=conv_params_hydro[:, 0],
                bounds=self.conv_routing_hydro_model_bound[0]
            )
            temp_b0 = change_param_range(
                param=conv_params_hydro[:, 1],
                bounds=self.conv_routing_hydro_model_bound[1]
            )
            temp_a = ((temp_a0 * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) *\
                      idx_matrix_torch).sum(0) 
            temp_b = ((temp_b0 * ai_batch_torch).unsqueeze(-1).repeat(1,Ngrid) *\
                      idx_matrix_torch).sum(0) 

            rout_a = temp_a.repeat(Nstep, 1).unsqueeze(-1)
            rout_b = temp_b.repeat(Nstep, 1).unsqueeze(-1)

            UH = UH_gamma(rout_a, rout_b, lenF=15)  # lenF: folter
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = UH_conv(rf, UH.float()).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            # rf_Q0 = Q0_sim.mean(-1, keepdim=True)
            # Q0_rout = UH_conv(rf_Q0, UH).permute([2, 0, 1])
            # rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            # Q1_rout = UH_conv(rf_Q1, UH).permute([2, 0, 1])
            # rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            # Q2_rout = UH_conv(rf_Q2, UH).permute([2, 0, 1])

            if comprout: 
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(Nstep, Ngrid, nmul)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: 
            # No routing, only output the average of all model sims.
            if init:
                Qs = torch.unsqueeze(Qsimavg, -1)
            else:
                Qs = torch.unsqueeze(Qsimmu, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if init:
            # Only return model states for warmup.
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Return all sim results.
            return dict(flow_sim=Qs,
                        # srflow=Q0_rout,
                        # ssflow=Q1_rout,
                        # gwflow=Q2_rout,
                        AET_hydro=AET.mean(-1, keepdim=True),
                        PET_hydro=PETm.mean(-1, keepdim=True),
                        SWE=SWE_sim.mean(-1, keepdim=True),
                        # flow_sim_no_rout=Qsim.unsqueeze(dim=2),
                        srflow_no_rout=Q0_sim.mean(-1, keepdim=True),
                        ssflow_no_rout=Q1_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=Q2_sim.mean(-1, keepdim=True),
                        recharge=recharge_sim.mean(-1, keepdim=True),
                        excs=excs_sim.mean(-1, keepdim=True),
                        evapfactor=evapfactor_sim.mean(-1, keepdim=True),
                        tosoil=tosoil_sim.mean(-1, keepdim=True),
                        percolation=PERC_sim.mean(-1, keepdim=True),
                        capillary=capillary_sim.mean(-1, keepdim=True)
                        )
