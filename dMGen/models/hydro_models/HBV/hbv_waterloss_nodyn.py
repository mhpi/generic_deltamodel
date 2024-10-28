import torch
from models.pet_models.potet import get_potet
import torch.nn.functional as F
import numpy as np



class HBVMulTDET_WaterLoss(torch.nn.Module):
    """
    Multi-component HBV Model *2.0* Pytorch version (dynamic and static param
    capable) adapted from Yalan Song.

    Supports optional Evapotranspiration parameter ET and capillary module.
    
    Modified from the original numpy version from Beck et al., 2020
    (http://www.gloh2o.org/hbv/), which runs the HBV-light hydrological model
    (Seibert, 2005).
    """
    def __init__(self, config):
        super(HBVMulTDET_WaterLoss, self).__init__()
        self.parameters_bound1 = dict(parBETA=[1,6],
                                      parK0=[0.05, 0.9],
                                      parBETAET=[0.3, 5]
                                      )
        self.parameters_bound2 = dict(parFC=[50, 1000],
                                      parK1=[0.01, 0.5],
                                      parK2=[0.001, 0.2],
                                      parLP=[0.2, 1],
                                      parPERC=[0, 10],
                                      parUZL=[0, 100],
                                      parTT=[-2.5, 2.5],
                                      parCFMAX=[0.5, 10],
                                      parCFR=[0, 0.1],
                                      parCWH=[0, 0.2],
                                      parC=[0,1],
                                      parTRbound=[0,20],
                                      parAc=[0, 2500]
                                      )

        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]

    def UH_gamma(self, a, b, lenF=10):
        # UH. a [time (same all time steps), batch, var]
        m = a.shape
        lenF = min(a.shape[0], lenF)
        w = torch.zeros([lenF, m[1], m[2]])
        aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
        theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
        t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
        t = t.cuda(aa.device)
        denom = (aa.lgamma().exp()) * (theta ** aa)
        mid = t ** (aa - 1)
        right = torch.exp(-t / theta)
        w = 1 / denom * mid * right
        w = w / w.sum(0)  # scale to 1 for each UH

        return w

    def UH_conv(self, x, UH, viewmode=1):
        """
        UH is a vector indicating the unit hydrograph
        the convolved dimension will be the last dimension
        UH convolution is
        Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
        conv1d does \integral(w(\tao)*x(t+\tao))d\tao
        hence we flip the UH
        https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
        view
        x: [batch, var, time]
        UH:[batch, var, uhLen]
        batch needs to be accommodated by channels and we make use of groups
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        https://pytorch.org/docs/stable/nn.functional.html
        """
        mm = x.shape;
        nb = mm[0]
        m = UH.shape[-1]
        padd = m - 1
        if viewmode == 1:
            xx = x.view([1, nb, mm[-1]])
            w = UH.view([nb, 1, m])
            groups = nb

        y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
        if padd != 0:
            y = y[:, :, 0:-padd]
        return y.view(mm)

    def source_flow_calculation(self, args, flow_out, c_NN, after_routing=True):
        varC_NN = args['var_c_nn']
        if 'DRAIN_SQKM' in varC_NN:
            area_name = 'DRAIN_SQKM'
        elif 'area_gages2' in varC_NN:
            area_name = 'area_gages2'
        else:
            print("area of basins are not available among attributes dataset")
        area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
            flow_out['flow_sim'].shape[
                0], 1, 1)
        # flow calculation. converting mm/day to m3/sec
        if after_routing == True:
            srflow = (1000 / 86400) * area * (flow_out['srflow']).repeat(1, 1, args['nmul'])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out['ssflow']).repeat(1, 1, args['nmul'])  # ras
            gwflow = (1000 / 86400) * area * (flow_out['gwflow']).repeat(1, 1, args['nmul'])
        else:
            srflow = (1000 / 86400) * area * (flow_out['srflow_no_rout']).repeat(1, 1, args['nmul'])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out['ssflow_no_rout']).repeat(1, 1, args['nmul'])  # ras
            gwflow = (1000 / 86400) * area * (flow_out['gwflow_no_rout']).repeat(1, 1, args['nmul'])
        # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        # ssflow = torch.clamp(ssflow, min=0.0)
        # gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow

    def param_bounds_2D(self, params, num, bounds, ndays, nmul):
        out_temp = (
                params[:, num * nmul: (num + 1) * nmul]
                * (bounds[1] - bounds[0])
                + bounds[0]
        )
        out = out_temp.unsqueeze(0).repeat(ndays, 1, 1).reshape(
            ndays, params.shape[0], nmul
        )
        return out

    def change_param_range(self, param, bounds):
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

def forward(self, x_hydro_model, c_hydro_model, params_raw, waterloss_params, Ai_batch , Ac_batch , idx_mat, args, muwts=None, warm_up=0,init=False, routing=False, comprout=False, conv_params_hydro=None):
        nearzero = args['nearzero']
        nmul = args['nmul']
        outstate=False

        # Initialization
        if warm_up > 0:
            raise Exception("This function currently does not support warm_up > 0.")
        else:
            # Without buff time, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            MELTWATER = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SM = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SUZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            SLZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(args['device'])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(args['device'])

        # Parameters
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[:, :, num, :],
                                                             bounds=self.parameters_bound[param])

        vars = args['observations']['var_t_hydro_model']
        vars_c = args['observations']['var_c_hydro_model']

        P = x_hydro_model[warm_up:, :, vars.index('P')]
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        mean_air_temp = x_hydro_model[warm_up:, :, vars.index('Temp')].unsqueeze(2).repeat(1, 1, nmul)

        PET = x_hydro_model[warm_up:, :, vars.index('PET')].unsqueeze(2).repeat(1, 1, nmul)

        Nstep, Ngrid = P.size()

        paraLst1 = []
        for ip in range(len(parascaLst1)): # not include routing. Scaling the parameters
            paraLst1.append( parascaLst1[ip][0] + parameters[:,ip,:]*(parascaLst1[ip][1]-parascaLst1[ip][0]) )
        paraLst2 = []
        for ip in range(len(parascaLst2)): # not include routing. Scaling the parameters
            paraLst2.append( parascaLst2[ip][0] + waterloss_parameters[:,ip,:]*(parascaLst2[ip][1]-parascaLst2[ip][0]) )




        parascaLst1 = [[1,6], [0.05,0.9],[0.3,5]]  # HBV para
            
        routscaLst = [[0,2.9], [0,6.5]]

        parascaLst2 = [[50,1000], [0.01,0.5], [0.001,0.2], [0.2,1],
                        [0,10], [0,100], [-2.5,2.5], [0.5,10], [0,0.1], [0,0.2],[0,1], [0,20], [0, 2500]]

        paraLst1 = []
        for ip in range(len(parascaLst1)): # not include routing. Scaling the parameters
            paraLst1.append( parascaLst1[ip][0] + parameters[:,ip,:]*(parascaLst1[ip][1]-parascaLst1[ip][0]) )
        paraLst2 = []
        for ip in range(len(parascaLst2)): # not include routing. Scaling the parameters
            paraLst2.append( parascaLst2[ip][0] + waterloss_parameters[:,ip,:]*(parascaLst2[ip][1]-parascaLst2[ip][0]) )

        parBETA,parK0, parBETAET = paraLst1
        parFC,  parK1, parK2, parLP, parPERC,parUZL,  parTT, parCFMAX, parCFR, parCWH,parC,parTRbound,parAc = paraLst2

        Nstep, Ngrid = P.size()

        # Initialize time series of model variables
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(x)
        
        mu2 = parAc.shape[-1]
        Ac_batch_torch = (torch.from_numpy(np.array(Ac_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # Ai_batch_torch = (torch.from_numpy(np.array(Ai_batch)).to(x)).unsqueeze(-1).repeat(1,mu2)
        # idx_matric_expand =  (torch.from_numpy(idx_matric).to(x)).unsqueeze(1).repeat(1,mu2)
        
        
        params_dict = dict()
       # parAscale_expand = parAscale.unsqueeze(0).repeat(len(Ai_batch),1,1)
        
        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]  # need to check later, seems repeating with line 52
            RAIN = torch.mul(PRECIP, (mean_air_temp[t, :, :] >= params_dict['parTT']).type(torch.float32))
            SNOW = torch.mul(PRECIP, (mean_air_temp[t, :, :] < params_dict['parTT']).type(torch.float32))

            # Snow
            SNOWPACK = SNOWPACK + SNOW
            melt = params_dict['parCFMAX'] * (mean_air_temp[t, :, :] - params_dict['parTT'])
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = parCFR * parCFMAX * (parTT - Tm[t, :, :])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = torch.clamp(MELTWATER - refreezing, min=nearzero)
            tosoil = MELTWATER - (params_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = torch.clamp(MELTWATER - tosoil, min=nearzero)

            # Soil and evaporation
            soil_wetness = (SM / params_dict['parFC']) ** params_dict['parBETA']
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge
            excess = SM - params_dict['parFC']
            # excess[excess < 0.0] = 0.0
            excess = torch.clamp(excess, min=0.0)
            SM = torch.clamp(SM - excess, min=nearzero)

            # NOTE: Different from HBVmul. Add an ET shape parameter parBETAET. This param can be static or dynamic
            evapfactor = (SM / (params_dict['parLP'] * params_dict['parFC'])) ** params_dict['parBETAET']
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PET[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=nearzero)  # SM can not be zero for gradient tracking.
            capillary = torch.min(SLZ, params_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / params_dict['parFC'], max=1.0)))

            SM = torch.clamp(SM + capillary, min=nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=nearzero)

            # Groundwater boxes
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, params_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = params_dict['parK0'] * torch.clamp(SUZ - params_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = params_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC

            ## Added for v2.    
            regional_flow = torch.clamp((Ac_batch_torch-parAc)/1000,min = -1, max = 1) * parTRbound*(Ac_batch_torch<2500)+\
            torch.exp(torch.clamp(-(Ac_batch_torch-2500)/50, min = -10.0,max = 0.0))* parTRbound*(Ac_batch_torch>=2500)
            #regional_flow = (RT.unsqueeze(-1).repeat(1,1,Ngrid) *idx_matric_expand).sum(0)
            #regional_flow = torch.clamp(regional_flow0, max = 0.0).unsqueeze(-1).repeat(1,mu)
            
            SLZ = torch.clamp(SLZ + regional_flow, min=0.0)
            Q2 = params_dict['parK2'] * SLZ
            SLZ = SLZ - Q2
            Qsimmu[t, :, :] = Q0 + Q1 + Q2

        # Get the initial average
        if muwts is None:
            Qsimave = Qsimmu.mean(-1)
        else:
            Qsimave = (Qsimmu * muwts).sum(-1)

        if routing is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim = Qsimmu.view(Nstep, Ngrid * nmul)
            else:
                # average the components, then do routing
                Qsim = Qsimave

            tempa = self.change_param_range(param=conv_params_hydro[:, 0],
                                            bounds=self.conv_routing_hydro_model_bound[0])
            tempb = self.change_param_range(param=conv_params_hydro[:, 1],
                                            bounds=self.conv_routing_hydro_model_bound[1])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])   # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])

            if comprout is True: # Qs is [time, [gage*mult], var] now
                Qstemp = Qsrout.view(Nstep, Ngrid, mu)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else: # no routing, output the initial average simulations

            Qs = torch.unsqueeze(Qsimave, -1) # add a dimension

        if outstate is True:
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            return Qs # total streamflow