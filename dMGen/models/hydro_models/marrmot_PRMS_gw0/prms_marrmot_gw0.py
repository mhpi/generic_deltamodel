import math

import pandas as pd
import torch
# from functorch import vmap, jacrev, jacfwd, vjp
import torch.nn.functional as F
from models.pet_models.potet import get_potet


class prms_marrmot_gw0(torch.nn.Module):
    """
    HBV Model Pytorch version (dynamic and static param capable) from dPL_Hydro_SNTEMP @ Farshid Rahmani.
    """
    def __init__(self):
        super(prms_marrmot_gw0, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.parameters_bound = dict(tt=[-3, 5],    # tt, Temperature threshold for snowfall and melt [oC]
                                     ddf=[0, 20],    # ddf,  Degree-day factor for snowmelt [mm/oC/d]
                                     alpha=[0, 1],     # alpha, Fraction of rainfall on soil moisture going to interception [-]
                                     beta=[0, 1],    # beta, Fraction of catchment where rain goes to soil moisture [-]
                                     stor=[0, 5],    # stor, Maximum interception capcity [mm]
                                     retip=[0, 50],    # retip, Maximum impervious area storage [mm]
                                     fscn=[0, 1],    # fscn, Fraction of SCX where SCN is located [-]
                                     scx=[0, 1],    # scx, Maximum contributing fraction area to saturation excess flow [-]
                                     flz=[0.005, 0.995],    # flz, Fraction of total soil moisture that is the lower zone [-]
                                     stot=[1, 2000],    # stot, Total soil moisture storage [mm]: REMX+SMAX
                                     cgw=[0, 20],    # cgw, Constant drainage to deep groundwater [mm/d]
                                     resmax=[1, 300],    # resmax, Maximum flow routing reservoir storage (used for scaling only, there is no overflow) [mm]
                                     k1=[0, 1],    # k1, Groundwater drainage coefficient [d-1]
                                     k2=[1, 5],    # k2, Groundwater drainage non-linearity [-]
                                     k3=[0, 1],    # k3, Interflow coefficient 1 [d-1]
                                     k4=[0, 1],    # k4, Interflow coefficient 2 [mm-1 d-1]
                                     k5=[0, 1],    # k5, Baseflow coefficient [d-1]
                                     k6=[0, 1],    # k6, Groundwater sink coefficient [d-1],
                                     cgw0=[0, 20],  # cgw, Constant drainage to deep groundwater [mm/d]
                                     k7=[0,1],
                                     k8=[0,1]
                                     )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]
        self.PET_coef_bound = [
            [0.01, 1]  # PET_coef -> for converting PET to AET  ( Farshid added this param to the model)
        ]

    def source_flow_calculation(self, args, flow_out, c_NN, after_routing=True):
        varC_NN = args["var_c_NN"]
        if "DRAIN_SQKM" in varC_NN:
            area_name = "DRAIN_SQKM"
        elif "area_gages2" in varC_NN:
            area_name = "area_gages2"
        else:
            print("area of basins are not available among attributes dataset")
        area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
            flow_out["flow_sim"].shape[
                0], 1, 1)
        # flow calculation. converting mm/day to m3/sec
        if after_routing == True:
            srflow = (1000 / 86400) * area * (
                    flow_out["srflow"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out["ssflow"]).repeat(1, 1, args["nmul"])  # ras
            gwflow = (1000 / 86400) * area * (flow_out["gwflow"]).repeat(1, 1, args["nmul"])
            bas_shallow = (1000 / 86400) * area * (flow_out["bas_shallow"]).repeat(1, 1, args["nmul"])
        else:
            srflow = (1000 / 86400) * area * (
                flow_out["srflow_no_rout"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
            ssflow = (1000 / 86400) * area * (flow_out["ssflow_no_rout"]).repeat(1, 1, args["nmul"])  # ras
            gwflow = (1000 / 86400) * area * (flow_out["gwflow_no_rout"]).repeat(1, 1, args["nmul"])
            bas_shallow = (1000 / 86400) * area * (flow_out["bas_shallow_no_rout"]).repeat(1, 1, args["nmul"])

        # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        # ssflow = torch.clamp(ssflow, min=0.0)
        # gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow, bas_shallow
    
    def multi_comp_semi_static_params(
        self, params, param_no, args, interval=30, method="average"
    ):
        # seperate the piece for each interval
        nmul = args["nmul"]
        param = params[:, :, param_no * nmul : (param_no + 1) * nmul]
        no_basins, no_days = param.shape[0], param.shape[1]
        interval_no = math.floor(no_days / interval)
        remainder = no_days % interval
        param_name_list = list()
        if method == "average":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = torch.mean(
                        param[:, 0:remainder, :], 1, keepdim=True
                    ).repeat((1, remainder, 1))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = torch.mean(
                    param[
                        :,
                        ((i * interval) + remainder) : (
                            ((i + 1) * interval) + remainder
                        ),
                        :,
                    ],
                    1,
                    keepdim=True,
                ).repeat((1, interval, 1))
                param_name_list.append(param_name)
        elif method == "single_val":
            for i in range(interval_no):
                if (remainder != 0) & (i == 0):
                    param00 = (param[:, 0:1, :]).repeat((1, remainder, 1))
                    param_name_list.append(param00)
                param_name = "param" + str(i)
                param_name = (
                    param[
                        :,
                        (((i) * interval) + remainder) : (((i) * interval) + remainder)
                        + 1,
                        :,
                    ]
                ).repeat((1, interval, 1))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def param_bounds(self, params, num, args, bounds):
        nmul = args["nmul"]
        if num in args["static_params_list_prms"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
            out = out_temp.repeat(1, params.shape[1]).reshape(
                params.shape[0], params.shape[1], nmul
            )

        elif num in args["semi_static_params_list_prms"]:
            out_temp = self.multi_comp_semi_static_params(
                params,
                num,
                args,
                interval=args["interval_for_semi_static_param_prms"][
                    args["semi_static_params_list_prms"].index(num)
                ],
                method=args["method_for_semi_static_param_prms"][
                    args["semi_static_params_list_prms"].index(num)
                ],
            )
            out = (
                    out_temp * (bounds[1] - bounds[0])
                    + bounds[0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (bounds[1] - bounds[0])
                    + bounds[0]
            )
        return out

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
        # UH is a vector indicating the unit hydrograph
        # the convolved dimension will be the last dimension
        # UH convolution is
        # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
        # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
        # hence we flip the UH
        # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
        # view
        # x: [batch, var, time]
        # UH:[batch, var, uhLen]
        # batch needs to be accommodated by channels and we make use of groups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html

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

    def forward(self, x_hydro_model, c_hydro_model, params_raw, args,  warm_up=0, init=False, routing=True, conv_params_hydro=None):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["varT_hydro_model"]
        vars_c = args["varC_hydro_model"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                # paramsinit = params[:, :warm_up, :]
                # PET_coefinit = PET_coef[:, :warm_up, :]
                warm_up_model = prms_marrmot_gw0().to(args["device"])
                Q_init, snow_storage, XIN_storage, RSTOR_storage, \
                    RECHR_storage, SMAV_storage, \
                    RES_storage, GW_storage,GW_storage0 = warm_up_model(xinit, c_hydro_model, params_raw, args,
                                                            warm_up=0, init=True, routing=False,
                                                            conv_params_hydro=None)
        else:
            # snow storage
            snow_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                       device=args["device"]) + 0.001
            # interception storage
            XIN_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                      device=args["device"]) + 0.001
            # RSTOR storage
            RSTOR_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                        device=args["device"]) + 0.001
            # storage in upper soil moisture zone
            RECHR_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                        device=args["device"]) + 0.001
            # storage in lower soil moisture zone
            SMAV_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                       device=args["device"]) + 0.001
            # storage in runoff reservoir
            RES_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                      device=args["device"]) + 0.001
            # GW storage
            GW_storage = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                     device=args["device"]) + 0.001
            # GW_storage0
            GW_storage0 = torch.zeros([x_hydro_model.shape[1], nmul], dtype=torch.float32,
                                     device=args["device"]) + 0.001

        ## parameters for prms_marrmot. there are 18 parameters in it. we take all params and make the changes
        # inside the for loop
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[:, :, num, :],
                                                         bounds=self.parameters_bound[param])

        # PWT_coef , for converting PET to AET
        # PET_coef = self.change_param_range(param=PET_param,
        #                                                  bounds=self.PET_coef_bound[0])
        #################
        # inputs
        Precip = (x_hydro_model[warm_up:, :, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul))
        Tmaxf = x_hydro_model[warm_up:, :, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x_hydro_model[warm_up:, :, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2

        Ndays, Ngrid = Precip.shape[0], Precip.shape[1]

        if args["pet_module"] == "potet_hamon":
            # dayl = (x_hydro_model[warm_up:, :, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul))
            # PET = get_potet(args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=PET_coef)     # mm/day
            raise NotImplementedError
        elif args["pet_module"] == "potet_hargreaves":
            day_of_year = x_hydro_model[warm_up:, :, vars.index("dayofyear")].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index("lat")].unsqueeze(0).unsqueeze(-1).repeat(Precip.shape[0], 1, nmul)
            PET = get_potet(args=args, tmin=Tminf, tmax=Tmaxf,
                            tmean=mean_air_temp, lat=lat,
                            day_of_year=day_of_year)
            # AET = PET_coef * PET     # here PET_coef converts PET to Actual ET here
        elif args["pet_module"] == "dataset":
            # here PET_coef converts PET to Actual ET
            PET = x_hydro_model[warm_up:, :, vars.index(args["pet_dataset_name"])].unsqueeze(-1).repeat(1, 1, nmul)
        # AET = PET_coef * PET
        # initialize the Q_sim and other fluxes
        Q_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        sas_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        sro_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        bas_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        ras_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        snk_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        AET = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        inf_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        PC_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        SEP_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        GAD_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        GW0_GW_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        ea_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        qres_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])
        ###adding new GW_storage0
        bas_shallow_sim = torch.zeros(Precip.shape, dtype=torch.float32, device=args["device"])

        # do static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in args["dyn_params_list_hydro"]:  ## it is a static parameter
                params_dict[key] = params_dict_raw[key][-1, :, :]

        # Do dynamic parameters based on dydrop ratio.
        # (Drops dynamic params for some basins (based on ratio), and substitutes
        # them for a static params, which is set to the value of the param on the
        # last day of data.
        if len(args['dy_params']['marrmot_PRMS_gw0']) > 0:
            params_dict_raw_dy = dict()
            pmat = torch.ones([Ngrid, 1]) * args["dy_drop"]
            for i, key in enumerate(args['dy_params']['marrmot_PRMS_gw0']):
                drmask = torch.bernoulli(pmat).detach_().to(args["device"])
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][-1, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dy[key] = dynPar * (1 - drmask) + staPar * drmask
                
        for t in range(Nstep):
            # do dynamic parameters
            for key in params_dict_raw.keys():
                if key in args['dy_params']['marrmot_PRMS_gw0']:  ## it is a dynamic parameter
                    # params_dict[key] = params_dict_raw[key][warm_up + t, :, :]
                    # Drop dynamic parameters as static in some basins
                    params_dict[key] = params_dict_raw_dyn[key][warm_up + t, :, :]

            scn = params_dict["fscn"] * params_dict["scx"]
            remx = (1 - params_dict["flz"]) * params_dict["stot"]
            smax = params_dict["flz"] * params_dict["stot"]

            delta_t = 1  # timestep (day)
            P = Precip[t, :, :]
            Ep = PET[t, :, :]
            T = mean_air_temp[t, :, :]

            # fluxes
            flux_ps = torch.mul(P, (T <= params_dict["tt"]).type(torch.float32))
            flux_pr = torch.mul(P, (T > params_dict["tt"]).type(torch.float32))
            snow_storage = snow_storage + flux_ps
            flux_m = torch.clamp(params_dict["ddf"] * (T - params_dict["tt"]), min=0.0)
            flux_m = torch.min(flux_m, snow_storage/delta_t)
            # flux_m = torch.clamp(flux_m, min=0.0)
            snow_storage = torch.clamp(snow_storage - flux_m, min=NEARZERO)
            # snow_storage = torch.clamp(snow_storage, min=NEARZERO)  # to prevent NaN  gradient, it is set to NEARZERO

            flux_pim = flux_pr * (1 - params_dict["beta"])
            flux_psm = flux_pr * params_dict["beta"]
            flux_pby = flux_psm * (1 - params_dict["alpha"])
            flux_pin = flux_psm * params_dict["alpha"]

            XIN_storage = XIN_storage + flux_pin
            flux_ptf = XIN_storage - params_dict["stor"]
            flux_ptf = torch.clamp(flux_ptf, min=0.0)
            XIN_storage = torch.clamp(XIN_storage - flux_ptf, min=NEARZERO)
            evap_max_in = Ep * params_dict["beta"]   # only can happen in pervious area
            flux_ein = torch.min(evap_max_in, XIN_storage/delta_t)
            XIN_storage = torch.clamp(XIN_storage - flux_ein, min=NEARZERO)

            flux_mim = flux_m * (1 - params_dict["beta"])
            flux_msm = flux_m * params_dict["beta"]
            RSTOR_storage = RSTOR_storage + flux_mim + flux_pim
            flux_sas = RSTOR_storage - params_dict["retip"]
            flux_sas = torch.clamp(flux_sas, min=0.0)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_sas, min=NEARZERO)
            evap_max_im = (1 - params_dict["beta"]) * Ep
            flux_eim = torch.min(evap_max_im, RSTOR_storage / delta_t)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_eim, min=NEARZERO)

            sro_lin_ratio = scn + (params_dict["scx"] - scn) * (RECHR_storage / remx)
            sro_lin_ratio = torch.clamp(sro_lin_ratio, min=0.0, max=1.0)
            flux_sro = sro_lin_ratio * (flux_msm + flux_ptf + flux_pby)
            flux_inf = flux_msm + flux_ptf + flux_pby - flux_sro
            RECHR_storage = RECHR_storage + flux_inf
            flux_pc = RECHR_storage - remx
            flux_pc = torch.clamp(flux_pc, min=0.0)
            RECHR_storage = RECHR_storage - flux_pc
            evap_max_a = (RECHR_storage / remx) * (Ep - flux_ein - flux_eim)
            evap_max_a = torch.clamp(evap_max_a, min=0.0)
            flux_ea = torch.min(evap_max_a, RECHR_storage / delta_t)
            RECHR_storage = torch.clamp(RECHR_storage - flux_ea, min=NEARZERO)

            SMAV_storage = SMAV_storage + flux_pc
            flux_excs = SMAV_storage - smax
            flux_excs = torch.clamp(flux_excs, min=0.0)
            SMAV_storage = SMAV_storage - flux_excs
            transp = torch.where(RECHR_storage < (Ep - flux_ein - flux_eim),
                                 (SMAV_storage/smax) * (Ep - flux_ein - flux_eim - flux_ea),
                                 torch.zeros(flux_excs.shape, dtype=torch.float32, device=args["device"]))
            transp = torch.clamp(transp, min=0.0)    # in case Ep - flux_ein - flux_eim - flux_ea was negative
            SMAV_storage = torch.clamp(SMAV_storage - transp, min=NEARZERO)

            flux_sep = torch.min(params_dict["cgw"], flux_excs)
            flux_qres = torch.clamp(flux_excs - flux_sep, min=0.0)

            RES_storage = RES_storage + flux_qres
            flux_ras = params_dict["k3"] * RES_storage + params_dict["k4"] * (RES_storage ** 2)
            flux_ras = torch.min(flux_ras, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_ras, min=NEARZERO)
            # RES_excess = RES_storage - resmax[:, t, :]   # if there is still overflow, it happend in discrete version
            # RES_excess = torch.clamp(RES_excess, min=0.0)
            # flux_ras = flux_ras + RES_excess
            # RES_storage = torch.clamp(RES_storage - RES_excess, min=NEARZERO)

            flux_gad = params_dict["k1"] * ((RES_storage / params_dict['resmax']) ** params_dict["k2"])
            flux_gad = torch.min(flux_gad, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_gad, min=NEARZERO)

            ######## new GW_storage0 added
            GW_storage0 = GW_storage0 + flux_gad + flux_sep
            flux_bas_shallow = params_dict["k7"] * GW_storage0
            GW_storage0 = torch.clamp(GW_storage0 - flux_bas_shallow, min=NEARZERO)
            flux_GW0_GW = params_dict["k8"] * GW_storage0
            GW_storage0 = torch.clamp(GW_storage0 - flux_GW0_GW, min=NEARZERO)
            GW_storage = GW_storage + flux_GW0_GW
            flux_bas = params_dict["k5"] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_bas, min=NEARZERO)
            flux_snk = params_dict["k6"] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_snk, min=NEARZERO)

            Q_sim[t, :, :] = (flux_sas + flux_sro + flux_bas + flux_ras + flux_bas_shallow)
            bas_shallow_sim[t,: , :] = flux_bas_shallow
            #################


            # original code
            # GW_storage = GW_storage + flux_gad + flux_sep
            # flux_bas = params_dict["k5"] * GW_storage
            # GW_storage = torch.clamp(GW_storage - flux_bas, min=NEARZERO)
            # flux_snk = params_dict["k6"] * GW_storage
            # GW_storage = torch.clamp(GW_storage - flux_snk, min=NEARZERO)

            # Q_sim[t, :, :] = (flux_sas + flux_sro + flux_bas + flux_ras)
            sas_sim[t, :, :] = flux_sas
            sro_sim[t, :, :] = flux_sro
            bas_sim[t, :, :] = flux_bas
            ras_sim[t, :, :] = flux_ras
            snk_sim[t, :, :] = flux_snk
            AET[t, :, :] = flux_ein + flux_eim + flux_ea + transp
            inf_sim[t, :, :] = flux_inf
            PC_sim[t, :, :] = flux_pc
            SEP_sim[t, :, :] = flux_sep
            GAD_sim[t, :, :] = flux_gad
            GW0_GW_sim[t, :, :] = flux_GW0_GW
            ea_sim[t, :, :] = flux_ea
            qres_sim[t, :, :] = flux_qres
        if routing == True:
            tempa = self.change_param_range(param=conv_params_hydro[:, 0],
                                            bounds=self.conv_routing_hydro_model_bound[0])
            tempb = self.change_param_range(param=conv_params_hydro[:, 1],
                                            bounds=self.conv_routing_hydro_model_bound[1])
            routa = tempa.repeat(Ndays, 1).unsqueeze(-1)
            routb = tempb.repeat(Ndays, 1).unsqueeze(-1)
            # Q_sim_new = Q_sim.mean(-1, keepdim=True).permute(1,0,2)
            UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = Q_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])

            rf_sas = sas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsas_rout = self.UH_conv(rf_sas, UH).permute([2, 0, 1])

            rf_sro = sro_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsro_rout = self.UH_conv(rf_sro, UH).permute([2, 0, 1])

            rf_ras = ras_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qras_rout = self.UH_conv(rf_ras, UH).permute([2, 0, 1])

            rf_bas = bas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qbas_rout = self.UH_conv(rf_bas, UH).permute([2, 0, 1])

            ### adding new GW_storage0
            rf_bas_shallow = bas_shallow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qbas_shallow_rout = self.UH_conv(rf_bas_shallow, UH).permute([2, 0, 1])


        else:
            Qsrout = Q_sim.mean(-1, keepdim=True)
            Qsas_rout = sas_sim.mean(-1, keepdim=True)
            Qsro_rout = sro_sim.mean(-1, keepdim=True)
            Qbas_rout = bas_sim.mean(-1, keepdim=True)
            Qras_rout = ras_sim.mean(-1, keepdim=True)
            ### adding new GW_storage0
            Qbas_shallow_rout = bas_shallow_sim.mean(-1, keepdim=True)

        if init:  # means we are in warm up. here we just return the storages to be used as initial values
            return Qsrout, snow_storage, XIN_storage, RSTOR_storage, \
                RECHR_storage, SMAV_storage, RES_storage, GW_storage, GW_storage0
        else:
            return dict(flow_sim=Qsrout,
                        srflow=Qsas_rout + Qsro_rout,
                        ssflow=Qras_rout,
                        gwflow=Qbas_rout,
                        sink=torch.mean(snk_sim, -1).unsqueeze(-1),
                        PET_hydro=PET.mean(-1, keepdim=True),
                        AET_hydro=AET.mean(-1, keepdim=True),
                        bas_shallow=Qbas_shallow_rout,   #### adding new GW_storage0
                        flow_sim_no_rout=Q_sim.mean(-1, keepdim=True),
                        srflow_no_rout=(sas_sim + sro_sim).mean(-1, keepdim=True),
                        ssflow_no_rout=ras_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=bas_sim.mean(-1, keepdim=True),
                        bas_shallow_no_rout=bas_shallow_sim.mean(-1, keepdim=True),
                        flux_inf=inf_sim.mean(-1, keepdim=True),
                        flux_pc=PC_sim.mean(-1, keepdim=True),
                        flux_sep=SEP_sim.mean(-1, keepdim=True),
                        flux_gad=GAD_sim.mean(-1, keepdim=True),
                        flux_gw0_gw=GW0_GW_sim.mean(-1, keepdim=True),
                        flux_ea=ea_sim.mean(-1, keepdim=True),
                        flux_qres=qres_sim.mean(-1, keepdim=True),
                        )
