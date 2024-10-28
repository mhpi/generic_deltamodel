import torch
import torch.nn.functional as F
from models.pet_models.potet import get_potet


class SACSMAMul(torch.nn.Module):
    """
    SAC-SMA Model Pytorch version (dynamic and static param capable) from dPL_Hydro_SNTEMP @ Farshid Rahmani.
    
    TODO: Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD().
    We suggest you read the class HBVMul() with original static parameters first.
    """
    def __init__(self):
        """Initiate a HBV instance"""
        super(SACSMAMul, self).__init__()
        self.parameters_bound = dict(pctim=[0.0, 1.0],
                                     smax=[1, 2000],
                                     f1=[0.005, 0.995],
                                     f2=[0.005, 0.995],
                                     kuz=[0.0, 1],
                                     rexp=[0.0, 7],
                                     f3=[0.005, 0.995],
                                     f4=[0.005, 0.995],
                                     pfree=[0, 1],
                                     klzp=[0, 1],
                                     klzs=[0, 1])
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]  # routing parameter b
        ]

    def split_1(self, p1, In):
        """
        Split flow (returns flux [mm/d])

        :param p1: fraction of flux to be diverted [-]
        :param In: incoming flux [mm/d]
        :return: divided flux
        """
        out = p1 * In
        return out

    def soilmoisture_1(self, S1, S1max, S2, S2max):
        """
        Water rebalance to equal relative storage (2 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :return: rebalanced water storage
        """
        mask = S1/S1max < S2/S2max
        mask = mask.type(torch.cuda.FloatTensor)
        out = ((S2 * S1max - S1 * S2max) / (S1max + S2max)) * mask
        return out

    def evap_7(self, S, Smax, Ep, dt):
        """
        Evaporation scaled by relative storage

        :param S: current storage [mm]
        :param Smax: maximum contributing storage [mm]
        :param Ep: potential evapotranspiration rate [mm/d]
        :param dt: time step size [d]
        :return: evaporation [mm]
        """
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def saturation_1(self, In, S, Smax):
        """
        Saturation excess from a store that has reached maximum capacity

        :param In: incoming flux [mm/d]
        :param S: current storage [mm]
        :param Smax: maximum storage [mm]
        :param args: smoothing variables (optional)
        :return: saturation excess
        """
        mask = S >= Smax
        mask = mask.type(torch.cuda.FloatTensor)
        out = In * mask

        return out

    def interflow_5(self, p1, S):
        """
        Linear interflow

        :param p1: time coefficient [d-1]
        :param S: current storage [mm]
        :return: interflow output
        """
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        """
        Evaporation at the potential rate

        :param S: current storage [mm]
        :param Ep: potential evaporation rate [mm/d]
        :param dt: time step size
        :return: evaporation output
        """
        out = torch.min(S / dt, Ep)
        return out

    def percolation_4(self, p1, p2, p3, p4, p5, S, Smax, dt):
        """
        Demand-based percolation scaled by available moisture

        :param p1: base percolation rate [mm/d]
        :param p2: percolation rate increase due to moisture deficiencies [mm/d]
        :param p3: non-linearity parameter [-]
        :param p4: summed deficiency across all model stores [mm]
        :param p5: summed capacity of model stores [mm]
        :param S: current storage in the supplying store [mm]
        :param Smax: maximum storage in the supplying store [mm]
        :param dt: time step size [d]
        :return: percolation output
        """
        # Prevent negative S values and ensure non-negative percolation demands
        S_rel = torch.max(torch.tensor(1e-8).cuda(), S / Smax)

        percolation_demand = p1 * (torch.tensor(1.0).cuda() + p2 * (p4 / p5) ** (torch.tensor(1.0).cuda() + p3))
        out = torch.max(torch.tensor(1e-8).cuda(), torch.min(S / dt, S_rel * percolation_demand))
        return out

    def soilmoisture_2(self, S1, S1max, S2, S2max, S3, S3max):
        """
        Water rebalance to equal relative storage (3 stores)

        :param S1: current storage in S1 [mm]
        :param S1max: maximum storage in S1 [mm]
        :param S2: current storage in S2 [mm]
        :param S2max: maximum storage in S2 [mm]
        :param S3: current storage in S3 [mm]
        :param S3max: maximum storage in S3 [mm]
        :return: rebalanced water storage
        """
        part1 = S2 * (S1 * (S2max + S3max) + S1max * (S2 + S3)) / ((S2max + S3max) * (S1max + S2max + S3max))
        mask = S1 / S1max < (S2 + S3) / (S2max + S3max)
        mask = mask.type(torch.cuda.FloatTensor)
        out = part1 * mask
        return out

    def baseflow_1(self,K,S):
        return K * S



    def deficitBasedDistribution_pytorch(self, S1, S1max, S2, S2max):
        # Calculate relative deficits
        rd1 = (S1max-S1 ) / S1max
        rd2 = (S2max-S2 ) / S2max

        # Calculate fractional split
        total_rd = rd1 + rd2
        mask = total_rd != torch.tensor(0.0).cuda()
        mask = mask.type(torch.cuda.FloatTensor)
        total_max = S1max + S2max
        f1 = rd1 / total_rd * mask + S1max / total_max*(torch.tensor(1.0)-mask)

        return f1

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


    def change_param_range(self, param, bounds):
        out = param * (bounds[1] - bounds[0]) + bounds[0]
        return out

    def source_flow_calculation(self, args, flow_out, c_NN):
        varC_NN = args["varC_NN"]
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
        srflow = (1000 / 86400) * area * (
                flow_out["srflow"]).repeat(1, 1, args["nmul"])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * (flow_out["ssflow"]).repeat(1, 1, args["nmul"])  # ras
        gwflow = (1000 / 86400) * area * (flow_out["gwflow"]).repeat(1, 1, args["nmul"])
        # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
        # ssflow = torch.clamp(ssflow, min=0.0)
        # gwflow = torch.clamp(gwflow, min=0.0)
        return srflow, ssflow, gwflow

    def forward(self, x_hydro_model, c_hydro_model, params_raw, args, muwts=None, warm_up=0, init=False, routing=False, comprout=False, conv_params_hydro=None):
        nmul = args["nmul"]
        # HBV(P, ETpot, T, parameters)
        #
        # Runs the HBV-light hydrological model (Seibert, 2005). NaN values have to be
        # removed from the inputs.

        PRECS = 1e-5

        # Initialization
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                warm_up_model = SACSMAMul().to(args["device"])
                Qsrout, UZTW_storage, UZFW_storage, LZTW_storage, \
                LZFWP_storage, LZFWS_storage = warm_up_model(xinit, c_hydro_model, params_raw, args,
                                                                      muwts=None, warm_up=0, init=True, routing=False,
                                                                      comprout=False, conv_params_hydro=None)
        else:
            # Without buff time, initialize state variables with zeros
            Ngrid = x_hydro_model.shape[1]
            UZTW_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            UZFW_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZTW_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZFWP_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            LZFWS_storage = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.0001).to(args["device"])
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        ## parameters for prms_marrmot. there are 18 parameters in it. we take all params and make the changes
        # inside the for loop
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = self.change_param_range(param=params_raw[:, :, num, :],
                                                             bounds=self.parameters_bound[param])

        vars = args["varT_hydro_model"]
        vars_c = args["varC_hydro_model"]
        P = x_hydro_model[warm_up:, :, vars.index("prcp(mm/day)")]
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        Tmaxf = x_hydro_model[warm_up:, :, vars.index("tmax(C)")].unsqueeze(2).repeat(1, 1, nmul)
        Tminf = x_hydro_model[warm_up:, :, vars.index("tmin(C)")].unsqueeze(2).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2

        if args["pet_module"] == "potet_hamon":
            # # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=args["nmul"])
            # PET = get_potet(
            #     args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=PET_coef
            # )  # mm/day
            raise NotImplementedError
        elif args["pet_module"] == "potet_hargreaves":
            day_of_year = x_hydro_model[warm_up:, :, vars.index("dayofyear")].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index("lat")].unsqueeze(0).unsqueeze(-1).repeat(day_of_year.shape[0], 1, nmul)
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                   nmul=args["nmul"])

            PET = get_potet(
                args=args, tmin=Tminf, tmax=Tmaxf,
                tmean=mean_air_temp, lat=lat,
                day_of_year=day_of_year
            )
            # AET = PET_coef * PET     # here PET_coef converts PET to Actual ET here
        elif args["pet_module"] == "dataset":
            # PET_coef = self.param_bounds_2D(PET_coef, 0, bounds=[0.01, 1.0], ndays=No_days,
            #                                 nmul=args["nmul"])
            # here PET_coef converts PET to Actual ET
            PET = x_hydro_model[warm_up:, :, vars.index(args["pet_dataset_name"])].unsqueeze(-1).repeat(1, 1, nmul)
            # AET = PET_coef * PET
        Q_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        srflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        ssflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        gwflow_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])
        AET = torch.zeros(Pm.shape, dtype=torch.float32, device=args["device"])

        # do static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in args["dyn_params_list_hydro"]:  ## it is a static parameter
                params_dict[key] = params_dict_raw[key][-1, :, :]

        Nstep, Ngrid = P.size()

        # Do dynamic parameters based on dydrop ratio.
        # (Drops dynamic params for some basins (based on ratio), and substitutes
        # them for a static params, which is set to the value of the param on the
        # last day of data.
        if len(args['dy_params']['SACSMA']) > 0:
            params_dict_raw_dy = dict()
            pmat = torch.ones([Ngrid, 1]) * args["dy_drop"]
            for i, key in enumerate(args['dy_params']['SACSMA']):
                drmask = torch.bernoulli(pmat).detach_().to(args["device"])
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][-1, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dy[key] = dynPar * (1 - drmask) + staPar * drmask
                
        for t in range(Nstep):
            # do dynamic parameters
            for key in params_dict_raw.keys():
                if key in args['dy_params']['SACSMA']:  ## it is a dynamic parameter
                    # params_dict[key] = params_dict_raw[key][warm_up + t, :, :]
                    # Drop dynamic parameters as static in some basins
                    params_dict[key] = params_dict_raw_dyn[key][warm_up + t, :, :]

            uztwm = params_dict["f1"] * params_dict["smax"]
            uzfwm = torch.clamp(params_dict["f2"] * (params_dict["smax"] - uztwm), min=0.005 / 4)
            lztwm = torch.clamp(params_dict["f3"] * (params_dict["smax"] - uztwm - uzfwm), min=0.005 / 4)
            lzfwpm = torch.clamp(params_dict["f4"] * (params_dict["smax"] - uztwm - uzfwm - lztwm), min=0.005 / 4)
            lzfwsm = torch.clamp((1 - params_dict["f4"]) * (params_dict["smax"] - uztwm - uzfwm - lztwm), min=0.005 / 4)
            pbase = lzfwpm * params_dict["klzp"] + lzfwsm * params_dict["klzs"]
            zperc = torch.clamp((lztwm + lzfwsm * (1 - params_dict["klzs"])) / (lzfwsm * params_dict["klzs"] +
                                                                                lzfwpm * params_dict["klzp"]) +
                                (lzfwpm * (1 - params_dict["klzp"])) / (lzfwsm * params_dict["klzs"] +
                                                                        lzfwpm * params_dict["klzp"]), max=100000.0)

            # Separate precipitation into liquid and solid components
            PRECIP = Pm[t, :, :]
            Ep = PET[t, :, :]
            flux_qdir = self.split_1(params_dict["pctim"], PRECIP)
            flux_peff = self.split_1(1 - params_dict["pctim"], PRECIP)
            UZTW_storage = UZTW_storage + flux_peff
            flux_Ru = torch.where((UZTW_storage / uztwm) < (UZFW_storage / uzfwm),
                                  (uztwm * UZFW_storage - uzfwm * UZTW_storage) / (uztwm + uzfwm),
                                  torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Ru = torch.min(flux_Ru, UZFW_storage)
            UZFW_storage = torch.clamp(UZFW_storage - flux_Ru, min=0.0001)
            UZTW_storage = UZTW_storage + flux_Ru
            flux_Twexu = torch.clamp(UZTW_storage - uztwm, min=0.0)
            UZTW_storage = torch.clamp(UZTW_storage - flux_Twexu, min=0.0001)
            # flux_Twexu = torch.clamp(flux_Twexu, min=0.0)
            flux_Euztw = Ep * UZTW_storage / (uztwm + 0.0001)  # to avoid nan values, we add 0.001
            flux_Euztw = torch.min(flux_Euztw, UZTW_storage)
            UZTW_storage = torch.clamp(UZTW_storage - flux_Euztw, min=0.0001)

            UZFW_storage = UZFW_storage + flux_Twexu
            flux_Qsur = torch.clamp(UZFW_storage - uzfwm, min=0.0)
            UZFW_storage = torch.clamp(UZFW_storage - flux_Qsur, min=0.0001)
            flux_Qint = params_dict["kuz"] * UZFW_storage
            UZFW_storage = torch.clamp(UZFW_storage - flux_Qint, min=0.0001)
            LZ_deficiency = (lztwm - LZTW_storage) + (lzfwpm - LZFWP_storage) + (lzfwsm - LZFWS_storage)
            LZ_deficiency = torch.clamp(LZ_deficiency, min=0.0)    # just to make sure there is no negative values
            LZ_capacity = lztwm + lzfwsm + lzfwpm
            Pc_demand = pbase * (1 + (zperc * ((LZ_deficiency / (LZ_capacity+0.0001)) ** (1 + params_dict["rexp"]))))
            flux_Pc = Pc_demand * UZFW_storage / uzfwm
            flux_Pc = torch.min(flux_Pc, UZFW_storage)
            UZFW_storage = torch.clamp(UZFW_storage - flux_Pc, min=0.0001)
            flux_Euzfw = torch.clamp(Ep - flux_Euztw, min=0.0)
            flux_Euzfw = torch.min(flux_Euzfw, UZFW_storage)
            UZFW_storage = torch.clamp(UZFW_storage - flux_Euzfw, min=0.0001)


            Rl_nominator = -LZTW_storage * (lzfwpm + lzfwsm) + lztwm * (LZFWP_storage + LZFWS_storage)
            Rl_denominator = (lzfwpm + lzfwsm) * (lztwm + lzfwpm + lzfwsm)
            flux_Rlp = torch.where((LZTW_storage / lztwm) < ((LZFWP_storage + LZFWS_storage) / (lzfwpm + lzfwsm)),
                                   lzfwpm * (Rl_nominator / Rl_denominator),
                                   torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Rlp = torch.min(flux_Rlp, LZFWP_storage)
            LZFWP_storage = torch.clamp(LZFWP_storage - flux_Rlp, min=0.0001)
            LZTW_storage = LZTW_storage + flux_Rlp

            flux_Rls = torch.where((LZTW_storage / lztwm) < ((LZFWP_storage + LZFWS_storage) / (lzfwpm + lzfwsm)),
                                   lzfwsm * (Rl_nominator / Rl_denominator),
                                   torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Rls = torch.min(flux_Rls, LZFWS_storage)
            LZFWS_storage = torch.clamp(LZFWS_storage - flux_Rls, min=0.0001)
            LZTW_storage = LZTW_storage + flux_Rls


            flux_Pctw = (1 - params_dict["pfree"]) * flux_Pc
            flux_Pcfw = params_dict["pfree"] * flux_Pc
            LZTW_storage = LZTW_storage + flux_Pctw

            flux_twexl = torch.clamp(LZTW_storage - lztwm, min=0.0)
            LZTW_storage = torch.clamp(LZTW_storage - flux_twexl, min=0.0001)
            flux_Elztw = torch.where((LZTW_storage > 0.0) & (Ep > flux_Euztw + flux_Euzfw),
                                     (Ep - flux_Euztw - flux_Euzfw) * (LZTW_storage / (uztwm + lztwm)),
                                     torch.zeros(flux_qdir.shape, dtype=torch.float32, device=args["device"]))
            flux_Elztw = torch.min(flux_Elztw, LZTW_storage)
            LZTW_storage = torch.clamp(LZTW_storage - flux_Elztw, min=0.0001)

            flux_Pcfwp = ((lzfwpm - LZFWP_storage) / (lzfwpm * (((lzfwpm - LZFWP_storage) / lzfwpm) + (
                        (lzfwsm - LZFWS_storage) / lzfwsm) + 0.0001))) * flux_Pcfw
            flux_twexlp = ((lzfwpm - LZFWP_storage) / (lzfwpm * (((lzfwpm - LZFWP_storage) / lzfwpm) + (
                        (lzfwsm - LZFWS_storage) / lzfwsm) + 0.0001))) * flux_twexl
            LZFWP_storage = LZFWP_storage + flux_Pcfwp + flux_twexlp
            flux_Qbfp = params_dict["klzp"] * LZFWP_storage
            LZFWP_storage = torch.clamp(LZFWP_storage - flux_Qbfp, min=0.0001)
            extra_LZFWP = torch.clamp(LZFWP_storage - lzfwpm, min=0.0)
            LZFWP_storage = torch.clamp(LZFWP_storage - extra_LZFWP,
                                        min=0.0001)  # I added this to make the storage not to exceed the max
            # just to make sure LZFWP_storage is always smaller than lzfwpm
            LZFWP_storage = torch.where(LZFWP_storage >= lzfwpm,
                                        lzfwpm - 0.0001,
                                        LZFWP_storage)
            flux_Qbfp = flux_Qbfp + extra_LZFWP
            # This line needs to be rechecked with the documents (flux_Pcfws + flux_Pcfwp != flux_Pcfw
            # flux_Pcfws = ((lzfwsm - LZFWS_storage) / (
            #             lzfwsm * ((lzfwsm - LZFWP_storage) / lzfwpm) + ((lzfwsm - LZFWS_storage) / lzfwsm))) * flux_Pcfw
            flux_Pcfws = torch.clamp(flux_Pcfw - flux_Pcfwp, min=0.0)
            flux_twexls = ((lzfwsm - LZFWS_storage) / (lzfwsm * (((lzfwpm - LZFWP_storage) / lzfwpm) + (
                        (lzfwsm - LZFWS_storage) / lzfwsm) + 0.0001))) * flux_twexl
            LZFWS_storage = LZFWS_storage + flux_Pcfws + flux_twexls

            flux_Qbfs = params_dict["klzs"] * LZFWS_storage
            LZFWS_storage = torch.clamp(LZFWS_storage - flux_Qbfs, min=0.0001)
            extra_LZFWS = torch.clamp(LZFWS_storage - lzfwsm, min=0.0)
            LZFWS_storage = torch.clamp(LZFWS_storage - extra_LZFWS,
                                        min=0.0001)  # I added this to make the storage not to exceed the max
            # just to make sure LZFWS_storage is always smaller than lzfwsm
            LZFWS_storage = torch.where(LZFWS_storage >= lzfwsm,
                                        lzfwsm - 0.0001,
                                        LZFWS_storage)
            flux_Qbfs = flux_Qbfs + extra_LZFWS
            Q_sim[t, :, :] = flux_qdir + flux_Qsur + flux_Qint + flux_Qbfp + flux_Qbfs
            srflow_sim[t, :, :] = flux_qdir + flux_Qsur
            ssflow_sim[t, :, :] = flux_Qint
            gwflow_sim[t, :, :] = flux_Qbfp + flux_Qbfs
            AET[t, :, :] = flux_Euztw + flux_Euzfw + flux_Elztw

        if routing == True:
            tempa = self.change_param_range(param=conv_params_hydro[:, 0],
                                            bounds=self.conv_routing_hydro_model_bound[0])
            tempb = self.change_param_range(param=conv_params_hydro[:, 1],
                                            bounds=self.conv_routing_hydro_model_bound[1])
            routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            # Q_sim_new = Q_sim.mean(-1, keepdim=True).permute(1,0,2)
            UH = self.UH_gamma(routa, routb, lenF=15)  # lenF: folter
            rf = Q_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = self.UH_conv(rf, UH).permute([2, 0, 1])

            rf_srflow = srflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            srflow_rout = self.UH_conv(rf_srflow, UH).permute([2, 0, 1])

            rf_ssflow = ssflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            ssflow_rout = self.UH_conv(rf_ssflow, UH).permute([2, 0, 1])

            rf_gwflow = gwflow_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            gwflow_rout = self.UH_conv(rf_gwflow, UH).permute([2, 0, 1])
        else:
            Qsrout = Q_sim.mean(-1, keepdim=True)
            srflow_rout = srflow_sim.mean(-1, keepdim=True)
            ssflow_rout = ssflow_sim.mean(-1, keepdim=True)
            gwflow_rout = gwflow_sim.mean(-1, keepdim=True)

        if init:  # means we are in warm up. here we just return the storages to be used as initial values
            return Qsrout, UZTW_storage, UZFW_storage, LZTW_storage, \
                LZFWP_storage, LZFWS_storage

        else:
            return dict(flow_sim=Qsrout,
                        srflow=srflow_rout,
                        ssflow=ssflow_rout,
                        gwflow=gwflow_rout,
                        PET_hydro=PET.mean(-1, keepdim=True),
                        AET_hydro=AET.mean(-1, keepdim=True),
                        flow_sim_no_rout=Q_sim.mean(-1, keepdim=True),
                        srflow_no_rout=srflow_sim.mean(-1, keepdim=True),
                        ssflow_no_rout=ssflow_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=gwflow_sim.mean(-1, keepdim=True),
                        )
























