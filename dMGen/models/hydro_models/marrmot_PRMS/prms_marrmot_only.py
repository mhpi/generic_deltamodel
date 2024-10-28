import pandas as pd
import torch
# from functorch import vmap, jacrev, jacfwd, vjp
import torch.nn.functional as F
from models.pet_models.potet import get_potet


class prms_marrmot_changed(torch.nn.Module):
    def __init__(self, args, settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}):
        super(prms_marrmot, self).__init__()
        self.args = args
        self.settings = settings

    def smoothThreshold_temperature_logistic(self, T, Tt, r=0.01):
        # By transforming the equation above to Sf = f(P,T,Tt,r)
        # Sf = P * 1/ (1+exp((T-Tt)/r))
        # T       : current temperature
        # Tt      : threshold temperature below which snowfall occurs
        # r       : [optional] smoothing parameter rho, default = 0.01
        # calculate multiplier
        out = 1 / (1 + torch.exp((T - Tt) / r))

        return out

    def snowfall_1(self, In, T, p1, varargin=0.01):
        out = In * (self.smoothThreshold_temperature_logistic(T, p1, r=varargin))
        return out

    def rainfall_1(self, In, T, p1, varargin=0.01):
        # inputs:
        # p1   - temperature threshold above which rainfall occurs [oC]
        # T    - current temperature [oC]
        # In   - incoming precipitation flux [mm/d]
        # varargin(1) - smoothing variable r (default 0.01)
        out = In * (1 - self.smoothThreshold_temperature_logistic(T, p1, r=varargin))
        return out

    def split_1(self, p1, In):
        # inputs:
        # p1   - fraction of flux to be diverted [-]
        # In   - incoming flux [mm/d]
        out = p1 * In
        return out

    def smoothThreshold_storage_logistic(self, S, Smax, r=2, e=1):
        Smax = torch.where(Smax < 0.0,
                           torch.zeros(Smax.shape, dtype=torch.float32, device=self.args["device"]),
                           Smax)

        out = torch.where(r * Smax == 0.0,
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / r)),
                          1 / (1 + torch.exp((S - Smax + r * e * Smax) / (r * Smax))))
        return out

    def interception_1(self, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)

        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax))   # , varargin_r, varargin_e
        return out

    def melt_1(self, p1, p2, T, S, dt):
        # Constraints:  f <= S/dt
        # inputs:
        # p1   - degree-day factor [mm/oC/d]
        # p2   - temperature threshold for snowmelt [oC]
        # T    - current temperature [oC]
        # S    - current storage [mm]
        # dt   - time step size [d]
        out = torch.min(p1 * (T - p2), S / dt)
        out = torch.clamp(out, min=0.0)
        return out

    def saturation_1(self, In, S, Smax, varargin_r=0.01, varargin_e=5.0):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)
        out = In * (1 - self.smoothThreshold_storage_logistic(S, Smax))   #, varargin_r, varargin_e
        return out

    def saturation_8(self, p1, p2, S, Smax, In):
        # description: Description:  Saturation excess flow from a store with different degrees
        # of saturation (min-max linear variant)
        # inputs:
        # p1   - minimum fraction contributing area [-]
        # p2   - maximum fraction contributing area [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # In   - incoming flux [mm/d]
        out = (p1 + (p2 - p1) * S / Smax) * In
        return out

    def effective_1(self, In1, In2):
        # description: General effective flow (returns flux [mm/d]) Constraints:  In1 > In2
        # inputs:
        # In1  - first flux [mm/d]
        # In2  - second flux [mm/d]
        out = torch.clamp(In1 - In2, min=0.0)
        return out

    def recharge_7(selfself, p1, fin):
        # Description:  Constant recharge limited by incoming flux
        # p1   - maximum recharge rate [mm/d]
        # fin  - incoming flux [mm/d]
        out = torch.min(p1, fin)
        return out

    def recharge_2(self, p1, S, Smax, flux):
        # Description:  Recharge as non-linear scaling of incoming flux
        # Constraints:  S >= 0
        # inputs:
        # p1   - recharge scaling non-linearity [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # flux - incoming flux [mm/d]
        S = torch.clamp(S, min=0.0)
        out = flux * ((S / Smax) ** p1)
        return out

    def interflow_4(self, p1, p2, S):
        # Description:  Combined linear and scaled quadratic interflow
        # Constraints: f <= S
        #              S >= 0     - prevents numerical issues with complex numbers
        # inputs:
        # p1   - time coefficient [d-1]
        # p2   - scaling factor [mm-1 d-1]
        # S    - current storage [mm]
        S = torch.clamp(S, min=0.0)
        out = torch.min(S, p1 * S + p2 * (S ** 2))
        return out

    def baseflow_1(self, p1, S):
        # Description:  Outflow from a linear reservoir
        # inputs:
        # p1   - time scale parameter [d-1]
        # S    - current storage [mm]
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        # Description:  Evaporation at the potential rate
        # Constraints:  f <= S/dt
        # inputs:
        # S    - current storage [mm]
        # Ep   - potential evaporation rate [mm/d]
        # dt   - time step size
        out = torch.min(S / dt, Ep)
        return out

    def evap_7(self, S, Smax, Ep, dt):
        # Description:  Evaporation scaled by relative storage
        # Constraints:  f <= S/dt
        # input:
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # Ep   - potential evapotranspiration rate [mm/d]
        # dt   - time step size [d]
        Ep = torch.clamp(Ep, min=0.0)
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def evap_15(self, Ep, S1, S1max, S2, S2min, dt):
        # Description:  Scaled evaporation if another store is below a threshold
        #  Constraints:  f <= S1/dt
        # inputs:
        # Ep    - potential evapotranspiration rate [mm/d]
        # S1    - current storage in S1 [mm]
        # S1max - maximum storage in S1 [mm]
        # S2    - current storage in S2 [mm]
        # S2min - minimum storage in S2 [mm]
        # dt    - time step size [d]

        # this needs to be checked because in MATLAB version there is a min function that does not make sense to me
        Ep = torch.clamp(Ep, min=0.0)
        out = (S1 / S1max * Ep) * self.smoothThreshold_storage_logistic(S2, S2min, S1 / dt)
        out = torch.clamp(out, min=0.0)
        return out

    def multi_comp_semi_static_params(
            self, params, param_no, args, interval=30, method="average"
    ):
        # seperate the piece for each interval
        nmul = args["nmul"]
        param = params[:, :, param_no * nmul: (param_no + 1) * nmul]
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
                    ((i * interval) + remainder): (
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
                    (((i) * interval) + remainder): (((i) * interval) + remainder)
                                                    + 1,
                    :,
                    ]
                ).repeat((1, interval, 1))
                param_name_list.append(param_name)
        else:
            print("this method is not defined yet in function semi_static_params")
        new_param = torch.cat(param_name_list, 1)
        return new_param

    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list_prms"]:
            out_temp = (
                    params[:, -1, num * nmul: (num + 1) * nmul]
                    * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                    + args["marrmot_paramCalLst"][num][0]
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
                    out_temp * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                    + args["marrmot_paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                    params[:, :, num * nmul: (num + 1) * nmul]
                    * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                    + args["marrmot_paramCalLst"][num][0]
            )
        return out

    def ODE_approx_IE(self, args, t, S1_old, S2_old, S3_old, S4_old, S5_old, S6_old, S7_old,
                      delta_S1, delta_S2, delta_S3, delta_S4, delta_S5, delta_S6, delta_S7):
        return S1_old




    def Run_for_one_day(self,S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        # storages:
        # There are 7 reservoirs in PRMS
        if S_tensor.dim() == 3:
            S1 = S_tensor[:, 0, :]
            S2 = S_tensor[:, 1, :]
            S3 = S_tensor[:, 2, :]
            S4 = S_tensor[:, 3, :]
            S5 = S_tensor[:, 4, :]
            S6 = S_tensor[:, 5, :]
            S7 = S_tensor[:, 6, :]
        elif S_tensor.dim() == 2:  # mostly for calculating Jacobian!
            S1 = S_tensor[:, 0]
            S2 = S_tensor[:, 1]
            S3 = S_tensor[:, 2]
            S4 = S_tensor[:, 3]
            S5 = S_tensor[:, 4]
            S6 = S_tensor[:, 5]
            S7 = S_tensor[:, 6]

        delta_t = 1  # timestep (day)
        # P = Precip[:, t, :]
        # Ep = PET[:, t, :]
        # T = mean_air_temp[:, t, :]

        # fluxes
        flux_ps = self.snowfall_1(P, T, tt)
        flux_pr = self.rainfall_1(P, T, tt)
        flux_pim = self.split_1(1 - beta, flux_pr)
        flux_psm = self.split_1(beta, flux_pr)
        flux_pby = self.split_1(1 - alpha, flux_psm)
        flux_pin = self.split_1(alpha, flux_psm)
        flux_ptf = self.interception_1(flux_pin, S2, stor)
        flux_m = self.melt_1(ddf, tt, T, S1, delta_t)
        flux_mim = self.split_1(1 - beta, flux_m)
        flux_msm = self.split_1(beta, flux_m)
        flux_sas = self.saturation_1(flux_pim + flux_mim, S3, retip)
        flux_sro = self.saturation_8(scn, scx, S4, remx, flux_msm + flux_ptf + flux_pby)
        flux_inf = self.effective_1(flux_msm + flux_ptf + flux_pby, flux_sro)
        flux_pc = self.saturation_1(flux_inf, S4, remx)
        flux_excs = self.saturation_1(flux_pc, S5, smax)
        flux_sep = self.recharge_7(cgw, flux_excs)
        flux_qres = self.effective_1(flux_excs, flux_sep)
        flux_gad = self.recharge_2(k2, S6, resmax, k1)
        flux_ras = self.interflow_4(k3, k4, S6)
        flux_bas = self.baseflow_1(k5, S7)
        flux_snk = self.baseflow_1(k6, S7)  # represents transbasin gw or undergage streamflow
        flux_ein = self.evap_1(S2, beta * Ep, delta_t)
        flux_eim = self.evap_1(S3, (1 - beta) * Ep, delta_t)
        flux_ea = self.evap_7(S4, remx, Ep - flux_ein - flux_eim, delta_t)
        flux_et = self.evap_15(Ep - flux_ein - flux_eim - flux_ea, S5, smax, S4,
                               Ep - flux_ein - flux_eim, delta_t)

        # stores ODEs
        dS1 = flux_ps - flux_m
        dS2 = flux_pin - flux_ein - flux_ptf
        dS3 = flux_pim + flux_mim - flux_eim - flux_sas
        dS4 = flux_inf - flux_ea - flux_pc
        dS5 = flux_pc - flux_et - flux_excs
        dS6 = flux_qres - flux_gad - flux_ras
        dS7 = flux_sep + flux_gad - flux_bas - flux_snk

        # dS_tensor dimension: [ batch, 7, nmul]
        dS_tensor = torch.cat((dS1.unsqueeze(1),
                               dS2.unsqueeze(1),
                               dS3.unsqueeze(1),
                               dS4.unsqueeze(1),
                               dS5.unsqueeze(1),
                               dS6.unsqueeze(1),
                               dS7.unsqueeze(1)), dim=1)

        fluxes_tensor = torch.cat((flux_ps.unsqueeze(1), flux_pr.unsqueeze(1),
                                   flux_pim.unsqueeze(1), flux_psm.unsqueeze(1),
                                   flux_pby.unsqueeze(1), flux_pin.unsqueeze(1),
                                   flux_ptf.unsqueeze(1), flux_m.unsqueeze(1),
                                   flux_mim.unsqueeze(1), flux_msm.unsqueeze(1),
                                   flux_sas.unsqueeze(1), flux_sro.unsqueeze(1),
                                   flux_inf.unsqueeze(1), flux_pc.unsqueeze(1),
                                   flux_excs.unsqueeze(1), flux_qres.unsqueeze(1),
                                   flux_sep.unsqueeze(1), flux_gad.unsqueeze(1),
                                   flux_ras.unsqueeze(1), flux_bas.unsqueeze(1),
                                   flux_snk.unsqueeze(1), flux_ein.unsqueeze(1),
                                   flux_eim.unsqueeze(1), flux_ea.unsqueeze(1),
                                   flux_et.unsqueeze(1)), dim=1)
        return dS_tensor, fluxes_tensor

    # def error_func(self,y,t, ):
    def error_func(self, S_tensor_new, S_tensor_old, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6, dt=1):
        delta_S,_ = self.Run_for_one_day(S_tensor_old, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6);  ##bs*ny
        # err = (y - self.y0)/self.dt - delta_S
        err = (S_tensor_new - S_tensor_old) / dt - delta_S
        return err  ##bs*ny

    # def ODEsolver_NR(self, y0, dt, t):
    def ODEsolver_NR(self, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        ALPHA = 1e-4;  # criteria for decrease
        MIN_LAMBDA = 0.1;  # min lambda
        MAX_LAMBDA = 0.5;  # max lambda
        # y = y0;  # initial guess
        y = torch.clone(S_tensor)  # initial guess
        # self.y0 = y0
        # self.dt = dt
        F = self.error_func(y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6)  # evaluate initial guess  bs*ny
        # bs, ny = y.shape
        # jac = torch.autograd.functional.jacobian(self.error_func, (y, t))  ##bs*ny*ny
        jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor,
                                                                   P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
        jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
        jac_new = jac_new.permute(2, 0, 1)

        if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
            exitflag = -1;  # matrix may be singular
        else:
            exitflag = 1;  # normal exit

        resnorm = torch.linalg.norm(F, float('inf'), dim=[1])  # calculate norm of the residuals
        resnorm0 = 100 * resnorm;
        # dy = torch.zeros(y.shape).to(y0);  # dummy values
        dy = torch.zeros(y.shape).to(S_tensor);  # dummy values
        ##%% solver
        Niter = 0;  # start counter
        # lambda_ = torch.tensor(1.0).to(y0)  # backtracking
        lambda_ = torch.tensor(1.0).to(S_tensor)  # backtracking
        while ((torch.max(resnorm) > self.settings["TolFun"] or lambda_ < 1) and exitflag >= 0 and Niter <=
               self.settings["MaxIter"]):
            if lambda_ == 1:
                ### Newton-Raphson solver
                Niter = Niter + 1;  ## increment counter
                ### update Jacobian, only if necessary
                if torch.max(resnorm / resnorm0) > 0.2:
                    jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
                    jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
                    jac_new = jac_new.permute(2, 0, 1)  ## bs*ny*ny

                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;  ## % matrix may be singular
                        break

                if torch.min(1 / torch.linalg.cond(jac_new, p=1)) <= 2.2204e-16:
                    dy = torch.bmm(torch.linalg.pinv(jac_new), -F.unsqueeze(-1)).squeeze(
                        -1);  ## bs*ny*ny  , bs*ny*1 = bs*ny
                else:
                    dy = -torch.linalg.lstsq(jac_new, F).solution;
                g = torch.bmm(F.unsqueeze(1), jac_new);  # %star; % gradient of resnorm  bs*1*ny, bs*ny*ny = bs*1*ny
                slope = torch.bmm(g, dy.unsqueeze(-1)).squeeze();  # %_star; % slope of gradient  bs*1*ny,bs*ny*1
                fold_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% objective function
                yold = torch.clone(y);  ##% initial value
                lambda_min = self.settings["TolX"] / torch.max(abs(dy) / torch.maximum(abs(yold), torch.tensor(1.0)));
            if lambda_ < lambda_min:
                exitflag = 2;  ##% x is too close to XOLD
                break
            elif torch.isnan(dy).any() or torch.isinf(dy).any():
                exitflag = -1;  ##% matrix may be singular
                break
            y = yold + dy * lambda_;  ## % next guess
            # F = self.error_func(y, t);  ## % evaluate this guess
            F = self.error_func(y, S_tensor, P, Ep, T,
                                tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                cgw, resmax, k1, k2, k3, k4, k5, k6)
            b = F[F!=F]
            if len(b) > 0:
                print("end")
            f_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% new objective function
            ###%% check for convergence
            lambda1 = lambda_;  ###% save previous lambda
            if torch.any(f_obj > fold_obj + ALPHA * lambda_ * slope):
                if lambda_ == 1:
                    a = torch.maximum(f_obj - fold_obj - slope, torch.tensor(0.0000001))
                    lambda_ = torch.min(-slope / 2.0 / (a));  ##% calculate lambda
                else:

                    A = 1 / (lambda1 - lambda2);  ##Scalar
                    B = torch.stack([torch.stack([1.0 / lambda1 ** 2.0, -1.0 / lambda2 ** 2.0]),
                                     torch.stack([-lambda2 / lambda1 ** 2.0, lambda1 / lambda2 ** 2.0])]);  ##2*2
                    C = torch.stack([f_obj - fold_obj - lambda1 * slope, f2_obj - fold_obj - lambda2 * slope]);  ##2*1
                    a = (A * B @ C)[0, :];
                    b = (A * B @ C)[1, :];
                    a = torch.maximum(a, torch.tensor(0.0000001))
                    b = torch.maximum(b, torch.tensor(0.0000001))
                    if torch.all(a == 0):
                        lambda_tmp = -slope / 2 / b;
                    else:
                        discriminant = b ** 2 - 3 * a * slope;
                        if torch.any(discriminant < 0):
                            lambda_tmp = MAX_LAMBDA * lambda1;
                        elif torch.any(b <= 0):
                            lambda_tmp = (-b + torch.sqrt(discriminant)) / 3 / a;
                        else:
                            lambda_tmp = -slope / (b + torch.sqrt(discriminant));

                    lambda_ = torch.min(
                        torch.minimum(lambda_tmp, torch.tensor(MAX_LAMBDA * lambda1)));  # % minimum step length

            elif torch.isnan(f_obj).any() or torch.isinf(f_obj).any():
                ## % limit undefined evaluation or overflow
                lambda_ = MAX_LAMBDA * lambda1;
            else:
                lambda_ = torch.tensor(1.0).to(S_tensor);  ### % fraction of Newton step

            if lambda_ < 1:
                lambda2 = lambda1;
                f2_obj = torch.clone(f_obj);  ##% save 2nd most previous value
                lambda_ = torch.maximum(lambda_, torch.tensor(MIN_LAMBDA * lambda1));  ###% minimum step length
                continue
            #lambda2 = lambda_
            resnorm0 = resnorm;  ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'), dim=[1]);  ###% calculate new resnorm
        print("day ", "Iteration ", Niter, "Flag ", exitflag)
        return y, F, exitflag

    def ODEsolver_NR_modified(self, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6):
        ALPHA = 1e-4;  # criteria for decrease
        MIN_LAMBDA = 0.1;  # min lambda
        MAX_LAMBDA = 0.5;  # max lambda
        # y = y0;  # initial guess
        y = torch.clone(S_tensor)  # initial guess
        # self.y0 = y0
        # self.dt = dt
        F = self.error_func(y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6)  # evaluate initial guess  bs*ny
        # bs, ny = y.shape
        # jac = torch.autograd.functional.jacobian(self.error_func, (y, t))  ##bs*ny*ny
        jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor,
                                                                   P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
        jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
        jac_new = jac_new.permute(2, 0, 1)

        if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
            exitflag = -1;  # matrix may be singular
        else:
            exitflag = 1;  # normal exit

        resnorm = torch.linalg.norm(F, float('inf'), dim=[1])  # calculate norm of the residuals
        resnorm0 = 100 * resnorm;
        # dy = torch.zeros(y.shape).to(y0);  # dummy values
        dy = torch.zeros(y.shape).to(S_tensor);  # dummy values
        ##%% solver
        Niter = 0;  # start counter
        # lambda_ = torch.tensor(1.0).to(y0)  # backtracking
        # lambda_ = torch.tensor(1.0).to(S_tensor)  # backtracking
        lambda_ = torch.ones(S_tensor.shape).to(S_tensor)
        while ((torch.max(resnorm) > self.settings["TolFun"] or lambda_ < 1) and exitflag >= 0 and Niter <=
               self.settings["MaxIter"]):
            if lambda_ == 1:
                ### Newton-Raphson solver
                Niter = Niter + 1;  ## increment counter
                ### update Jacobian, only if necessary
                if torch.max(resnorm / resnorm0) > 0.2:
                    jac = torch.autograd.functional.jacobian(self.error_func, (y, S_tensor, P, Ep, T,
                                            tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                            cgw, resmax, k1, k2, k3, k4, k5, k6))
                    jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
                    jac_new = jac_new.permute(2, 0, 1)  ## bs*ny*ny

                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;  ## % matrix may be singular
                        break

                if torch.min(1 / torch.linalg.cond(jac_new, p=1)) <= 2.2204e-16:
                    dy = torch.bmm(torch.linalg.pinv(jac_new), -F.unsqueeze(-1)).squeeze(
                        -1);  ## bs*ny*ny  , bs*ny*1 = bs*ny
                else:
                    dy = -torch.linalg.lstsq(jac_new, F).solution;
                g = torch.bmm(F.unsqueeze(1), jac_new);  # %star; % gradient of resnorm  bs*1*ny, bs*ny*ny = bs*1*ny
                slope = torch.bmm(g, dy.unsqueeze(-1)).squeeze();  # %_star; % slope of gradient  bs*1*ny,bs*ny*1
                fold_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% objective function
                yold = torch.clone(y);  ##% initial value
                lambda_min = self.settings["TolX"] / torch.max(abs(dy) / torch.maximum(abs(yold), torch.tensor(1.0)));
            if lambda_ < lambda_min:
                exitflag = 2;  ##% x is too close to XOLD
                break
            elif torch.isnan(dy).any() or torch.isinf(dy).any():
                exitflag = -1;  ##% matrix may be singular
                break
            y = yold + dy * lambda_;  ## % next guess
            # F = self.error_func(y, t);  ## % evaluate this guess
            F = self.error_func(y, S_tensor, P, Ep, T,
                                tt, ddf, alpha, beta, stor, retip, fscn, scx, scn, flz, stot, remx, smax,
                                cgw, resmax, k1, k2, k3, k4, k5, k6)
            a = F[F!=F]
            if len(a) > 0:
                print("end")
            f_obj = torch.bmm(F.unsqueeze(1), F.unsqueeze(-1)).squeeze();  ###% new objective function
            ###%% check for convergence
            lambda1 = lambda_;  ###% save previous lambda

            if torch.any(f_obj > fold_obj + ALPHA * lambda_ * slope):
                if lambda_ == 1:
                    lambda_ = torch.min(-slope / 2.0 / (f_obj - fold_obj - slope));  ##% calculate lambda
                else:

                    A = 1 / (lambda1 - lambda2);  ##Scalar
                    B = torch.stack([torch.stack([1.0 / lambda1 ** 2.0, -1.0 / lambda2 ** 2.0]),
                                     torch.stack([-lambda2 / lambda1 ** 2.0, lambda1 / lambda2 ** 2.0])]);  ##2*2
                    C = torch.stack([f_obj - fold_obj - lambda1 * slope, f2_obj - fold_obj - lambda2 * slope]);  ##2*1
                    a = (A * B @ C)[0, :];
                    b = (A * B @ C)[1, :];

                    if torch.all(a == 0):
                        lambda_tmp = -slope / 2 / b;
                    else:
                        discriminant = b ** 2 - 3 * a * slope;
                        if torch.any(discriminant < 0):
                            lambda_tmp = MAX_LAMBDA * lambda1;
                        elif torch.any(b <= 0):
                            lambda_tmp = (-b + torch.sqrt(discriminant)) / 3 / a;
                        else:
                            lambda_tmp = -slope / (b + torch.sqrt(discriminant));

                    lambda_ = torch.min(
                        torch.minimum(lambda_tmp, torch.tensor(MAX_LAMBDA * lambda1)));  # % minimum step length

            elif torch.isnan(f_obj).any() or torch.isinf(f_obj).any():
                ## % limit undefined evaluation or overflow
                lambda_ = MAX_LAMBDA * lambda1;
            else:
                lambda_ = torch.tensor(1.0).to(S_tensor);  ### % fraction of Newton step

            if lambda_ < 1:
                lambda2 = lambda1;
                f2_obj = torch.clone(f_obj);  ##% save 2nd most previous value
                lambda_ = torch.maximum(lambda_, torch.tensor(MIN_LAMBDA * lambda1));  ###% minimum step length
                continue

            resnorm0 = resnorm;  ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'), dim=[1]);  ###% calculate new resnorm
        # print("day ", t.detach().cpu().numpy(), "Iteration ", Niter, "Flag ", exitflag)
        return y, F, exitflag


    def f3D(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = prms_marrmot(args=args)
                S_tensor = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            # All storages in prms. There are 7.
            S_tensor = torch.zeros(
                [x.shape[0], 7, nmul], dtype=torch.float32, device=args["device"]
            ) + 2



        ## parameters for prms_marrmot. there are 18 parameters in it
        tt = self.multi_comp_parameter_bounds(params, 0, args)
        ddf = self.multi_comp_parameter_bounds(params, 1, args)
        alpha = self.multi_comp_parameter_bounds(params, 2, args)
        beta = self.multi_comp_parameter_bounds(params, 3, args)
        stor = self.multi_comp_parameter_bounds(params, 4, args)
        retip = self.multi_comp_parameter_bounds(params, 5, args)
        fscn = self.multi_comp_parameter_bounds(params, 6, args)
        scx = self.multi_comp_parameter_bounds(params, 7, args)
        scn = fscn * scx
        flz = self.multi_comp_parameter_bounds(params, 8, args)
        stot = self.multi_comp_parameter_bounds(params, 9, args)
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = self.multi_comp_parameter_bounds(params, 10, args)
        resmax = self.multi_comp_parameter_bounds(params, 11, args)
        k1 = self.multi_comp_parameter_bounds(params, 12, args)
        k2 = self.multi_comp_parameter_bounds(params, 13, args)
        k3 = self.multi_comp_parameter_bounds(params, 14, args)
        k4 = self.multi_comp_parameter_bounds(params, 15, args)
        k5 = self.multi_comp_parameter_bounds(params, 16, args)
        k6 = self.multi_comp_parameter_bounds(params, 17, args)
        #################
        # inputs
        Precip = (
            x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2
        dayl = (
            x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]
        hamon_coef = torch.ones(dayl.shape, dtype=torch.float32, device=args["device"]) * 0.006  # this can be param
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        )

        # initialize the Q_sim
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])


        for t in range(Ndays):
            P = Precip[:, t, :]
            Ep = PET[:, t, :]
            T = mean_air_temp[:, t, :]
            delta_S, _ = self.Run_for_one_day(S_tensor, P, Ep, T,
                                            tt[:, t, :], ddf[:, t, :], alpha[:, t, :], beta[:, t, :], stor[:, t, :],
                                              retip[:, t, :], fscn[:, t, :], scx[:, t, :], scn[:, t, :],
                                              flz[:, t, :], stot[:, t, :], remx[:, t, :], smax[:, t, :],
                                            cgw[:, t, :], resmax[:, t, :], k1[:, t, :], k2[:, t, :],
                                              k3[:, t, :], k4[:, t, :], k5[:, t, :], k6[:, t, :])

            S, error, exit_flag = self.ODEsolver_NR(S_tensor, P, Ep, T,
                                            tt[:, t, :], ddf[:, t, :], alpha[:, t, :], beta[:, t, :], stor[:, t, :],
                                              retip[:, t, :], fscn[:, t, :], scx[:, t, :], scn[:, t, :],
                                              flz[:, t, :], stot[:, t, :], remx[:, t, :], smax[:, t, :],
                                            cgw[:, t, :], resmax[:, t, :], k1[:, t, :], k2[:, t, :],
                                              k3[:, t, :], k4[:, t, :], k5[:, t, :], k6[:, t, :])
        return S_tensor, dS_tensor, fluxes_tensor
    def forward(self, x, c_PRMS, params, args, warm_up=0, init=False):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        bs = self.args["hyperparameters"]["batch_size"]
        vars = args["optData"]["varT_PRMS"]
        vars_c_PRMS = args["optData"]["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, 0:warm_up, :]
                warm_up_model = prms_marrmot(args=args)
                S_tensor, _, _ = warm_up_model(xinit, c_PRMS, paramsinit, args, warm_up=0, init=True)
        else:
            # All storages in prms. There are 7.
            # S_tensor = torch.zeros(
            #     [x.shape[0] * nmul, 7], dtype=torch.float32, device=args["device"]
            # ) + 2
            S_tensor = torch.tensor([15,7,3,8,22,10,10]).unsqueeze(0).to(x)

        tt = torch.tensor([1.0]).repeat([1, 365]).to(x)
        ddf = torch.tensor([10.0]).repeat([1, 365]).to(x)
        alpha = torch.tensor([0.5]).repeat([1, 365]).to(x)
        beta = torch.tensor([0.5]).repeat([1, 365]).to(x)
        stor = torch.tensor([2.5]).repeat([1, 365]).to(x)
        retip = torch.tensor([25.0]).repeat([1, 365]).to(x)
        fscn = torch.tensor([0.5]).repeat([1, 365]).to(x)
        scx = torch.tensor([0.5]).repeat([1, 365]).to(x)
        scn = fscn * scx
        flz = torch.tensor([0.5]).repeat([1, 365]).to(x)
        stot = torch.tensor([1000.0]).repeat([1, 365]).to(x)
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = torch.tensor([10.0]).repeat([1, 365]).to(x)
        resmax = torch.tensor([150.0]).repeat([1, 365]).to(x)
        k1 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k2 = torch.tensor([2.5]).repeat([1, 365]).to(x)
        k3 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k4 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k5 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        k6 = torch.tensor([0.5]).repeat([1, 365]).to(x)
        # ## parameters for prms_marrmot. there are 18 parameters in it
        # tt = self.multi_comp_parameter_bounds(params, 0, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # ddf = self.multi_comp_parameter_bounds(params, 1, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # alpha = self.multi_comp_parameter_bounds(params, 2, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # beta = self.multi_comp_parameter_bounds(params, 3, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # stor = self.multi_comp_parameter_bounds(params, 4, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # retip = self.multi_comp_parameter_bounds(params, 5, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # fscn = self.multi_comp_parameter_bounds(params, 6, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # scx = self.multi_comp_parameter_bounds(params, 7, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # scn = fscn * scx
        # flz = self.multi_comp_parameter_bounds(params, 8, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # stot = self.multi_comp_parameter_bounds(params, 9, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # remx = (1 - flz) * stot
        # smax = flz * stot
        # cgw = self.multi_comp_parameter_bounds(params, 10, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # resmax = self.multi_comp_parameter_bounds(params, 11, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k1 = self.multi_comp_parameter_bounds(params, 12, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k2 = self.multi_comp_parameter_bounds(params, 13, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k3 = self.multi_comp_parameter_bounds(params, 14, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k4 = self.multi_comp_parameter_bounds(params, 15, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k5 = self.multi_comp_parameter_bounds(params, 16, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # k6 = self.multi_comp_parameter_bounds(params, 17, args).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        #################
        # inputs
        climate_path = r"G:\\Farshid\\GitHub\\MARRMoT\\MARRMoT\\prms_climate_data.csv"
        climate = torch.tensor((pd.read_csv(climate_path, header=None)).to_numpy()).to(x)
        Precip = climate[0:365,0:1].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        mean_air_temp = climate[0:365,2:3].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        PET = climate[0:365,1:2].permute(1,0).unsqueeze(-1).repeat(1, 1, nmul).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])


        # Precip = (
        #     x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        # ).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # mean_air_temp = ((Tmaxf + Tminf) / 2).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        # dayl = (
        #     x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        # ).permute(0, 2, 1).reshape([bs * nmul, params.shape[1]])
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]
        # hamon_coef = torch.ones(dayl.shape, dtype=torch.float32, device=args["device"]) * 0.006  # this can be param
        # PET = get_potet(
        #     args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        # )

        # initialize the Q_sim
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])

        for t in range(Ndays):
            P = Precip[:, t]
            Ep = PET[:, t]
            T = mean_air_temp[:, t]
            delta_S, fluxes_tensor = self.Run_for_one_day(S_tensor, P, Ep, T,
                                              tt[:, t], ddf[:, t], alpha[:, t], beta[:, t],
                                              stor[:, t],
                                              retip[:, t], fscn[:, t], scx[:, t], scn[:, t],
                                              flz[:, t], stot[:, t], remx[:, t], smax[:, t],
                                              cgw[:, t], resmax[:, t], k1[:, t], k2[:, t],
                                              k3[:, t], k4[:, t], k5[:, t], k6[:, t])
            a = fluxes_tensor[fluxes_tensor!=fluxes_tensor]
            if len(a) > 0:
                print("end")
            S_tensor, error, exit_flag = self.ODEsolver_NR(S_tensor, P, Ep, T,
                                                    tt[:, t], ddf[:, t], alpha[:, t], beta[:, t],
                                                    stor[:, t],
                                                    retip[:, t], fscn[:, t], scx[:, t], scn[:, t],
                                                    flz[:, t], stot[:, t], remx[:, t], smax[:, t],
                                                    cgw[:, t], resmax[:, t], k1[:, t], k2[:, t],
                                                    k3[:, t], k4[:, t], k5[:, t], k6[:, t])






        return S_tensor, dS_tensor, fluxes_tensor
            # in marrmot code, there are three ways of solving it:
            # 1) Newton-Raphson
            # 2) fsolve
            # 3) isqnonlin



class prms_marrmot(torch.nn.Module):
    def __init__(self):
        super(prms_marrmot, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()


    def smoothThreshold_temperature_logistic(self, T, Tt, r=0.01):
        # By transforming the equation above to Sf = f(P,T,Tt,r)
        # Sf = P * 1/ (1+exp((T-Tt)/r))
        # T       : current temperature
        # Tt      : threshold temperature below which snowfall occurs
        # r       : [optional] smoothing parameter rho, default = 0.01
        # calculate multiplier
        # out = 1 / (1 + torch.exp((T - Tt) / r))
        # out = torch.where(T < Tt,
        #                   torch.ones(T.shape, dtype=torch.float32, device=T.device),
        #                   torch.zeros(T.shape, dtype=torch.float32, device=T.device))
        out = (T <= Tt).type(torch.float32)
        # out = torch.clamp(F.logsigmoid(-(T - Tt) / r), 1e-5, 1.0 - 1e-5)
        # out = (torch.zeros(T.shape).to(T)) + 0.1
        return out


    def snowfall_1(self, In, T, p1, varargin = 0.01):
        out = In * (self.smoothThreshold_temperature_logistic(T, p1))   #, r=varargin
        return out

    def rainfall_1(self, In, T, p1, varargin = 0.01):
        # inputs:
        # p1   - temperature threshold above which rainfall occurs [oC]
        # T    - current temperature [oC]
        # In   - incoming precipitation flux [mm/d]
        # varargin(1) - smoothing variable r (default 0.01)
        out = In * (1 - self.smoothThreshold_temperature_logistic(T, p1))   #, r=varargin
        return out

    def split_1(self, p1, In):
        # inputs:
        # p1   - fraction of flux to be diverted [-]
        # In   - incoming flux [mm/d]
        out = p1 * In
        return out

    def smoothThreshold_storage_logistic(self, args, S, Smax, r=0.01, e=5.0):   # r= 0.01, e = 5.0
        # Smax = torch.clamp(Smax, min=0.0)
        #
        # a_temp = torch.where(r * Smax == 0.0,
        #                   (S - Smax + r * e * Smax) / r,
        #                   (S - Smax + r * e * Smax) / (r * Smax))
        # a_temp2 = torch.clamp(a_temp, min=-6.0, max=10.0)
        # out = 1 / (1 + torch.exp(a_temp2))

        # out = torch.where(r * Smax == 0.0,
        #                   1 / (1 + torch.exp((S - Smax + r * e * Smax) / r)),
        #                   1 / (1 + torch.exp((S - Smax + r * e * Smax) / (r * Smax))))

        # out = torch.where(r * Smax == 0.0,
        #                   torch.clamp(F.logsigmoid(-(S - Smax + r * e * Smax) / r),  1e-5, 1.0 - 1e-5),
        #                   torch.clamp(F.logsigmoid(-(S - Smax + r * e * Smax) / (r * Smax)),  1e-5, 1.0 - 1e-5))
        out = (torch.zeros(S.shape).to(S)) + 0.05
        return out

    def interception_1(self, args, In, S, Smax, varargin_r=2, varargin_e=1):   # varargin_r=0.01, varargin_e=5.0
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)

        out = In * (1 - self.smoothThreshold_storage_logistic(args, S, Smax))   # , varargin_r, varargin_e
        return out

    def melt_1(self, p1, p2, T, S, dt):
        # Constraints:  f <= S/dt
        # inputs:
        # p1   - degree-day factor [mm/oC/d]
        # p2   - temperature threshold for snowmelt [oC]
        # T    - current temperature [oC]
        # S    - current storage [mm]
        # dt   - time step size [d]
        out = torch.min(p1 * (T - p2), S / dt)
        out = torch.clamp(out, min=0.0)
        return out

    def saturation_1(self, args, In, S, Smax, varargin_r=2, varargin_e=2):
        # inputs:
        # In   - incoming flux [mm/d]
        # S    - current storage [mm]
        # Smax - maximum storage [mm]
        # varargin_r - smoothing variable r (default 0.01)
        # varargin_e - smoothing variable e (default 5.00)
        out = In * (1 - self.smoothThreshold_storage_logistic(args, S, Smax))   #, varargin_r, varargin_e
        return out

    def saturation_8(self, p1, p2, S, Smax, In):
        # description: Description:  Saturation excess flow from a store with different degrees
        # of saturation (min-max linear variant)
        # inputs:
        # p1   - minimum fraction contributing area [-]
        # p2   - maximum fraction contributing area [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # In   - incoming flux [mm/d]
        out = (p1 + (p2 - p1) * S / Smax) * In
        return out

    def effective_1(self, In1, In2):
        # description: General effective flow (returns flux [mm/d]) Constraints:  In1 > In2
        # inputs:
        # In1  - first flux [mm/d]
        # In2  - second flux [mm/d]
        out = torch.clamp(In1 - In2, min=0.0)
        return out

    def recharge_7(selfself, p1, fin):
        # Description:  Constant recharge limited by incoming flux
        # p1   - maximum recharge rate [mm/d]
        # fin  - incoming flux [mm/d]
        out = torch.min(p1, fin)
        return out

    def recharge_2(self, p1, S, Smax, flux):
        # Description:  Recharge as non-linear scaling of incoming flux
        # Constraints:  S >= 0
        # inputs:
        # p1   - recharge scaling non-linearity [-]
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # flux - incoming flux [mm/d]
        S = torch.clamp(S, min=0.0)
        out = flux * ((S / Smax) ** p1)
        return out

    def interflow_4(self, p1, p2, S):
        # Description:  Combined linear and scaled quadratic interflow
        # Constraints: f <= S
        #              S >= 0     - prevents numerical issues with complex numbers
        # inputs:
        # p1   - time coefficient [d-1]
        # p2   - scaling factor [mm-1 d-1]
        # S    - current storage [mm]
        S = torch.clamp(S, min=0.0)
        out = torch.min(S, p1 * S + p2 * (S ** 2))
        return out

    def baseflow_1(self, p1, S):
        # Description:  Outflow from a linear reservoir
        # inputs:
        # p1   - time scale parameter [d-1]
        # S    - current storage [mm]
        out = p1 * S
        return out

    def evap_1(self, S, Ep, dt):
        # Description:  Evaporation at the potential rate
        # Constraints:  f <= S/dt
        # inputs:
        # S    - current storage [mm]
        # Ep   - potential evaporation rate [mm/d]
        # dt   - time step size
        out = torch.min(S / dt, Ep)
        return out

    def evap_7(self, S, Smax, Ep, dt):
        # Description:  Evaporation scaled by relative storage
        # Constraints:  f <= S/dt
        # input:
        # S    - current storage [mm]
        # Smax - maximum contributing storage [mm]
        # Ep   - potential evapotranspiration rate [mm/d]
        # dt   - time step size [d]
        out = torch.min(S / Smax * Ep, S / dt)
        return out

    def evap_15(self, args, Ep, S1, S1max, S2, S2min, dt):
        # Description:  Scaled evaporation if another store is below a threshold
        #  Constraints:  f <= S1/dt
        # inputs:
        # Ep    - potential evapotranspiration rate [mm/d]
        # S1    - current storage in S1 [mm]
        # S1max - maximum storage in S1 [mm]
        # S2    - current storage in S2 [mm]
        # S2min - minimum storage in S2 [mm]
        # dt    - time step size [d]

        # this needs to be checked because in MATLAB version there is a min function that does not make sense to me
        Ep = torch.clamp(Ep, min=0.0)
        out = (S1 / S1max * Ep) * self.smoothThreshold_storage_logistic(S2, S2min, S1 / dt)
        out = torch.clamp(out, min=0.0)
        return out

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
    def multi_comp_parameter_bounds(self, params, num, args):
        nmul = args["nmul"]
        if num in args["static_params_list_prms"]:
            out_temp = (
                params[:, -1, num * nmul : (num + 1) * nmul]
                * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
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
                out_temp * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
            )

        else:  # dynamic
            out = (
                params[:, :, num * nmul : (num + 1) * nmul]
                * (args["marrmot_paramCalLst"][num][1] - args["marrmot_paramCalLst"][num][0])
                + args["marrmot_paramCalLst"][num][0]
            )
        return out

    def ODE_approx_IE(args, t, S1_old, S2_old, S3_old, S4_old, S5_old, S6_old, S7_old,
                      delta_S1, delta_S2, delta_S3, delta_S4, delta_S5, delta_S6, delta_S7):
        return S1_old

    def UH_gamma_notCum(self, a, b, lenF):
        # UH. a [time (same all time steps), batch, var]
        # a = torch.abs(a)
        if a.dim() == 2:
            m = a.shape
            a1 = a.repeat(1, lenF)
            b1 = b.repeat(1, lenF)
            alpha = F.relu(a1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.1
            beta = F.relu(b1).view(m[0], lenF, 1).permute(1, 0, 2) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], 1)
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            # ww = torch.cumsum(w, dim=0)
            # www = ww / ww.sum(0)  # scale to 1 for each UH
            www = w / w.sum(0)
        elif a.dim() == 3:
            m = a.shape
            a1 = a.repeat(1, 1, lenF)
            b1 = b.repeat(1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.1
            beta = F.relu(b1).view(m[0], m[1], lenF).permute(2, 0, 1) + 0.5
            # x = torch.arange(0.5, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            x = torch.linspace(0.01, 1, lenF).view(lenF, 1, 1).repeat(1, m[0], m[1])
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            # ww = torch.cumsum(w, dim=0)
            # www = ww / ww.sum(0)  # scale to 1 for each UH
            www = w/w.sum(0)
        elif a.dim() == 4:
            m = a.shape
            a1 = a.repeat(1, 1, 1, lenF)
            b1 = b.repeat(1, 1, 1, lenF)
            alpha = F.relu(a1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.1
            beta = F.relu(b1).view(m[0], m[1], m[2], lenF).permute(3, 0, 1, 2) + 0.5
            x = (
                torch.linspace(0.001, 20, lenF)
                .view(lenF, 1, 1, 1)
                .repeat(1, m[0], m[1], m[2])
            )
            if torch.cuda.is_available():
                x = x.cuda(a.device)
            # w = torch.pow(beta, alpha) * torch.pow(x, alpha - 1) * torch.exp((-1) * beta * x) / alpha.lgamma()
            denom = (alpha.lgamma().exp()) * torch.pow(beta, alpha)
            right = torch.exp((-1) * x / beta)
            mid = torch.pow(x, alpha - 1)
            w = 1 / denom * mid * right
            # ww = torch.cumsum(w, dim=0)
            # www = ww / ww.sum(0)  # scale to 1 for each UH
            www = w / w.sum(0)
        return www

    def UH_conv(self, x_sample, UH, bias, viewmode=1):
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
        # batch needs to be accommodated by channels and we make use of gr
        # ++++---------------------------------+
        #
        # oups
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # https://pytorch.org/docs/stable/nn.functional.html
        if UH.shape[1] == 1:
            x = x_sample[:, 0:1, :]
            mm = x.shape
            nb = mm[0]
            m = UH.shape[-1]
            padd = m - 1
            if viewmode == 1:
                xx = x.view([1, nb, mm[-1]])
                w = UH.view([nb, 1, m])
                groups = nb

                # y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
                # y = y[:, :, 0:-padd]

            x_sample1 = x_sample.permute(1, 0, 2)
            a = torch.arange(x_sample.shape[1])
            y = F.conv1d(
                x_sample1[a],
                torch.flip(w, [2]),
                groups=groups,
                padding=0,
                stride=1,
                bias=bias,
            )
            y = y.permute(1, 0, 2)
        elif UH.shape[1] > 1:
            w = torch.flip(UH, [2])
            y = x_sample * w
            y = y.sum(2)
            if bias is not None:
                y = y + bias
            y = y.unsqueeze(3)

        return y


    def forward(self, x, c_PRMS, params, args, Hamon_coef, warm_up=0, init=False, routing=True):
        NEARZERO = args["NEARZERO"]
        nmul = args["nmul"]
        vars = args["varT_PRMS"]
        vars_c_PRMS = args["varC_PRMS"]
        if warm_up > 0:
            with torch.no_grad():
                xinit = x[:, 0:warm_up, :]
                paramsinit = params[:, :warm_up, :]
                warm_up_model = prms_marrmot()
                Q_init, snow_storage, XIN_storage, RSTOR_storage, \
                    RECHR_storage, SMAV_storage, \
                    RES_storage, GW_storage = warm_up_model(xinit, c_PRMS, paramsinit, args, Hamon_coef,
                                                            warm_up=0, init=True)
        else:

            # snow storage
            snow_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            # interception storage
            XIN_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 0.75
            # RSTOR storage
            RSTOR_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            #storage in upper soil moisture zone
            RECHR_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            # storage in lower soil moisture zone
            SMAV_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            # storage in runoff reservoir
            RES_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2
            # GW storage
            GW_storage = torch.zeros(
                [x.shape[0], nmul], dtype=torch.float32, device=args["device"]
            ) + 2

        ## parameters for prms_marrmot. there are 18 parameters in it
        params = params[:, warm_up:, :]
        tt = self.param_bounds(params, 0, args, bounds=args["marrmot_paramCalLst"][0])
        ddf = self.param_bounds(params, 1, args, bounds=args["marrmot_paramCalLst"][1])
        alpha = self.param_bounds(params, 2, args, bounds=args["marrmot_paramCalLst"][2])  # can br found in attr
        beta = self.param_bounds(params, 3, args, bounds=args["marrmot_paramCalLst"][3])    # can be found in attr
        stor = self.param_bounds(params, 4, args, bounds=args["marrmot_paramCalLst"][4])
        retip = self.param_bounds(params, 5, args, bounds=args["marrmot_paramCalLst"][5])
        fscn = self.param_bounds(params, 6, args, bounds=args["marrmot_paramCalLst"][6])
        scx = self.param_bounds(params, 7, args, bounds=args["marrmot_paramCalLst"][7])
        scn = fscn * scx
        flz = self.param_bounds(params, 8, args, bounds=args["marrmot_paramCalLst"][8])
        stot = self.param_bounds(params, 9, args, bounds=args["marrmot_paramCalLst"][9])
        remx = (1 - flz) * stot
        smax = flz * stot
        cgw = self.param_bounds(params, 10, args, bounds=args["marrmot_paramCalLst"][10])
        resmax = self.param_bounds(params, 11, args, bounds=args["marrmot_paramCalLst"][11])
        k1 = self.param_bounds(params, 12, args, bounds=args["marrmot_paramCalLst"][12])
        k2 = self.param_bounds(params, 13, args, bounds=args["marrmot_paramCalLst"][13])
        k3 = self.param_bounds(params, 14, args, bounds=args["marrmot_paramCalLst"][14])
        k4 = self.param_bounds(params, 15, args, bounds=args["marrmot_paramCalLst"][15])
        k5 = self.param_bounds(params, 16, args, bounds=args["marrmot_paramCalLst"][16])
        k6 = self.param_bounds(params, 17, args, bounds=args["marrmot_paramCalLst"][17]) * 0.0 # because we don't have any sink in the watersheds! Do we?
        hamon_coef = self.param_bounds(params, 18, args, bounds=args["marrmot_paramCalLst"][18])

        if routing == True:
            tempa = self.param_bounds(params, 19, args, bounds=args["conv_PRMS"][0])
            tempb = self.param_bounds(params, 20, args, bounds=args["conv_PRMS"][1])
            # tempa = self.multi_comp_parameter_bounds(params, 18, args)
            # tempb = self.multi_comp_parameter_bounds(params, 19, args)
        #################
        # inputs
        Precip = (
            x[:, warm_up:, vars.index("prcp(mm/day)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Tmaxf = x[:, warm_up:, vars.index("tmax(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        Tminf = x[:, warm_up:, vars.index("tmin(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        mean_air_temp = (Tmaxf + Tminf) / 2
        dayl = (
            x[:, warm_up:, vars.index("dayl(s)")].unsqueeze(-1).repeat(1, 1, nmul)
        )
        Ngrid, Ndays = Precip.shape[0], Precip.shape[1]
        # PET = (
        #     x[:, warm_up:, vars.index("pet_nldas")].unsqueeze(-1).repeat(1, 1, nmul)
        # )
        # t_monthly = x[:, warm_up:, vars.index("t_monthly(C)")].unsqueeze(-1).repeat(1, 1, nmul)
        # it is poorly coded. need to fix it later.
        # hamon_coef = self.param_bounds(Hamon_coef, 0, args, bounds=args["SNTEMP_paramCalLst"][5])
        PET = get_potet(
            args=args, mean_air_temp=mean_air_temp, dayl=dayl, hamon_coef=hamon_coef
        ) * 86400 * 1000  # converting m/sec to mm/day

        # initialize the Q_sim and other fluxes
        Q_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        sas_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        sro_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        bas_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        ras_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        snk_sim = torch.zeros(PET.shape, dtype=torch.float32, device=args["device"])
        for t in range(Ndays):
            delta_t = 1 # timestep (day)
            P = Precip[:, t, :]
            Ep = PET[:, t, :]
            T = mean_air_temp[:, t, :]

            # fluxes
            flux_ps = torch.mul(P, (T <= tt[:, t, :]).type(torch.float32))
            flux_pr = torch.mul(P, (T > tt[:, t, :]).type(torch.float32))
            snow_storage = snow_storage + flux_ps
            flux_m = ddf[:, t, :] * (T - tt[:, t, :])
            flux_m = torch.min(flux_m, snow_storage/delta_t)
            flux_m = torch.clamp(flux_m, min=0.0)
            snow_storage = snow_storage - flux_m
            snow_storage = torch.clamp(snow_storage, min=NEARZERO)  # to prevent NaN  gradient, it is set to NEARZERO

            flux_pim = flux_pr * (1 - beta[:, t, :])
            flux_psm = flux_pr * beta[:, t, :]
            flux_pby = flux_psm * (1 - alpha[:, t, :])
            flux_pin = flux_psm * alpha[:, t, :]

            XIN_storage = XIN_storage + flux_pin
            flux_ptf = XIN_storage - stor[:, t, :]
            flux_ptf = torch.clamp(flux_ptf, min=0.0)
            XIN_storage = torch.clamp(XIN_storage - flux_ptf, min=NEARZERO)
            evap_max_in = Ep * beta[:, t, :]   # only can happen in pervious area
            flux_ein = torch.min(evap_max_in, XIN_storage/delta_t)
            XIN_storage = torch.clamp(XIN_storage - flux_ein, min=NEARZERO)


            flux_mim = flux_m * (1 - beta[:, t, :])
            flux_msm = flux_m * beta[:, t, :]
            RSTOR_storage = RSTOR_storage + flux_mim + flux_pim
            flux_sas = RSTOR_storage - retip[:, t, :]
            flux_sas = torch.clamp(flux_sas, min=0.0)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_sas, min=NEARZERO)
            evap_max_im = (1 - beta[:, t, :]) * Ep
            flux_eim = torch.min(evap_max_im, RSTOR_storage / delta_t)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_eim, min=NEARZERO)


            sro_lin_ratio = scn[:, t, :] + (scx[:, t, :] - scn[:, t, :]) * (RECHR_storage / remx[:, t, :])
            sro_lin_ratio = torch.clamp(sro_lin_ratio, min=0.0, max=1.0)
            flux_sro = sro_lin_ratio * (flux_msm + flux_ptf + flux_pby)
            flux_inf = torch.clamp(flux_msm + flux_ptf + flux_pby - flux_sro, min=0.0)
            RECHR_storage = RECHR_storage + flux_inf
            flux_pc = RECHR_storage - remx[:, t, :]
            flux_pc = torch.clamp(flux_pc, min=0.0)
            RECHR_storage = RECHR_storage - flux_pc
            evap_max_a = (RECHR_storage / remx[:, t, :]) * (Ep - flux_ein - flux_eim)
            evap_max_a = torch.clamp(evap_max_a, min=0.0)
            flux_ea = torch.min(evap_max_a, RECHR_storage / delta_t)
            RECHR_storage = torch.clamp(RECHR_storage - flux_ea, min=NEARZERO)

            SMAV_storage = SMAV_storage + flux_pc
            flux_excs = SMAV_storage - smax[:, t, :]
            flux_excs = torch.clamp(flux_excs, min=0.0)
            SMAV_storage = SMAV_storage - flux_excs
            transp = torch.where(RECHR_storage < (Ep - flux_ein - flux_eim),
                                 (SMAV_storage/smax[:, t, :]) * (Ep - flux_ein - flux_eim - flux_ea),
                                 torch.zeros(flux_excs.shape, dtype=torch.float32, device=args["device"]))
            transp = torch.clamp(transp, min=0.0)    # in case Ep - flux_ein - flux_eim - flux_ea was negative
            SMAV_storage = torch.clamp(SMAV_storage - transp, min=NEARZERO)

            flux_sep = torch.min(cgw[:, t, :], flux_excs)
            flux_qres = torch.clamp(flux_excs - flux_sep, min=0.0)

            RES_storage = RES_storage + flux_qres
            flux_ras = k3[:, t, :] * RES_storage + k4[:, t, :] * (RES_storage ** 2)
            flux_ras = torch.min(flux_ras, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_ras, min=NEARZERO)
            # RES_excess = RES_storage - resmax[:, t, :]   # if there is still overflow, it happend in discrete version
            # RES_excess = torch.clamp(RES_excess, min=0.0)
            # flux_ras = flux_ras + RES_excess
            # RES_storage = torch.clamp(RES_storage - RES_excess, min=NEARZERO)

            flux_gad = k1[:, t, :] * ((RES_storage / resmax[:, t, :]) ** k2[:, t, :])
            flux_gad = torch.min(flux_gad, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_gad, min=NEARZERO)

            GW_storage = GW_storage + flux_gad + flux_sep
            flux_bas = k5[:, t, :] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_bas, min=NEARZERO)
            flux_snk = k6[:, t, :] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_snk, min=NEARZERO)

            Q_sim[:, t, :] = (flux_sas + flux_sro + flux_bas + flux_ras)
            sas_sim[:, t, :] = flux_sas
            sro_sim[:, t, :] = flux_sro
            bas_sim[:, t, :] = flux_bas
            ras_sim[:, t, :] = flux_ras
            snk_sim[:, t, :] = flux_snk

        if routing == True:
            # routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            # routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            # UH = self.UH_gamma_notCum(tempa.unsqueeze(-1), tempb.unsqueeze(-1), lenF=15)  # lenF: folter
            # rf = Q_sim.unsqueeze(-1).permute([0, 1, 3, 2])
            # UH = UH.permute(1, 2, 0, 3)  # dim: gage*var*time
            # Qsrout = self.UH_conv(rf, UH, bias=None).squeeze(-1)
            tempa_new = tempa.mean(-1, keepdim=True).permute(1,0,2)
            tempb_new = tempb.mean(-1, keepdim=True).permute(1,0,2)
            # Q_sim_new = Q_sim.mean(-1, keepdim=True).permute(1,0,2)
            UH = UH_gamma(tempa_new, tempb_new, lenF=15)  # lenF: folter
            rf = Q_sim.mean(-1, keepdim=True).permute([0, 2, 1])  # dim:gage*var*time
            UH = UH.permute([1, 2, 0])  # dim: gage*var*time
            Qsrout = UH_conv(rf, UH).permute([0, 2, 1])

            rf_sas = sas_sim.mean(-1, keepdim=True).permute([0, 2, 1])
            Qsas_rout = UH_conv(rf_sas, UH).permute([0, 2, 1])

            rf_sro = sro_sim.mean(-1, keepdim=True).permute([0, 2, 1])
            Qsro_rout = UH_conv(rf_sro, UH).permute([0, 2, 1])

            rf_ras = ras_sim.mean(-1, keepdim=True).permute([0, 2, 1])
            Qras_rout = UH_conv(rf_ras, UH).permute([0, 2, 1])

            rf_bas = bas_sim.mean(-1, keepdim=True).permute([0, 2, 1])
            Qbas_rout = UH_conv(rf_bas, UH).permute([0, 2, 1])

        else:
            Qsrout = Q_sim.mean(-1, keepdim=True)
            Qsas_rout = sas_sim.mean(-1, keepdim=True)
            Qsro_rout = sro_sim.mean(-1, keepdim=True)
            Qbas_rout = bas_sim.mean(-1, keepdim=True)
            Qras_rout = ras_sim.mean(-1, keepdim=True)


        if init:  # means we are in warm up
            return Qsrout, snow_storage, XIN_storage, RSTOR_storage, \
                RECHR_storage, SMAV_storage, RES_storage, GW_storage
        else:
            Qall = torch.cat((
                # Qsrout,
                Qsas_rout + Qsro_rout + Qbas_rout + Qras_rout,
                Qsas_rout,
                Qsro_rout,
                Qbas_rout,
                Qras_rout,
                torch.mean(snk_sim, -1).unsqueeze(-1)), dim=-1
                    )
            return torch.clamp(Qall, min=0.0)

def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    lenF = min(a.shape[0], lenF)
    w = torch.zeros([lenF, m[1],m[2]])
    aa = F.relu(a[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.1 # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF,:,:]).view([lenF, m[1],m[2]])+0.5 # minimum 0.5
    t = torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp())*(theta**aa)
    mid= t**(aa-1)
    right=torch.exp(-t/theta)
    w = 1/denom*mid*right
    w = w/w.sum(0)  # scale to 1 for each UH

    return w

def UH_conv(x,UH,viewmode=1):
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

    mm= x.shape; nb=mm[0]
    m = UH.shape[-1]
    padd = m-1
    if viewmode==1:
        xx = x.view([1,nb,mm[-1]])
        w = UH.view([nb,1,m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
    if padd != 0:
        y = y[:, :, 0:-padd]
    return y.view(mm)
















class Farshid_NRODEsolver(torch.nn.Module):
    ###This is a nonlinear solver using Newton Raphson method to solve ODEs
    ### Input y0, g (RHS), t_start, t_end

    def __init__(self, settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(NRODESolver, self).__init__()
        self.settings = settings
        # self.y0 = y0
        # self.g = g
        # self.dt = dt


    def forward(self, y0,dt,t ):
        ALPHA = 1e-4; # criteria for decrease
        MIN_LAMBDA = 0.1;  # min lambda
        MAX_LAMBDA = 0.5;  # max lambda
        y = y0; # initial guess
        self.y0 = y0
        self.dt = dt
        F = self.error_func(y,t); # evaluate initial guess  bs*ny
        bs, ny = y.shape
        jac = torch.autograd.functional.jacobian(self.error_func, (y,t))  ##bs*ny*ny
        jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
        jac_new = jac_new.permute(2, 0, 1)

        if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
            exitflag = -1; # matrix may be singular
        else:
            exitflag = 1; # normal exit


        resnorm = torch.linalg.norm(F, float('inf'),dim= [1]) # calculate norm of the residuals
        resnorm0 = 100*resnorm;
        dy = torch.zeros(y.shape).to(y0); # dummy values
        ##%% solver
        Niter = 0; # start counter
        lambda_ = torch.tensor(1.0).to(y0) # backtracking
        while ((torch.max(resnorm)>self.settings["TolFun"] or lambda_<1) and exitflag>=0 and Niter<=self.settings["MaxIter"]):
            if lambda_==1:
                ### Newton-Raphson solver
                Niter = Niter+1; ## increment counter
                ### update Jacobian, only if necessary
                if torch.max(resnorm/resnorm0) > 0.2:
                    jac = torch.autograd.functional.jacobian(self.error_func, (y,t))
                    jac_new = torch.diagonal(jac[0], offset=0, dim1=0, dim2=2)
                    jac_new = jac_new.permute(2, 0, 1) ## bs*ny*ny

                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;## % matrix may be singular
                        break

                if torch.min(1/torch.linalg.cond(jac_new,p=1)) <= 2.2204e-16:
                    dy = torch.bmm(torch.linalg.pinv(jac_new) , -F.unsqueeze(-1)).squeeze(-1); ## bs*ny*ny  , bs*ny*1 = bs*ny
                else:
                    dy = -torch.linalg.lstsq(jac_new, F).solution;
                g = torch.bmm(F.unsqueeze(1),jac_new);   #%star; % gradient of resnorm  bs*1*ny, bs*ny*ny = bs*1*ny
                slope = torch.bmm(g,dy.unsqueeze(-1)).squeeze();   #%_star; % slope of gradient  bs*1*ny,bs*ny*1
                fold_obj = torch.bmm(F.unsqueeze(1),F.unsqueeze(-1)).squeeze(); ###% objective function
                yold = y; ##% initial value
                lambda_min = self.settings["TolX"]/torch.max(abs(dy)/torch.maximum(abs(yold), torch.tensor(1.0)));
            if lambda_<lambda_min:
                exitflag = 2; ##% x is too close to XOLD
                break
            elif torch.isnan(dy).any() or torch.isinf(dy).any():
                exitflag = -1; ##% matrix may be singular
                break
            y = yold+dy*lambda_;## % next guess
            F = self.error_func(y,t); ## % evaluate this guess
            f_obj = torch.bmm(F.unsqueeze(1),F.unsqueeze(-1)).squeeze(); ###% new objective function
            ###%% check for convergence
            lambda1 = lambda_; ###% save previous lambda

            if torch.any(f_obj>fold_obj+ALPHA*lambda_*slope):
                if lambda_==1:
                    lambda_ = torch.min(-slope/2.0/(f_obj-fold_obj-slope)); ##% calculate lambda
                else:

                    A = 1/(lambda1 - lambda2); ##Scalar
                    B = torch.stack([torch.stack([1.0/lambda1**2.0,-1.0/lambda2**2.0]),torch.stack([-lambda2/lambda1**2.0,lambda1/lambda2**2.0])]); ##2*2
                    C = torch.stack([f_obj-fold_obj-lambda1*slope,f2_obj-fold_obj-lambda2*slope]);  ##2*1
                    a = (A*B@C)[0,:] ;
                    b = (A*B@C)[1,:] ;

                    if torch.all(a==0):
                        lambda_tmp = -slope/2/b;
                    else:
                        discriminant = b**2 - 3*a*slope;
                        if torch.any(discriminant<0):
                            lambda_tmp = MAX_LAMBDA*lambda1;
                        elif torch.any(b<=0):
                            lambda_tmp = (-b+torch.sqrt(discriminant))/3/a;
                        else:
                            lambda_tmp = -slope/(b+torch.sqrt(discriminant));


                    lambda_ = torch.min(torch.minimum(lambda_tmp,torch.tensor(MAX_LAMBDA*lambda1))); #% minimum step length

            elif torch.isnan(f_obj).any() or torch.isinf(f_obj).any():
               ## % limit undefined evaluation or overflow
                lambda_ = MAX_LAMBDA*lambda1;
            else:
                lambda_ = torch.tensor(1.0).to(y0); ### % fraction of Newton step

            if lambda_<1:
                lambda2 = lambda1; f2_obj = f_obj; ##% save 2nd most previous value
                lambda_ = torch.maximum(lambda_,torch.tensor(MIN_LAMBDA*lambda1)); ###% minimum step length
                continue

            resnorm0 = resnorm; ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'),dim= [1]); ###% calculate new resnorm
        print("day ",t.detach().cpu().numpy(),"Iteration ",Niter,"Flag ",exitflag)
        return y, F, exitflag

    def error_func(self,y,t):

        delta_S,_ = self.g(t,y);  ##bs*ny
        err = (y - self.y0)/self.dt - delta_S;
        return err  ##bs*ny
