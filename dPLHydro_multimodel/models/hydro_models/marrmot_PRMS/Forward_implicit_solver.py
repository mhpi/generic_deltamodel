# import pydevd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.set_device(0)
device = torch.device("cuda")
dtype=torch.double


class NRODESolver(nn.Module):
    ###This is a nonlinear solver using Newton Raphson method to solve ODEs
    ### Input y0, g (RHS), t_start, t_end

    def __init__(self, y0, g, dt, settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(NRODESolver, self).__init__()
        self.settings = settings
        self.y0 = y0
        self.g = g
        self.dt = dt


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


class RHS(nn.Module):
    def __init__(self, theta,delta_t,climate_data):
        super().__init__()

        self.register_parameter("theta", theta)
        self.delta_t = delta_t
        self.climate_data = climate_data

    def forward(self, t, y):
        ##parameters
        smax   = self.theta[0];    # % Maximum soil moisture storage     [mm],
        b      = self.theta[1];    # % Soil depth distribution parameter [-]
        a      = self.theta[2];    # % Runoff distribution fraction [-]
        kf     = self.theta[3];    # % Fast runoff coefficient [d-1]
        ks     = self.theta[4];   #  % Slow runoff coefficient [d-1]

        ##% stores
        S1 = y[:,0];
        S2 = y[:,1];
        S3 = y[:,2];
        S4 = y[:,3];
        S5 = y[:,4];
        dS = torch.zeros(y.shape).to(y)
        fluxes = torch.zeros((y.shape[0],8)).to(y)

        climate_in = self.climate_data[int(t),:,:];   ##% climate at this step
        P  = climate_in[:,0];
        Ep = climate_in[:,1];
        T  = climate_in[:,2];

        ##% fluxes functions
        flux_ea  = self.evap_7(S1,smax,Ep,self.delta_t);
        flux_pe  = self.saturation_2(S1,smax,b,P);
        flux_pf  = self.split_1(a,flux_pe);
        flux_ps  = self.split_1(1-a,flux_pe);
        flux_qf1 = self.baseflow_1(kf,S2);
        flux_qf2 = self.baseflow_1(kf,S3);
        flux_qf3 = self.baseflow_1(kf,S4);
        flux_qs  = self.baseflow_1(ks,S5);

        ##% stores ODEs
        dS[:,0] = P - flux_ea - flux_pe;
        dS[:,1] = flux_pf - flux_qf1;
        dS[:,2] = flux_qf1 - flux_qf2;
        dS[:,3] = flux_qf2 - flux_qf3;
        dS[:,4] = flux_ps - flux_qs;
        fluxes[:,0] =flux_ea
        fluxes[:,1] =flux_pe
        fluxes[:,2] =flux_pf
        fluxes[:,3] =flux_ps
        fluxes[:,4] =flux_qf1
        fluxes[:,5] =flux_qf2
        fluxes[:,6] =flux_qf3
        fluxes[:,7] =flux_qs

        return dS,fluxes

    def evap_7(self,S,Smax,Ep,dt):
        return torch.minimum(S/Smax*Ep,S/dt);

    def saturation_2(self,S,Smax,p1,In):
        return (1- torch.minimum(torch.tensor(1.0),torch.maximum(torch.tensor(0.0),(1.0-S/Smax)))**p1)*In;

    def split_1(self,p1,In):
        return p1*In;

    def baseflow_1(self,p1,S):
        return p1*S;




class NRBacksolveFunction(torch.autograd.Function):


    def forward(
        ctx,
        y0,
        theta,
        delta_t,
        input,
        nflux
    ):

        bs,ny = y0.shape
        rho = input.shape[0]
        with torch.no_grad():
            g = RHS(theta,delta_t,input)
            NR = NRODESolver(y0, g, delta_t)
            ySolution = torch.zeros((rho,bs,ny),device = device,dtype = dtype)

            fluxSolution = torch.zeros((rho,bs,8),device = device,dtype = dtype)

            Residual = torch.zeros((rho,bs,ny),device = device,dtype = dtype)
            NRFlag = torch.zeros((rho),device = device,dtype = dtype)
            ySolution[0,:,:] = y0
            with torch.no_grad():
                for day in range(rho):
                    if day == 0:
                        yold = ySolution[0,:,:]
                    else:
                        yold = ySolution[day-1,:,:]
                    t = torch.tensor(day).to(yold)
                    yNew, F, exitflag = NR(yold,delta_t,t)

                    dy, flux = g(t,yNew);

                    ySolution[day,:,:]  = yold + dy*delta_t
                    Residual[day,:,:]  =  F
                    fluxSolution[day,:,:]  =  flux*delta_t
                    NRFlag[day]  =   exitflag

        ctx.save_for_backward( ySolution, theta,delta_t)


        return ySolution, fluxSolution, Residual,NRFlag


def goforward():

    input_theta = [ 35,                                                 #    % Soil moisture depth [mm]
                    3.7,                                                 #  % Soil depth distribution parameter [-]
                    0.4,                                                 # % Fraction of soil moisture excess that goes to fast runoff [-]
                    0.25,                                                #  % Runoff coefficient of the upper three stores [d-1]
                    0.01]                                               # % Runoff coefficient of the lower store [d-1]

    theta  = nn.Parameter(
            torch.tensor(
                input_theta,
                dtype=torch.double,
                requires_grad=True,
                device =device,
            )
        )

    input_s00  = [15,                                              # % Initial soil moisture storage [mm]
                 7,                                          #     % Initial fast flow 1 storage [mm]
                 3,                                          #     % Initial fast flow 2 storage [mm]
                 8,                                          #     % Initial fast flow 3 storage [mm]
                 22]                                          #   % Initial slow flow storage [mm]

    input_s01 = [ 13.4425,   9.5697,  14.3160,  13.7041, 290.4901]
    input_s02 = [ 11.2545,   8.2934,   9.2703,  10.9482, 306.2097]
    input_s0 = np.concatenate((np.expand_dims(np.array(input_s00),axis = 0),np.expand_dims(np.array(input_s01),axis = 0),np.expand_dims(np.array(input_s02),axis = 0)),axis = 0)

    bs = len(input_s0)
    ny = len(input_s00)
    delta_t  = torch.tensor(1.0).to(device = device,dtype = dtype)
    data = pd.read_csv("/mnt/sdb/yxs275/torchode/MARRMoT_example_data.csv",  header=None, )
    to_day = 100

    climate_data = torch.tensor(data.to_numpy()[:bs*to_day,:3],device = device,dtype = dtype).view(bs,to_day,3) ##bs*nt*ny

    climate_data = climate_data.permute(1, 0, 2) ##nt*bs*ny
    y0 = torch.tensor(input_s0,device = device,dtype = dtype)  #bs*ny
    nflux = 8

    ySolution, fluxSolution, Residual,NRFlag = NRBacksolveFunction.apply( y0, theta, delta_t, climate_data, nflux)

    prediction = (fluxSolution[:,:,6]+fluxSolution[:,:,7]).permute([1,0]).detach().cpu().numpy().flatten()
    time = pd.date_range(f'{1989}-01-01',f'{1992}-12-31', freq='d')
    plt.plot_date(time[:len(prediction)], prediction,"k",label = "Prediction")
    plt.plot_date(time[:len(prediction)], data.to_numpy()[:len(prediction),3],"r",label = "Observation")
    plt.ylabel('Streamflow [mm/d]')
    plt.legend()
    plt.show(block=True)

# goforward()

