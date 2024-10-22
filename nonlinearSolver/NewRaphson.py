import torch
import torch.nn as nn
from HydroModels.Jacobian import batchJacobian_AD_new
import time
class NRODESolver(nn.Module):
    ###This is a nonlinear solver using Newton Raphson method to solve ODEs
    ### Input y0, g (RHS), t_start, t_end

    def __init__(self, y0, g,dt,settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 50}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(NRODESolver, self).__init__()
        self.settings = settings
        self.y0 = y0
        self.g = g
        self.dt = dt
        self.t0 = time.time()


    def forward(self, y0,dt,t ):

        y = y0; # initial guess
        self.y0 = y0
        self.dt = dt

        y_ = y.detach().requires_grad_()

        F = self.error_func(y_,t); # evaluate initial guess  bs*ny


        jac_new = batchJacobian_AD_new(y_,F,graphed=True,doSqueeze=True)

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


                    y_ = y.detach().requires_grad_()

                    F = self.error_func(y_,t); # evaluate initial guess  bs*ny


                    jac_new = batchJacobian_AD_new(y_,F,graphed=True,doSqueeze=True)
                    if torch.isnan(jac_new).any() or torch.isinf(jac_new).any():
                        exitflag = -1;## % matrix may be singular
                        break


                dy = torch.bmm(torch.linalg.pinv(jac_new) , -F.unsqueeze(-1)).squeeze(-1); ## bs*ny*ny  , bs*ny*1 = bs*ny

                yold = y; ##% initial value

            y = yold+dy*lambda_;## % next guess
            F = self.error_func(y,t); ## % evaluate this guess


            resnorm0 = resnorm; ##% old resnorm
            resnorm = torch.linalg.norm(F, float('inf'),dim= [1]); ###% calculate new resnorm
        now = time.time()
        print("day ",t.detach().cpu().numpy(),"Iteration ",Niter,"Flag ",exitflag, "    time: ", now - self.t0)
        self.t0 = now
        return y, F, Niter

    def error_func(self,y,t):

        delta_S,_ = self.g(t,y);  ##bs*ny
        err = (y - self.y0)/self.dt - delta_S;
        return err  ##bs*ny
