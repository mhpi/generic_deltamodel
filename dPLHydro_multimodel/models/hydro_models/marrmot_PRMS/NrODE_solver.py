import torch

# from functorch import vmap, jacrev, jacfwd, vjp



class Farshid_NRODEsolver(torch.nn.Module):
    ###This is a nonlinear solver using Newton Raphson method to solve ODEs
    ### Input y0, g (RHS), t_start, t_end

    def __init__(self, settings={'TolX': 1e-12, 'TolFun': 1e-6, 'MaxIter': 1000}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(Farshid_NRODEsolver, self).__init__()
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