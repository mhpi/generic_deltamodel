import torch
import torch.nn as nn
import math

'''to have the model to be both best in temporal test and PUR test:
The discharge is
(1-Lambda) * StefaLand_finetune + lambda * SL
Lambda is a function, e.g., Gaussian kernel, describing how far we are far training gages with a hyperparameter to determine how much we want to weigh StefaLand. SL is a supervised learning method that is best for temporal test.
Lambda=gaussian_kernel(eta, distance) as distance goes large this kernel fades to 0. It could be a radial basis function. Eta is a hyperparameter.
We can use validation data to determine eta '''

def gaussian_kernel(dist_km: torch.Tensor, eta_km: float):
    # dist_km: [B] or [B,1] — station-wise distance to nearest training gage
    return torch.exp(-(dist_km**2) / (2.0 * (eta_km**2)))

class BlendedDischarge(nn.Module):
    """
    Wraps two models
    Blends their time-series outputs with a spatially varying lambda.
    Expects batch_data_dict to carry a per-sample distance tensor 'min_dist_km'.
    """
    def __init__(self, stefa_model: nn.Module, sl_model: nn.Module, eta_km: float, lambda_min=0.0, lambda_max=1.0):
        super().__init__()
        self.stefa = stefa_model
        self.sl    = sl_model
        self.eta_km = float(eta_km)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)

    @torch.no_grad()
    def forward(self, batch_data_dict, is_mask: bool = False):
        # Get predictions from each model (match your exp_pretrain2 usage)
        out_stefa = self.stefa(batch_data_dict, is_mask=is_mask)
        out_sl    = self.sl(batch_data_dict,    is_mask=is_mask)

        y_stefa = out_stefa['outputs_time_series']  # [B, T, F]
        y_sl    = out_sl['outputs_time_series']     # [B, T, F]

        # distance per-sample (station) must be provided in the batch
        # shape [B] or [B,1] or [B,T] (we'll broadcast); unit: km
        d = batch_data_dict.get('min_dist_km', None)
        if d is None:
            raise RuntimeError("batch_data_dict['min_dist_km'] is required for spatial blending.")

        if d.dim() == 1:
            d = d[:, None, None]         # [B,1,1]
        elif d.dim() == 2:
            d = d[:, :, None]            # [B,T,1] or [B,1,?] -> we assume [B,T] becomes [B,T,1]

        lam = gaussian_kernel(d, self.eta_km)       # same shape as d
        lam = lam.clamp_(self.lambda_min, self.lambda_max)

        # Broadcast across feature dim
        if lam.dim() == 3 and lam.shape[-1] == 1:
            lam = lam.expand_as(y_stefa)

        y_blend = (1.0 - lam) * y_stefa + lam * y_sl

        out = dict(out_stefa)  
        out['outputs_time_series'] = y_blend
        out['lambda'] = lam
        out['outputs_time_series_stefa'] = y_stefa
        out['outputs_time_series_sl'] = y_sl
        return out
