import warnings
from typing import Any, Optional, Tuple

import logging

import numpy as np
import numpy.typing as npt
import scipy.stats
from pydantic import BaseModel, ConfigDict, model_validator
import scipy.stats as stats

log = logging.getLogger()


class Metrics(BaseModel):
    """Metrics for model evaluation.
    
    Adapted from Yalan Song, Tadd Bindas, Farshid Rahmani.
    """
    def __init__(
        self,
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        ) -> None:
        super().__init__(pred=pred, target=target)

    def model_post_init(self, __context: Any):
        self.bias = self._bias(self.pred, self.target)
        self.bias_rel = self._bias_rel(self.pred, self.target)

        self.rmse = self._rmse(self.pred, self.target)
        self.rmse_ub = self._rmse_ub(self.pred, self.target)
        self.rmse_fdc = self._rmse_fdc(self.pred, self.target)

        self.corr = np.full(self.ngrid, np.nan)
        self.corr_spearman = np.full(self.ngrid, np.nan)
        self.r2 = np.full(self.ngrid, np.nan)
        self.nse = np.full(self.ngrid, np.nan)

        self.flv = np.full(self.ngrid, np.nan)
        self.fhv = np.full(self.ngrid, np.nan)
        self.pbias = np.full(self.ngrid, np.nan)
        self.pbias_mid = np.full(self.ngrid, np.nan)

        self.flv_abs = np.full(self.ngrid, np.nan)
        self.fhv_abs = np.full(self.ngrid, np.nan)
        self.pbias_abs = np.full(self.ngrid, np.nan)
        self.pbias_abs_mid = np.full(self.ngrid, np.nan)

        self.kge = np.full(self.ngrid, np.nan)
        self.kge_12 = np.full(self.ngrid, np.nan)

        self.rmse_low = np.full(self.ngrid, np.nan)
        self.rmse_mid = np.full(self.ngrid, np.nan)
        self.rmse_high = np.full(self.ngrid, np.nan)

        self.d_max = np.full(self.ngrid, np.nan)
        self.d_max_rel = np.full(self.ngrid, np.nan)

        for i in range(0, self.ngrid):
            _pred = self.pred[i]
            _target = self.target[i]
            idx = np.where(
                np.logical_and(~np.isnan(_pred), ~np.isnan(_target))
            )[0]

            if idx.shape[0] > 0:
                pred = _pred[idx]
                target = _target[idx]

                pred_sort = np.sort(pred)
                target_sort = np.sort(target)
                index_low = round(0.3 * pred_sort.shape[0])
                index_high = round(0.98 * pred_sort.shape[0])

                low_pred = pred_sort[:index_low]
                mid_pred = pred_sort[index_low:index_high]
                high_pred = pred_sort[index_high:]

                low_target = target_sort[:index_low]
                mid_target = target_sort[index_low:index_high]
                high_target = target_sort[index_high:]
                
                self.flv[i] = self._bias_percent(low_pred, low_target, offset=0.0001)
                self.fhv[i] = self._bias_percent(high_pred, high_target)
                self.pbias[i] = self._bias_percent(pred, target)
                self.pbias_mid[i] = self._bias_percent(mid_pred, mid_target)

                self.flv_abs[i] = self._bias_percent(low_pred, low_target, offset=0.0001)
                self.fhv_abs[i] = self._bias_percent(high_pred, high_target)
                self.pbias_abs[i] = self._bias_percent(pred, target)
                self.pbias_abs_mid[i] = self._bias_percent(mid_pred, mid_target)
                
                self.rmse_low[i] = self._rmse(low_pred, low_target, axis=0)
                self.rmse_mid[i] = self._rmse(mid_pred, mid_target, axis=0)
                self.rmse_high[i] = self._rmse(high_pred, high_target, axis=0)

                target_max = np.nanmax(target)
                pred_max = self.pred_max(pred, lb=10, ub=11)

                self.d_max[i] = pred_max - target_max
                self.d_max_rel[i] = (pred_max - target_max) / target_max * 100

                if idx.shape[0] > 1:
                    # At least two points needed for correlation.
                    self.corr[i] = self._corr(pred, target)
                    self.corr_spearman[i] = self._corr_spearman(pred, target)

                    _pred_mean = pred.mean()
                    _target_mean = target.mean()
                    _pred_std = np.std(pred)
                    _target_std = np.std(target)
                    self.kge[i] = self._kge(
                        _pred_mean, _target_mean, _pred_std, _target_std, self.corr[i]
                    )
                    self.kge_12[i] = self._kge(
                        _pred_mean, _target_mean, _pred_std, _target_std, self.corr[i]
                    )
                    
                    self.nse[i] = self.r2[i] = self._nse_r2(pred, target, _target_mean)

        return super().model_post_init(__context)

    @property
    def ngrid(self) -> int:
        """Calculate number of items in grid."""
        return self.pred.shape[0]
    
    @property
    def nt(self) -> int:
        """Calculate number of time steps."""
        return self.pred.shape[1]

    def tile_mean(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate mean of target
        """
        return np.tile(np.nanmean(data, axis=1), (self.nt, 1)).transpose()
    
    @staticmethod
    def _bias(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        """Calculate bias."""
        return np.nanmean(abs(pred - target)/(target + 0.0001), axis=1)

    @staticmethod
    def _bias_rel(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        """Calculate relative bias.
        
        Don't sum together because if NaNs are present at different idx,
        the sum will be off.
        """
        pred_sum = np.nansum(pred, axis=1)
        target_sum = np.nansum(target, axis=1)
        return (pred_sum - target_sum) / target_sum

    @staticmethod
    def _bias_percent(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        offset: float = 0.0,
        ) -> np.float64:
        """Calculate percent bias."""
        p_bias = np.sum(pred - target) / (np.sum(target) + offset) * 100
        return p_bias
    
    @staticmethod
    def _rmse(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        axis: Optional[int] = 1,
        ) -> npt.NDArray[np.float64]:
        """Calculate root mean square error."""
        return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))
    
    def _rmse_ub(
        self,
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        """Calculate unbiased root mean square error."""
        pred_mean = self.tile_mean(self.pred)
        target_mean = self.tile_mean(self.target)
        pred_anom = self.pred - pred_mean
        target_anom = self.target - target_mean
        return self._rmse(pred_anom, target_anom)
    
    def _rmse_fdc(
        self,
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        """Calculate flow duration curve root mean square error."""
        pred_fdc = self._calc_fdc(pred)
        target_fdc = self._calc_fdc(target)
        return self._rmse(pred_fdc, target_fdc)

    def _calc_fdc(
        self,
        data: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
        """Calculate flow duration curve for each grid point."""
        fdc_100 = np.full([self.ngrid, 100], np.nan)
        for i in range(self.ngrid):
            data_slice = data[i]
            non_nan_data_slice = data_slice[~np.isnan(data_slice)]
            
            if len(non_nan_data_slice) == 0:
                non_nan_data_slice = np.full(self.nt, 0)
           
            sorted_data = np.sort(non_nan_data_slice)[::-1]
            Nlen = len(non_nan_data_slice)
            ind = (np.arange(100) / 100 * Nlen).astype(int)
            fdc_flow = sorted_data[ind]
            
            if len(fdc_flow) != 100:
                raise Exception("Unknown assimilation variable")
            else:
                fdc_100[i] = fdc_flow
        
        return fdc_100

    @staticmethod
    def pred_max(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        lb: int = 0,
        ub: int = 10,
        ) -> np.float64:
        """Calculate maximum value of predictions."""
        idx_max = np.nanargmax(target)
        if (idx_max < lb):
            lb = idx_max
        elif (ub > len(pred) - idx_max):
            ub = len(pred) - idx_max
        else:
            pass
        return np.nanmax(pred[idx_max - lb:idx_max + ub])
    
    @staticmethod
    def _corr(
        pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.float64]:
        """Calculate correlation."""
        corr = stats.pearsonr(pred, target)[0]
        return corr

    @staticmethod
    def _corr_spearman(
        pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
        ) -> np.float64:
        """Calculate Spearman correlation."""
        corr_spearman = stats.spearmanr(pred, target)[0]
        return corr_spearman
    
    @staticmethod
    def _kge(
        pred_mean: np.float64,
        target_mean: np.float64,
        pred_std: np.float64,
        target_std: np.float64,
        corr: np.float64,
        ) -> npt.NDArray[np.float64]:
        """Calculate Kling-Gupta Efficiency (KGE)."""
        kge = 1 - np.sqrt(
            (corr - 1) ** 2
            + (pred_std / target_std - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2
        )
        return kge

    @staticmethod
    def _kge_12(
        pred_mean: np.float64,
        target_mean: np.float64,
        pred_std: np.float64,
        target_std: np.float64,
        corr: np.float64,
        ) -> npt.NDArray[np.float64]:
        """Calculate Kling-Gupta Efficiency (KGE) 1-2."""
        kge_12 = 1 - np.sqrt(
            (corr - 1) ** 2
            + ((pred_std * target_mean) / (target_std * pred_mean) - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2
        )
        return kge_12

    @staticmethod
    def _nse_r2(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        target_mean: np.float64,
        ) -> Tuple[np.float64, np.float64]:
        """Calculate Nash-Sutcliffe Efficiency (NSE) == R^2."""
        sst = np.sum((target - target_mean) ** 2)
        ssres = np.sum((target - pred) ** 2)
        return 1 - ssres / sst



    # outDict = {
    #     'Bias': Bias,
    #     'Bias_ab': Bias_ab,
    #     'RMSE': RMSE,
    #     'ubRMSE': ubRMSE,
    #     'Corr': Corr,
    #     'CorrSp': CorrSp,
    #     'R2': R2,
    #     'NSE': NSE,
    #     'FLV': PBiaslow, # FLV the low flows bias bottom 30%, log space
    #     'FHV': PBiashigh, # FHV the peak flows bias 2%
    #     'PBias': PBias,
    #     'PBiasother': PBiasother,
    #     'absFLV': absPBiaslow,
    #     'absFHV': absPBiashigh,
    #     'absPBias': absPBias,
    #     'absPBiasother': absPBiasother,
    #     'KGE': KGE,
    #     'KGE12': KGE12,
    #     'fdcRMSE': FDCRMSE,
    #     'lowRMSE': RMSElow,
    #     'highRMSE': RMSEhigh,
    #     'midRMSE': RMSEother,
    #     'rdMax': dMax_rel,
    #     'dMax': dMax
    # }

    # return outDict







































# def metrics(pred, target):
#     ngrid, nt = pred.shape
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         # Bias
#         Bias = np.nanmean(abs(pred - target)/(target+0.00001), axis=1)
#         Bias_ab = (np.nansum(pred, axis=1) - np.nansum(target, axis=1))/np.nansum(target, axis=1)
#         # RMSE
#         RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
#         # ubRMSE
#         #dMax_rel = (np.nanmax(pred,axis=1)-np.nanmax(target,axis=1))/np.nanmax(target,axis=1)
#         predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
#         targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
#         predAnom = pred - predMean
#         targetAnom = target - targetMean
#         ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
#         # FDC metric
#         predFDC = calculate_fdc(pred)
#         targetFDC = calculate_fdc(target)
#         FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))
#         # rho R2 NSE
#         dMax_rel = np.full(ngrid, np.nan)
#         dMax = np.full(ngrid, np.nan)
#         Corr = np.full(ngrid, np.nan)
#         CorrSp = np.full(ngrid, np.nan)
#         R2 = np.full(ngrid, np.nan)
#         NSE = np.full(ngrid, np.nan)
#         PBiaslow = np.full(ngrid, np.nan)
#         PBiashigh = np.full(ngrid, np.nan)
#         PBias = np.full(ngrid, np.nan)
#         PBiasother = np.full(ngrid, np.nan)
#         absPBiaslow = np.full(ngrid, np.nan)
#         absPBiashigh = np.full(ngrid, np.nan)
#         absPBias = np.full(ngrid, np.nan)
#         absPBiasother = np.full(ngrid, np.nan)
#         KGE = np.full(ngrid, np.nan)
#         KGE12 = np.full(ngrid, np.nan)
#         RMSElow = np.full(ngrid, np.nan)
#         RMSEhigh = np.full(ngrid, np.nan)
#         RMSEother = np.full(ngrid, np.nan)
#         for k in range(0, ngrid):
#             x = pred[k, :]
#             y = target[k, :]

#             ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]

#             if ind.shape[0] > 0:
#                 xx = x[ind]
#                 yy = y[ind]
#                 maxobs = np.nanmax(yy)
#                 maxIdx = np.nanargmax(yy)
#                 window_lower = 10
#                 window_upper = 11
#                 if (maxIdx < window_lower):
#                     window_lower = maxIdx
#                 elif (window_upper > len(xx) - maxIdx):
#                     window_upper = len(xx) - maxIdx

#                 maxpred = np.nanmax(xx[maxIdx - window_lower:maxIdx + window_upper])
#                 #  maxpred = np.nanmax(x)
#                 dMax[k] = maxpred - maxobs
#                 dMax_rel[k] = (maxpred - maxobs) / maxobs * 100


#                 # percent bias
#                 PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

#                 # FHV the peak flows bias 2%
#                 # FLV the low flows bias bottom 30%, log space
#                 pred_sort = np.sort(xx)
#                 target_sort = np.sort(yy)
#                 indexlow = round(0.3 * len(pred_sort))
#                 indexhigh = round(0.98 * len(pred_sort))
#                 lowpred = pred_sort[:indexlow]
#                 highpred = pred_sort[indexhigh:]
#                 otherpred = pred_sort[indexlow:indexhigh]
#                 lowtarget = target_sort[:indexlow]
#                 hightarget = target_sort[indexhigh:]
#                 othertarget = target_sort[indexlow:indexhigh]
#                 PBiaslow[k] = np.sum((lowpred - lowtarget)) / (np.sum(lowtarget) +0.0001)* 100
#                 PBiashigh[k] = np.sum((highpred - hightarget) )/ np.sum(hightarget) * 100
#                 PBiasother[k] = np.sum((otherpred - othertarget)) / np.sum(othertarget) * 100
#                 absPBiaslow[k] = np.sum(abs(lowpred - lowtarget)) / ((np.sum(lowtarget) +0.0001))* 100
#                 absPBiashigh[k] = np.sum(abs(highpred - hightarget) )/ np.sum(hightarget) * 100
#                 absPBiasother[k] = np.sum(abs(otherpred - othertarget)) / np.sum(othertarget) * 100

#                 RMSElow[k] = np.sqrt(np.nanmean((lowpred - lowtarget)**2))
#                 RMSEhigh[k] = np.sqrt(np.nanmean((highpred - hightarget)**2))
#                 RMSEother[k] = np.sqrt(np.nanmean((otherpred - othertarget)**2))

#                 if ind.shape[0] > 1:
#                     # Theoretically at least two points for correlation
#                     Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
#                     CorrSp[k] = scipy.stats.spearmanr(xx, yy)[0]
#                     yymean = yy.mean()
#                     yystd = np.std(yy)
#                     xxmean = xx.mean()
#                     xxstd = np.std(xx)
#                     KGE[k] = 1 - np.sqrt((Corr[k]-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)
#                     KGE12[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + ((xxstd*yymean)/ (yystd*xxmean) - 1) ** 2 + (xxmean / yymean - 1) ** 2)
#                     SST = np.sum((yy-yymean)**2)
#                     SSReg = np.sum((xx-yymean)**2)
#                     SSRes = np.sum((yy-xx)**2)
#                     R2[k] = 1-SSRes/SST
#                     NSE[k] = 1-SSRes/SST

#     outDict = {
#         'Bias': Bias,
#         'Bias_ab': Bias_ab,
#         'RMSE': RMSE,
#         'ubRMSE': ubRMSE,
#         'Corr': Corr,
#         'CorrSp': CorrSp,
#         'R2': R2,
#         'NSE': NSE,
#         'FLV': PBiaslow, # FLV the low flows bias bottom 30%, log space
#         'FHV': PBiashigh, # FHV the peak flows bias 2%
#         'PBias': PBias,
#         'PBiasother': PBiasother,
#         'absFLV': absPBiaslow,
#         'absFHV': absPBiashigh,
#         'absPBias': absPBias,
#         'absPBiasother': absPBiasother,
#         'KGE': KGE,
#         'KGE12': KGE12,
#         'fdcRMSE': FDCRMSE,
#         'lowRMSE': RMSElow,
#         'highRMSE': RMSEhigh,
#         'midRMSE': RMSEother,
#         'rdMax': dMax_rel,
#         'dMax': dMax
#     }

#     return outDict


# def calculate_fdc(data):
#     """Calculate flow duration curve (FDC) for each gage."""
#     # data = Ngrid * Nday
#     Ngrid, Nday = data.shape
#     FDC100 = np.full([Ngrid, 100], np.nan)
#     for ii in range(Ngrid):
#         tempdata0 = data[ii, :]
#         tempdata = tempdata0[~np.isnan(tempdata0)]
#         # deal with no data case for some gages
#         if len(tempdata)==0:
#             tempdata = np.full(Nday, 0)
#         # sort from large to small
#         temp_sort = np.sort(tempdata)[::-1]
#         # select 100 quantile points
#         Nlen = len(tempdata)
#         ind = (np.arange(100)/100*Nlen).astype(int)
#         FDCflow = temp_sort[ind]
#         if len(FDCflow) != 100:
#             raise Exception('unknown assimilation variable')
#         else:
#             FDC100[ii, :] = FDCflow

#     return FDC100
