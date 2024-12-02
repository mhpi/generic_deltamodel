import logging
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger()


class Metrics(BaseModel):
    """Metrics for model evaluation.

    Using pydantic BaseModel for validation.
    Metrics are calculated at each grid point and are listed below.
    
    Adapted from Tadd Bindas, Yalan Song, Farshid Rahmani.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: npt.NDArray[np.float32]
    target: npt.NDArray[np.float32]
    bias: npt.NDArray[np.float32] = np.ndarray([])
    bias_rel: npt.NDArray[np.float32] = np.ndarray([])

    rmse: npt.NDArray[np.float32] = np.ndarray([])
    rmse_ub: npt.NDArray[np.float32] = np.ndarray([])
    rmse_fdc: npt.NDArray[np.float32] = np.ndarray([])

    mae: npt.NDArray[np.float32] = np.ndarray([])

    corr: npt.NDArray[np.float32] = np.ndarray([])
    corr_spearman: npt.NDArray[np.float32] = np.ndarray([])
    r2: npt.NDArray[np.float32] = np.ndarray([])
    nse: npt.NDArray[np.float32] = np.ndarray([])

    flv: npt.NDArray[np.float32] = np.ndarray([])
    fhv: npt.NDArray[np.float32] = np.ndarray([])
    pbias: npt.NDArray[np.float32] = np.ndarray([])
    pbias_mid: npt.NDArray[np.float32] = np.ndarray([])
    flv_abs: npt.NDArray[np.float32] = np.ndarray([])
    fhv_abs: npt.NDArray[np.float32] = np.ndarray([])
    pbias_abs: npt.NDArray[np.float32] = np.ndarray([])
    pbias_abs_mid: npt.NDArray[np.float32] = np.ndarray([])

    kge: npt.NDArray[np.float32] = np.ndarray([])
    kge_12: npt.NDArray[np.float32] = np.ndarray([])
    
    rmse_low: npt.NDArray[np.float32] = np.ndarray([])
    rmse_mid: npt.NDArray[np.float32] = np.ndarray([])
    rmse_high: npt.NDArray[np.float32] = np.ndarray([])

    d_max: npt.NDArray[np.float32] = np.ndarray([])
    d_max_rel: npt.NDArray[np.float32] = np.ndarray([])

    def __init__(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> None:
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)

        super().__init__(pred=pred, target=target)

    def model_post_init(self, __context: Any):
        """Calculate metrics.
        
        Parameters
        ----------
        __context : Any
            Context object.

        Returns
        -------
        Any
            Context object.
        """
        self.bias = self._bias(self.pred, self.target)
        self.bias_rel = self._bias_rel(self.pred, self.target)

        self.rmse = self._rmse(self.pred, self.target)
        self.rmse_ub = self._rmse_ub(self.pred, self.target)
        self.rmse_fdc = self._rmse_fdc(self.pred, self.target)

        self.mae = self._mae(self.pred, self.target)

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


    @model_validator(mode="after")
    @classmethod
    def validate_pred(cls, metrics: Any) -> Any:
        """Checks that there are no NaN predictions.
        
        Parameters
        ----------
        metrics : Any
            Metrics object.

        Raises
        ------
        ValueError
            If there are NaN predictions.
        
        Returns
        -------
        Any
            Metrics object.
        """
        pred = metrics.pred
        if np.isnan(pred).sum() > 0:
            msg = "Pred contains NaN, check your gradient chain"
            log.exception(msg)
            raise ValueError(msg)
        return metrics
    
    def model_dump_json(self, *args, **kwargs) -> str:
        """Dump metrics to json."""
        model_dict = self.model_dump()
        for key, value in model_dict.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value.tolist())

        if hasattr(self, 'pred'):
            del self.pred
        if hasattr(self, 'target'):
            del self.target

        return super().model_dump_json(*args, **kwargs)
    
    @property
    def ngrid(self) -> int:
        """Calculate number of items in grid."""
        return self.pred.shape[0]
    
    @property
    def nt(self) -> int:
        """Calculate number of time steps."""
        return self.pred.shape[1]

    def tile_mean(self, data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate mean of target.

        Parameters
        ----------
        data : npt.NDArray[np.float32]
            Data to calculate mean.
        
        Returns
        -------
        npt.NDArray[np.float32]
            Mean of data.
        """
        return np.tile(np.nanmean(data, axis=1), (self.nt, 1)).transpose()
    
    @staticmethod
    def _bias(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate bias."""
        return np.nanmean(abs(pred - target)/(target + 0.0001), axis=1)

    @staticmethod
    def _bias_rel(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate relative bias.
        
        Don't sum together because if NaNs are present at different idx,
        the sum will be off.
        """
        pred_sum = np.nansum(pred, axis=1)
        target_sum = np.nansum(target, axis=1)
        return (pred_sum - target_sum) / target_sum

    @staticmethod
    def _bias_percent(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        offset: float = 0.0,
    ) -> np.float32:
        """Calculate percent bias."""
        p_bias = np.sum(pred - target) / (np.sum(target) + offset) * 100
        return p_bias
    
    @staticmethod
    def _rmse(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        axis: Optional[int] = 1,
    ) -> npt.NDArray[np.float32]:
        """Calculate root mean square error."""
        return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))
    
    def _rmse_ub(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate unbiased root mean square error."""
        pred_mean = self.tile_mean(self.pred)
        target_mean = self.tile_mean(self.target)
        pred_anom = self.pred - pred_mean
        target_anom = self.target - target_mean
        return self._rmse(pred_anom, target_anom)
    
    def _rmse_fdc(
        self,
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Calculate flow duration curve root mean square error."""
        pred_fdc = self._calc_fdc(pred)
        target_fdc = self._calc_fdc(target)
        return self._rmse(pred_fdc, target_fdc)

    @staticmethod
    def _mae(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        axis: Optional[int] = 1,
    ) -> npt.NDArray[np.float32]:
        """Calculate mean absolute error."""
        return np.nanmean(np.abs(pred - target), axis=axis)
    
    def _calc_fdc(
        self,
        data: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
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
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        lb: int = 0,
        ub: int = 10,
    ) -> np.float32:
        """Calculate maximum value of predictions.
        
        Parameters
        ----------
        pred : npt.NDArray[np.float32]
            Predictions.
        target : npt.NDArray[np.float32]
            Target values.
        lb : int, optional
            Lower bound. Default is 0.
        ub : int, optional
            Upper bound. Default is 10.
        """
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
        pred: npt.NDArray[np.float32], target: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Calculate correlation."""
        corr = stats.pearsonr(pred, target)[0]
        return corr

    @staticmethod
    def _corr_spearman(
        pred: npt.NDArray[np.float32], target: npt.NDArray[np.float32]
    ) -> np.float32:
        """Calculate Spearman correlation."""
        corr_spearman = stats.spearmanr(pred, target)[0]
        return corr_spearman
    
    @staticmethod
    def _kge(
        pred_mean: np.float32,
        target_mean: np.float32,
        pred_std: np.float32,
        target_std: np.float32,
        corr: np.float32,
    ) -> npt.NDArray[np.float32]:
        """Calculate Kling-Gupta Efficiency (KGE)."""
        kge = 1 - np.sqrt(
            (corr - 1) ** 2
            + (pred_std / target_std - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2
        )
        return kge

    @staticmethod
    def _kge_12(
        pred_mean: np.float32,
        target_mean: np.float32,
        pred_std: np.float32,
        target_std: np.float32,
        corr: np.float32,
    ) -> npt.NDArray[np.float32]:
        """Calculate Kling-Gupta Efficiency (KGE) 1-2."""
        kge_12 = 1 - np.sqrt(
            (corr - 1) ** 2
            + ((pred_std * target_mean) / (target_std * pred_mean) - 1) ** 2
            + (pred_mean / target_mean - 1) ** 2
        )
        return kge_12

    @staticmethod
    def _nse_r2(
        pred: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32],
        target_mean: np.float32,
    ) -> Tuple[np.float32, np.float32]:
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