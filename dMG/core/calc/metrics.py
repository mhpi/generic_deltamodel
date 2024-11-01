# TODO: Replace stat.py with this updated and more maintanable version from dMCdev @Tadd Bindas.
import logging
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger()



class Metrics(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: npt.NDArray[np.float64]
    target: npt.NDArray[np.float64]
    bias: npt.NDArray[np.float64] = np.ndarray([])
    rmse: npt.NDArray[np.float64] = np.ndarray([])
    ub_rmse: npt.NDArray[np.float64] = np.ndarray([])
    fdc_rmse: npt.NDArray[np.float64] = np.ndarray([])
    corr: npt.NDArray[np.float64] = np.ndarray([])
    corr_spearman: npt.NDArray[np.float64] = np.ndarray([])
    r2: npt.NDArray[np.float64] = np.ndarray([])
    nse: npt.NDArray[np.float64] = np.ndarray([])
    flv: npt.NDArray[np.float64] = np.ndarray([])
    fhv: npt.NDArray[np.float64] = np.ndarray([])
    pbias: npt.NDArray[np.float64] = np.ndarray([])
    pbias_mid: npt.NDArray[np.float64] = np.ndarray([])
    kge: npt.NDArray[np.float64] = np.ndarray([])
    kge_12: npt.NDArray[np.float64] = np.ndarray([])
    rmse_low: npt.NDArray[np.float64] = np.ndarray([])
    rmse_high: npt.NDArray[np.float64] = np.ndarray([])
    rmse_mid: npt.NDArray[np.float64] = np.ndarray([])

    def __init__(self, pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]):
        super(Metrics, self).__init__(pred=pred, target=target)

    def model_post_init(self, __context: Any):
        self.bias = self._bias(self.pred, self.target)
        self.rmse = self._rmse(self.pred, self.target)

        pred_mean = self.tile_mean(self.pred)
        target_mean = self.tile_mean(self.target)
        pred_anom = self.pred - pred_mean
        target_anom = self.target - target_mean
        self.ub_rmse = self._rmse(pred_anom, target_anom)

        pred_fdc = self._calc_fdc(self.pred)
        target_fdc = self._calc_fdc(self.target)
        self.fdc_rmse = self._rmse(pred_fdc, target_fdc)

        self.corr = np.full(self.ngrid, np.nan)
        self.corr_spearman = np.full(self.ngrid, np.nan)
        self.r2 = np.full(self.ngrid, np.nan)
        self.nse = np.full(self.ngrid, np.nan)
        self.flv = np.full(self.ngrid, np.nan)
        self.fhv = np.full(self.ngrid, np.nan)
        self.pbias = np.full(self.ngrid, np.nan)
        self.pbias_mid = np.full(self.ngrid, np.nan)
        self.kge = np.full(self.ngrid, np.nan)
        self.kge_12 = np.full(self.ngrid, np.nan)
        self.rmse_low = np.full(self.ngrid, np.nan)
        self.rmse_high = np.full(self.ngrid, np.nan)
        self.rmse_mid = np.full(self.ngrid, np.nan)
        for i in range(0, self.ngrid):
            _pred = self.pred[i]
            _target = self.target[i]
            non_nan_idx = np.where(
                np.logical_and(~np.isnan(_pred), ~np.isnan(_target))
            )[0]
            if non_nan_idx.shape[0] > 0:
                pred = _pred[non_nan_idx]
                target = _target[non_nan_idx]

                pred_sort = np.sort(pred)
                target_sort = np.sort(target)
                index_low = round(0.3 * pred_sort.shape[0])
                index_high = round(0.98 * pred_sort.shape[0])
                low_pred = pred_sort[:index_low]
                high_pred = pred_sort[index_high:]
                mid_pred = pred_sort[index_low:index_high]
                low_target = target_sort[:index_low]
                high_target = target_sort[index_high:]
                mid_target = target_sort[index_low:index_high]

                self.pbias[i] = self._p_bias(pred, target)
                self.flv[i] = self._p_bias(low_pred, low_target)
                self.fhv[i] = self._p_bias(high_pred, high_target)
                self.pbias_mid[i] = self._p_bias(mid_pred, mid_target)
                self.rmse_low[i] = self._rmse(low_pred, low_target, axis=0)
                self.rmse_high[i] = self._rmse(high_pred, high_target, axis=0)
                self.rmse_mid[i] = self._rmse(mid_pred, mid_target, axis=0)

                if non_nan_idx.shape[0] > 1:
                    self.corr[i] = self._corr(pred, target)
                    self.corr_spearman[i] = self._corr_spearman(pred, target)
                    _pred_mean = pred.mean()
                    _target_mean = target.mean()
                    _pred_std = np.std(pred)
                    _target_std = np.std(target)
                    self.kge[i] = self._kge(
                        _pred_mean, _target_mean, _pred_std, _target_std, self.corr[i]
                    )
                    self.kge_12[i] = self._kge_12(
                        _pred_mean, _target_mean, _pred_std, _target_std, self.corr[i]
                    )
                    self.nse[i], self.r2[i] = self._nse_r2(pred, target, _target_mean)
        return super().model_post_init(__context)

    @model_validator(mode="after")
    @classmethod
    def validate_pred(cls, metrics: Any) -> Any:
        pred = metrics.pred
        if np.isnan(pred).sum() > 0:
            msg = "pred contains NaN, check your gradient chain"
            log.exception(msg)
            raise ValueError(msg)
        return metrics

    def _calc_fdc(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate flow duration curve for each grid
        """
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
                raise Exception("unknown assimilation variable")
            else:
                fdc_100[i] = fdc_flow
        return fdc_100

    def model_dump_json(self, *args, **kwargs) -> str:
        model_dict = self.model_dump()
        for key, value in model_dict.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value.tolist())

        # I don't want these saved to disk as that's going to waste space
        if hasattr(self, "pred"):
            del self.pred
        if hasattr(self, "target"):
            del self.target

        return super().model_dump_json(*args, **kwargs)

    @property
    def ngrid(self) -> int:
        """
        Calculate number of grids
        """
        return self.pred.shape[0]

    @property
    def nt(self) -> int:
        """
        Calculate number of time steps
        """
        return self.pred.shape[1]

    def tile_mean(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Calculate mean of target
        """
        return np.tile(np.nanmean(data, axis=1), (self.nt, 1)).transpose()

    @staticmethod
    def _rmse(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        axis: Optional[int] = 1,
    ) -> npt.NDArray[np.float64]:
        """
        Calculate root mean square error
        """
        return np.sqrt(np.nanmean((pred - target) ** 2, axis=axis))

    @staticmethod
    def _bias(
        pred: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate bias
        """
        return np.nanmean((pred - target), axis=1)

    @staticmethod
    def _p_bias(
        pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> np.float64:
        """
        Calculate p bias
        """
        p_bias = np.sum(pred - target) / np.sum(target) * 100
        return p_bias

    @staticmethod
    def _corr(
        pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Calculate correlation
        """
        corr = stats.pearsonr(pred, target)[0]
        return corr

    @staticmethod
    def _corr_spearman(
        pred: npt.NDArray[np.float64], target: npt.NDArray[np.float64]
    ) -> np.float64:
        """
        Calculate spearman r
        """
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
        """
        Calculate KGE
        """
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
        """
        Calculate KGE 1-2
        """
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
        """
        Calculate NSE/R2
        """
        sst = np.sum((target - target_mean) ** 2)
        ssres = np.sum((target - pred) ** 2)
        r2 = 1 - ssres / sst
        nse = 1 - ssres / sst
        return nse, r2
