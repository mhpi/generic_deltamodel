import csv
import json
import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger()


class FireMetrics(BaseModel):
    """
    Fire occurrence prediction metrics following the paper:
    "Part I: Improving Wildfire Occurrence Prediction for CONUS Using Deep Learning and Fire Weather Variables"
    
    Focuses on binary classification metrics appropriate for fire occurrence prediction.
    Adapted from the original Metrics class to handle fire-specific evaluation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: NDArray[np.float32]  # Predicted fire probabilities [0-1]
    target: NDArray[np.float32]  # Binary fire occurrence [0,1]
    
    # Threshold-dependent metrics (will be computed for multiple thresholds)
    csi: NDArray[np.float32] = np.ndarray([])  # Critical Success Index
    pod: NDArray[np.float32] = np.ndarray([])  # Probability of Detection
    far: NDArray[np.float32] = np.ndarray([])  # False Alarm Rate
    pofd: NDArray[np.float32] = np.ndarray([])  # Probability of False Detection
    
    # Threshold-independent metrics
    auc: NDArray[np.float32] = np.ndarray([])  # Area Under ROC Curve
    brier_score: NDArray[np.float32] = np.ndarray([])  # Brier Score
    brier_skill_score: NDArray[np.float32] = np.ndarray([])  # Brier Skill Score
    
    # Fire-specific metrics
    hit_rate: NDArray[np.float32] = np.ndarray([])  # True Positive Rate
    miss_rate: NDArray[np.float32] = np.ndarray([])  # False Negative Rate
    false_alarm_ratio: NDArray[np.float32] = np.ndarray([])  # False Alarm Ratio
    
    # Spatial metrics (for different neighborhood sizes like in the paper)
    csi_40km: NDArray[np.float32] = np.ndarray([])  # CSI for 40km neighborhood
    csi_80km: NDArray[np.float32] = np.ndarray([])  # CSI for 80km neighborhood  
    csi_120km: NDArray[np.float32] = np.ndarray([])  # CSI for 120km neighborhood
    
    # Probability thresholds to evaluate
    thresholds: NDArray[np.float32] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    def __init__(
        self,
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        thresholds: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """
        Initialize fire metrics.
        
        Parameters
        ----------
        pred : NDArray[np.float32]
            Predicted fire probabilities, shape [grid_cells, time] or [time, grid_cells]
        target : NDArray[np.float32] 
            Binary fire occurrence (0/1), same shape as pred
        thresholds : Optional[NDArray[np.float32]]
            Probability thresholds to evaluate, default is [0.1, 0.2, ..., 0.9]
        """
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)
            
        # Ensure shapes match
        if pred.shape != target.shape:
            log.warning(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
            # Try to align shapes
            if pred.shape[0] == target.shape[1] and pred.shape[1] == target.shape[0]:
                pred = pred.T
                log.info("Transposed predictions to match target shape")
        
        if thresholds is not None:
            thresholds = thresholds
        else:
            thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        super().__init__(pred=pred, target=target, thresholds=thresholds)

    def model_post_init(self, __context: Any) -> Any:
        """Calculate all fire metrics after initialization."""
        
        # Ensure predictions are probabilities [0,1]
        self.pred = np.clip(self.pred, 0.0, 1.0)
        
        # Ensure targets are binary [0,1]
        self.target = np.clip(self.target, 0.0, 1.0)
        
        log.info(f"Computing fire metrics for {self.ngrid} grid cells, {self.nt} time steps")
        log.info(f"Fire occurrence rate: {np.mean(self.target):.4f}")
        
        # Initialize metric arrays
        n_thresholds = len(self.thresholds)
        self.csi = np.full((self.ngrid, n_thresholds), np.nan)
        self.pod = np.full((self.ngrid, n_thresholds), np.nan)
        self.far = np.full((self.ngrid, n_thresholds), np.nan)
        self.pofd = np.full((self.ngrid, n_thresholds), np.nan)
        
        # Threshold-independent metrics
        self.auc = np.full(self.ngrid, np.nan)
        self.brier_score = np.full(self.ngrid, np.nan)
        self.brier_skill_score = np.full(self.ngrid, np.nan)
        
        # Compute metrics for each grid cell
        for i in range(self.ngrid):
            pred_i = self.pred[i]
            target_i = self.target[i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
            if not np.any(valid_mask):
                continue
                
            pred_valid = pred_i[valid_mask]
            target_valid = target_i[valid_mask]
            
            # Skip if no variation in target (all 0s or all 1s)
            if len(np.unique(target_valid)) < 2:
                continue
            
            # Compute threshold-dependent metrics
            for j, threshold in enumerate(self.thresholds):
                pred_binary = (pred_valid >= threshold).astype(int)
                
                # Compute confusion matrix elements
                tp, fp, tn, fn = self._confusion_matrix_elements(pred_binary, target_valid)
                
                # Compute metrics
                self.csi[i, j] = self._critical_success_index(tp, fp, fn)
                self.pod[i, j] = self._probability_of_detection(tp, fn)
                self.far[i, j] = self._false_alarm_rate(fp, tn)
                self.pofd[i, j] = self._probability_of_false_detection(fp, tn)
            
            # Compute threshold-independent metrics
            self.auc[i] = self._area_under_curve(pred_valid, target_valid)
            self.brier_score[i] = self._brier_score(pred_valid, target_valid)
            self.brier_skill_score[i] = self._brier_skill_score(pred_valid, target_valid)
        
        # Compute best-threshold metrics (maximum CSI for each grid cell)
        self.hit_rate = np.full(self.ngrid, np.nan)
        self.miss_rate = np.full(self.ngrid, np.nan)
        self.false_alarm_ratio = np.full(self.ngrid, np.nan)
        
        for i in range(self.ngrid):
            if not np.isnan(self.csi[i]).all():
                # Find threshold with maximum CSI
                best_threshold_idx = np.nanargmax(self.csi[i])
                
                # Compute metrics at best threshold
                pred_i = self.pred[i]
                target_i = self.target[i]
                valid_mask = ~(np.isnan(pred_i) | np.isnan(target_i))
                
                if np.any(valid_mask):
                    pred_valid = pred_i[valid_mask]
                    target_valid = target_i[valid_mask]
                    
                    best_threshold = self.thresholds[best_threshold_idx]
                    pred_binary = (pred_valid >= best_threshold).astype(int)
                    
                    tp, fp, tn, fn = self._confusion_matrix_elements(pred_binary, target_valid)
                    
                    self.hit_rate[i] = self._probability_of_detection(tp, fn)
                    self.miss_rate[i] = fn / (tp + fn) if (tp + fn) > 0 else np.nan
                    self.false_alarm_ratio[i] = fp / (tp + fp) if (tp + fp) > 0 else np.nan
        
        return super().model_post_init(__context)

    def _confusion_matrix_elements(
        self, pred_binary: NDArray[np.int32], target: NDArray[np.float32]
    ) -> Tuple[int, int, int, int]:
        """Compute confusion matrix elements."""
        tp = np.sum((pred_binary == 1) & (target == 1))
        fp = np.sum((pred_binary == 1) & (target == 0))
        tn = np.sum((pred_binary == 0) & (target == 0))
        fn = np.sum((pred_binary == 0) & (target == 1))
        return tp, fp, tn, fn

    def _critical_success_index(self, tp: int, fp: int, fn: int) -> float:
        """Compute Critical Success Index (CSI) = TP / (TP + FP + FN)."""
        denominator = tp + fp + fn
        return tp / denominator if denominator > 0 else np.nan

    def _probability_of_detection(self, tp: int, fn: int) -> float:
        """Compute Probability of Detection (POD) = TP / (TP + FN)."""
        denominator = tp + fn
        return tp / denominator if denominator > 0 else np.nan

    def _false_alarm_rate(self, fp: int, tn: int) -> float:
        """Compute False Alarm Rate (FAR) = FP / (FP + TN)."""
        denominator = fp + tn
        return fp / denominator if denominator > 0 else np.nan

    def _probability_of_false_detection(self, fp: int, tn: int) -> float:
        """Compute Probability of False Detection = FP / (FP + TN)."""
        return self._false_alarm_rate(fp, tn)

    def _area_under_curve(self, pred: NDArray[np.float32], target: NDArray[np.float32]) -> float:
        """Compute Area Under ROC Curve using trapezoidal rule."""
        # Sort by prediction values
        sorted_indices = np.argsort(pred)
        sorted_pred = pred[sorted_indices]
        sorted_target = target[sorted_indices]
        
        # Compute ROC curve points
        thresholds = np.unique(sorted_pred)
        if len(thresholds) == 1:
            return np.nan
            
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            pred_binary = (sorted_pred >= threshold).astype(int)
            tp, fp, tn, fn = self._confusion_matrix_elements(pred_binary, sorted_target)
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Add endpoints
        tpr_values = [0] + tpr_values + [1]
        fpr_values = [0] + fpr_values + [1]
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr_values, fpr_values)
        return auc

    def _brier_score(self, pred: NDArray[np.float32], target: NDArray[np.float32]) -> float:
        """Compute Brier Score = mean((pred - target)^2)."""
        return np.mean((pred - target) ** 2)

    def _brier_skill_score(self, pred: NDArray[np.float32], target: NDArray[np.float32]) -> float:
        """Compute Brier Skill Score relative to climatology."""
        brier_score = self._brier_score(pred, target)
        climatology = np.mean(target)
        brier_score_ref = np.mean((climatology - target) ** 2)
        
        if brier_score_ref == 0:
            return np.nan
        
        return 1 - (brier_score / brier_score_ref)

    def get_best_threshold_metrics(self) -> dict:
        """Get metrics at the threshold that maximizes CSI for each grid cell."""
        best_metrics = {}
        
        for i in range(self.ngrid):
            if not np.isnan(self.csi[i]).all():
                best_threshold_idx = np.nanargmax(self.csi[i])
                best_threshold = self.thresholds[best_threshold_idx]
                
                best_metrics[f'grid_{i}'] = {
                    'best_threshold': float(best_threshold),
                    'best_csi': float(self.csi[i, best_threshold_idx]),
                    'best_pod': float(self.pod[i, best_threshold_idx]),
                    'best_far': float(self.far[i, best_threshold_idx]),
                    'auc': float(self.auc[i]) if not np.isnan(self.auc[i]) else None,
                    'brier_score': float(self.brier_score[i]) if not np.isnan(self.brier_score[i]) else None,
                    'brier_skill_score': float(self.brier_skill_score[i]) if not np.isnan(self.brier_skill_score[i]) else None,
                }
        
        return best_metrics

    def calc_stats(self) -> dict[str, dict[str, float]]:
        """Calculate aggregate statistics of fire metrics."""
        stats = {}
        
        # For threshold-dependent metrics, use the best threshold for each grid cell
        max_csi = np.nanmax(self.csi, axis=1)
        max_pod = np.array([self.pod[i, np.nanargmax(self.csi[i])] if not np.isnan(self.csi[i]).all() else np.nan 
                           for i in range(self.ngrid)])
        min_far = np.array([self.far[i, np.nanargmax(self.csi[i])] if not np.isnan(self.csi[i]).all() else np.nan 
                           for i in range(self.ngrid)])
        
        metrics_dict = {
            'csi_max': max_csi,
            'pod_at_best_csi': max_pod,
            'far_at_best_csi': min_far,
            'auc': self.auc,
            'brier_score': self.brier_score,
            'brier_skill_score': self.brier_skill_score,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'false_alarm_ratio': self.false_alarm_ratio,
        }
        
        # Calculate statistics for each metric
        for key, values in metrics_dict.items():
            if len(values) > 0 and not np.isnan(values).all():
                stats[key] = {
                    'median': float(np.nanmedian(values)),
                    'mean': float(np.nanmean(values)),
                    'std': float(np.nanstd(values)),
                    'min': float(np.nanmin(values)),
                    'max': float(np.nanmax(values)),
                    'count_valid': int(np.sum(~np.isnan(values))),
                }
            else:
                stats[key] = {
                    'median': None, 'mean': None, 'std': None,
                    'min': None, 'max': None, 'count_valid': 0
                }
        
        return stats

    def dump_metrics(self, path: str) -> None:
        """Dump all fire metrics and aggregate statistics to files."""
        os.makedirs(path, exist_ok=True)
        
        # Save aggregate statistics
        stats = self.calc_stats()
        stats_path = os.path.join(path, 'fire_metrics_agg.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Save best threshold metrics for each grid cell
        best_metrics = self.get_best_threshold_metrics()
        best_path = os.path.join(path, 'fire_metrics_best_thresholds.json')
        with open(best_path, 'w') as f:
            json.dump(best_metrics, f, indent=4)
        
        # Save detailed threshold analysis
        threshold_analysis = self._threshold_analysis()
        threshold_path = os.path.join(path, 'fire_metrics_by_threshold.json')
        with open(threshold_path, 'w') as f:
            json.dump(threshold_analysis, f, indent=4)
        
        log.info(f"Fire metrics saved to {path}")

    def _threshold_analysis(self) -> dict:
        """Analyze metrics across all thresholds."""
        analysis = {}
        
        for j, threshold in enumerate(self.thresholds):
            csi_values = self.csi[:, j]
            pod_values = self.pod[:, j]
            far_values = self.far[:, j]
            
            analysis[f'threshold_{threshold:.1f}'] = {
                'csi': {
                    'mean': float(np.nanmean(csi_values)),
                    'median': float(np.nanmedian(csi_values)),
                    'std': float(np.nanstd(csi_values)),
                },
                'pod': {
                    'mean': float(np.nanmean(pod_values)),
                    'median': float(np.nanmedian(pod_values)),
                    'std': float(np.nanstd(pod_values)),
                },
                'far': {
                    'mean': float(np.nanmean(far_values)),
                    'median': float(np.nanmedian(far_values)),
                    'std': float(np.nanstd(far_values)),
                },
            }
        
        return analysis

    @property
    def ngrid(self) -> int:
        """Number of grid cells."""
        return self.pred.shape[0]

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return self.pred.shape[1]

    @model_validator(mode='after')
    @classmethod
    def validate_fire_data(cls, metrics: Any) -> Any:
        """Validate fire prediction data."""
        pred = metrics.pred
        target = metrics.target
        
        # Check for NaN predictions
        if np.isnan(pred).sum() > 0:
            log.warning(f"Predictions contain {np.isnan(pred).sum()} NaN values")
        
        # Check prediction range
        if np.any((pred < 0) | (pred > 1)):
            log.warning("Predictions outside [0,1] range, clipping values")
        
        # Check target values
        unique_targets = np.unique(target[~np.isnan(target)])
        if len(unique_targets) > 2 or not np.all(np.isin(unique_targets, [0, 1])):
            log.warning(f"Target values are not binary: {unique_targets}")
        
        # Check for class imbalance
        fire_rate = np.nanmean(target)
        if fire_rate < 0.01:
            log.warning(f"Very low fire occurrence rate: {fire_rate:.4f}")
        elif fire_rate > 0.5:
            log.warning(f"High fire occurrence rate: {fire_rate:.4f}")
        
        return metrics