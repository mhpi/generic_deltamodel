import csv
import json
import logging
import os
from typing import Any, Optional, Tuple, Dict, List

import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger()


class FireMetrics(BaseModel):
    """
    Enhanced Fire occurrence prediction metrics with support for:
    1. Multi-day forecasting (up to 10 days)
    2. Spatial neighborhood analysis (40km, 80km, 120km)
    3. Day-by-day performance tracking
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pred: NDArray[np.float32]  # Predicted fire probabilities [0-1]
    target: NDArray[np.float32]  # Binary fire occurrence [0,1]
    
    # Multi-day forecast support
    forecast_days: int = 1  # Number of forecast days
    grid_resolution_km: float = 40.0  # Grid resolution in kilometers
    
    # Spatial neighborhood CSI results
    csi_40km: Dict[int, float] = {}  # CSI for each forecast day at 40km
    csi_80km: Dict[int, float] = {}  # CSI for each forecast day at 80km  
    csi_120km: Dict[int, float] = {}  # CSI for each forecast day at 120km
    
    # Best thresholds for each day and neighborhood
    best_thresholds_40km: Dict[int, float] = {}
    best_thresholds_80km: Dict[int, float] = {}
    best_thresholds_120km: Dict[int, float] = {}
    
    # Standard metrics for compatibility
    csi: NDArray[np.float32] = np.ndarray([])
    pod: NDArray[np.float32] = np.ndarray([])
    far: NDArray[np.float32] = np.ndarray([])
    auc: NDArray[np.float32] = np.ndarray([])
    brier_score: NDArray[np.float32] = np.ndarray([])
    
    def __init__(
        self,
        pred: NDArray[np.float32],
        target: NDArray[np.float32],
        forecast_days: int = 1,
        grid_resolution_km: float = 40.0,
    ) -> None:
        """
        Initialize enhanced fire metrics.
        
        Parameters
        ----------
        pred : NDArray[np.float32]
            Predicted fire probabilities
            For multi-day: [forecast_days, grid_cells, time] or [grid_cells, time, forecast_days]
            For single day: [grid_cells, time]
        target : NDArray[np.float32] 
            Binary fire occurrence (0/1), same structure as pred
        forecast_days : int
            Number of forecast days (1-10)
        grid_resolution_km : float
            Grid resolution in kilometers (default 40km)
        """
        
        # Handle different input shapes for multi-day forecasting
        if forecast_days > 1:
            if pred.ndim == 3:
                # Assume [forecast_days, grid_cells, time] or [grid_cells, time, forecast_days]
                if pred.shape[0] == forecast_days:
                    # [forecast_days, grid_cells, time] - keep as is
                    pass
                elif pred.shape[2] == forecast_days:
                    # [grid_cells, time, forecast_days] - transpose
                    pred = np.transpose(pred, (2, 0, 1))
                    target = np.transpose(target, (2, 0, 1)) if target.ndim == 3 else target
                else:
                    log.warning(f"Unexpected prediction shape for {forecast_days} days: {pred.shape}")
                    # Assume first day only
                    forecast_days = 1
                    if pred.ndim == 3:
                        pred = pred[0] if pred.shape[0] < pred.shape[2] else pred[:, :, 0]
                        target = target[0] if target.ndim == 3 and target.shape[0] < target.shape[2] else target[:, :, 0] if target.ndim == 3 else target
            else:
                # Single day case
                forecast_days = 1
        
        # Ensure shapes are consistent
        if pred.ndim == 1:
            pred = np.expand_dims(pred, axis=0)
        if target.ndim == 1:
            target = np.expand_dims(target, axis=0)
            
        log.info(f"Enhanced FireMetrics initialized: {forecast_days} forecast days")
        log.info(f"Prediction shape: {pred.shape}, Target shape: {target.shape}")
        
            
        super().__init__(
            pred=pred, 
            target=target, 
            forecast_days=forecast_days,
            grid_resolution_km=grid_resolution_km
        )

    def model_post_init(self, __context: Any) -> Any:
        """Calculate all fire metrics after initialization."""
        
        # Ensure predictions are probabilities [0,1]
        self.pred = np.clip(self.pred, 0.0, 1.0)
        
        # Ensure targets are binary [0,1]
        self.target = np.clip(self.target, 0.0, 1.0)
        
        log.info(f"Computing enhanced fire metrics for {self.forecast_days} forecast days")
        
        # Compute spatial neighborhood CSI for each forecast day
        self._compute_spatial_csi()
        
        # Compute standard metrics for compatibility (using day 1 or aggregated)
        self._compute_standard_metrics()
        
        return super().model_post_init(__context)

    def _compute_spatial_csi(self):
        """Compute CSI at different spatial neighborhoods for each forecast day."""
        neighborhoods = [40, 80, 120]  # km
        
        for day in range(self.forecast_days):
            log.info(f"Computing spatial CSI for forecast day {day + 1}")
            
            # Extract predictions and targets for this day
            if self.forecast_days > 1:
                if self.pred.ndim == 3:
                    day_pred = self.pred[day]  # [grid_cells, time]
                    day_target = self.target[day] if self.target.ndim == 3 else self.target
                else:
                    day_pred = self.pred
                    day_target = self.target
            else:
                day_pred = self.pred
                day_target = self.target
            
            # Reshape to 2D spatial grid for neighborhood analysis
            if day_pred.ndim == 2:
                # [grid_cells, time] - need to aggregate or select time
                # For now, take maximum over time (worst case scenario)
                pred_2d = np.max(day_pred, axis=1) if day_pred.shape[1] > 1 else day_pred[:, 0]
                target_2d = np.max(day_target, axis=1) if day_target.shape[1] > 1 else day_target[:, 0]
            else:
                pred_2d = day_pred.flatten()
                target_2d = day_target.flatten()
            
            # Convert to spatial grid (assuming square grid)
            grid_size = int(np.sqrt(len(pred_2d)))
            if grid_size * grid_size != len(pred_2d):
                log.warning(f"Cannot form square grid from {len(pred_2d)} cells, using flattened analysis")
                pred_spatial = pred_2d.reshape(-1, 1)
                target_spatial = target_2d.reshape(-1, 1)
            else:
                pred_spatial = pred_2d.reshape(grid_size, grid_size)
                target_spatial = target_2d.reshape(grid_size, grid_size)
            
            # Find optimal threshold and CSI for each neighborhood
            for neighborhood in neighborhoods:
                best_csi, best_threshold = self._find_optimal_spatial_csi(
                    pred_spatial, target_spatial, neighborhood
                )
                
                # Store results
                if neighborhood == 40:
                    self.csi_40km[day] = best_csi
                    self.best_thresholds_40km[day] = best_threshold
                elif neighborhood == 80:
                    self.csi_80km[day] = best_csi
                    self.best_thresholds_80km[day] = best_threshold
                elif neighborhood == 120:
                    self.csi_120km[day] = best_csi
                    self.best_thresholds_120km[day] = best_threshold
                
                log.info(f"  Day {day + 1}, {neighborhood}km: CSI={best_csi:.4f}, Threshold={best_threshold:.4f}")

    def _find_optimal_spatial_csi(self, pred_spatial, target_spatial, neighborhood_km):
        """Find optimal threshold for maximum CSI at given neighborhood."""
        thresholds = np.linspace(0.001, 0.999, 100)
        best_csi = 0.0
        best_threshold = 0.5
        
        for threshold in thresholds:
            csi = self._calculate_spatial_csi(pred_spatial, target_spatial, threshold, neighborhood_km)
            if csi > best_csi:
                best_csi = csi
                best_threshold = threshold
        
        return best_csi, best_threshold

    def _calculate_spatial_csi(self, pred_spatial, target_spatial, threshold, neighborhood_km):
        """Calculate CSI considering spatial neighborhoods."""
        # Binarize predictions
        pred_binary = (pred_spatial >= threshold).astype(int)
        
        if neighborhood_km == 40:  # Direct hit (grid resolution)
            hits = np.logical_and(target_spatial > 0, pred_binary > 0)
            hits_count = np.sum(hits)
        else:
            # Calculate neighborhood hits
            hits_count = self._calculate_neighborhood_hits(
                target_spatial, pred_binary, neighborhood_km
            )
        
        # Calculate metrics
        observed_fires = np.sum(target_spatial > 0)
        predicted_fires = np.sum(pred_binary > 0)
        
        misses = max(0, observed_fires - hits_count)
        false_alarms = max(0, predicted_fires - hits_count)
        
        # CSI calculation
        denominator = hits_count + misses + false_alarms
        csi = hits_count / denominator if denominator > 0 else 0.0
        
        return min(1.0, csi)  # Ensure CSI <= 1

    def _calculate_neighborhood_hits(self, obs, pred_binary, neighborhood_km):
        """Calculate hits considering neighborhood around observed fires."""
        if obs.ndim == 1:
            # Flattened case - treat as direct hits
            return np.sum(np.logical_and(obs > 0, pred_binary > 0))
        
        mask = self._create_neighborhood_mask(neighborhood_km)
        obs_fires = obs > 0
        hits_count = 0
        
        # Get coordinates of all observed fires
        fire_locations = np.where(obs_fires)
        
        for i in range(len(fire_locations[0])):
            fire_y, fire_x = fire_locations[0][i], fire_locations[1][i]
            
            # Define neighborhood bounds
            mask_size = mask.shape[0]
            half_size = mask_size // 2
            
            y_start = max(0, fire_y - half_size)
            y_end = min(obs.shape[0], fire_y + half_size + 1)
            x_start = max(0, fire_x - half_size)
            x_end = min(obs.shape[1], fire_x + half_size + 1)
            
            # Extract neighborhood predictions
            neighborhood_pred = pred_binary[y_start:y_end, x_start:x_end]
            
            # Extract corresponding mask portion
            mask_y_start = max(0, half_size - fire_y) if fire_y < half_size else 0
            mask_y_end = mask_y_start + (y_end - y_start)
            mask_x_start = max(0, half_size - fire_x) if fire_x < half_size else 0
            mask_x_end = mask_x_start + (x_end - x_start)
            
            if (mask_y_end <= mask.shape[0] and mask_x_end <= mask.shape[1] and
                mask_y_start >= 0 and mask_x_start >= 0):
                neighborhood_mask = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
                
                # Check if any prediction within the masked neighborhood
                if neighborhood_pred.shape == neighborhood_mask.shape and np.any(neighborhood_pred & neighborhood_mask):
                    hits_count += 1
        
        return hits_count

    def _create_neighborhood_mask(self, size_km):
        """Create a circular neighborhood mask for the given size in km."""
        radius_pixels = size_km / self.grid_resolution_km
        size = int(2 * radius_pixels + 1)
        center = size // 2
        
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius_pixels**2
        return mask

    def _compute_standard_metrics(self):
        """Compute standard metrics for compatibility."""
        # Use first day or aggregate across days
        if self.forecast_days > 1 and self.pred.ndim == 3:
            # Use day 1 for standard metrics
            pred_std = self.pred[0]
            target_std = self.target[0] if self.target.ndim == 3 else self.target
        else:
            pred_std = self.pred
            target_std = self.target
        
        if pred_std.ndim == 1:
            pred_std = np.expand_dims(pred_std, axis=0)
        if target_std.ndim == 1:
            target_std = np.expand_dims(target_std, axis=0)
        
        ngrid = pred_std.shape[0]
        
        # Initialize arrays
        self.csi = np.full(ngrid, np.nan)
        self.pod = np.full(ngrid, np.nan)
        self.far = np.full(ngrid, np.nan)
        self.auc = np.full(ngrid, np.nan)
        self.brier_score = np.full(ngrid, np.nan)
        
        # Compute for each grid cell (simplified version)
        for i in range(min(ngrid, 1000)):  # Limit for performance
            if pred_std.shape[1] > 1:
                pred_cell = pred_std[i, :]
                target_cell = target_std[i, :]
            else:
                pred_cell = pred_std[i]
                target_cell = target_std[i]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(pred_cell) | np.isnan(target_cell))
            if not np.any(valid_mask) or len(np.unique(target_cell[valid_mask])) < 2:
                continue
            
            pred_valid = pred_cell[valid_mask]
            target_valid = target_cell[valid_mask]
            
            # Find best threshold
            thresholds = np.linspace(0.1, 0.9, 9)
            best_csi = 0
            best_metrics = {'csi': 0, 'pod': 0, 'far': 0}
            
            for threshold in thresholds:
                pred_binary = (pred_valid >= threshold).astype(int)
                tp, fp, tn, fn = self._confusion_matrix_elements(pred_binary, target_valid)
                
                csi = self._critical_success_index(tp, fp, fn)
                if csi > best_csi:
                    best_csi = csi
                    best_metrics = {
                        'csi': csi,
                        'pod': self._probability_of_detection(tp, fn),
                        'far': self._false_alarm_rate(fp, tn)
                    }
            
            self.csi[i] = best_metrics['csi']
            self.pod[i] = best_metrics['pod']
            self.far[i] = best_metrics['far']
            self.brier_score[i] = self._brier_score(pred_valid, target_valid)

    def _confusion_matrix_elements(self, pred_binary, target):
        """Compute confusion matrix elements."""
        tp = np.sum((pred_binary == 1) & (target == 1))
        fp = np.sum((pred_binary == 1) & (target == 0))
        tn = np.sum((pred_binary == 0) & (target == 0))
        fn = np.sum((pred_binary == 0) & (target == 1))
        return tp, fp, tn, fn

    def _critical_success_index(self, tp, fp, fn):
        """Compute Critical Success Index."""
        denominator = tp + fp + fn
        return tp / denominator if denominator > 0 else np.nan

    def _probability_of_detection(self, tp, fn):
        """Compute Probability of Detection."""
        denominator = tp + fn
        return tp / denominator if denominator > 0 else np.nan

    def _false_alarm_rate(self, fp, tn):
        """Compute False Alarm Rate."""
        denominator = fp + tn
        return fp / denominator if denominator > 0 else np.nan

    def _brier_score(self, pred, target):
        """Compute Brier Score."""
        return np.mean((pred - target) ** 2)

    def get_forecast_summary(self) -> Dict:
        """Get summary of multi-day forecast performance."""
        summary = {
            'forecast_days': self.forecast_days,
            'neighborhoods': {
                '40km': {
                    'csi_by_day': dict(self.csi_40km),
                    'mean_csi': np.mean(list(self.csi_40km.values())) if self.csi_40km else 0.0,
                    'csi_degradation': self._calculate_degradation(self.csi_40km),
                    'best_thresholds': dict(self.best_thresholds_40km)
                },
                '80km': {
                    'csi_by_day': dict(self.csi_80km),
                    'mean_csi': np.mean(list(self.csi_80km.values())) if self.csi_80km else 0.0,
                    'csi_degradation': self._calculate_degradation(self.csi_80km),
                    'best_thresholds': dict(self.best_thresholds_80km)
                },
                '120km': {
                    'csi_by_day': dict(self.csi_120km),
                    'mean_csi': np.mean(list(self.csi_120km.values())) if self.csi_120km else 0.0,
                    'csi_degradation': self._calculate_degradation(self.csi_120km),
                    'best_thresholds': dict(self.best_thresholds_120km)
                }
            }
        }
        return summary

    def _calculate_degradation(self, csi_by_day: Dict[int, float]) -> Dict:
        """Calculate how CSI degrades over forecast days."""
        if len(csi_by_day) < 2:
            return {'total_degradation': 0.0, 'daily_degradation': 0.0}
        
        days = sorted(csi_by_day.keys())
        csi_values = [csi_by_day[day] for day in days]
        
        total_degradation = csi_values[0] - csi_values[-1]
        daily_degradation = total_degradation / (len(days) - 1) if len(days) > 1 else 0.0
        
        return {
            'total_degradation': total_degradation,
            'daily_degradation': daily_degradation,
            'day_1_csi': csi_values[0],
            'final_day_csi': csi_values[-1]
        }

    def dump_metrics(self, path: str) -> None:
        """Dump all fire metrics including multi-day forecast analysis."""
        os.makedirs(path, exist_ok=True)
        
        # Enhanced forecast summary
        forecast_summary = self.get_forecast_summary()
        forecast_path = os.path.join(path, 'fire_forecast_metrics.json')
        with open(forecast_path, 'w') as f:
            json.dump(forecast_summary, f, indent=4)
        
        # Standard metrics for compatibility
        if len(self.csi) > 0:
            stats = self.calc_stats()
            stats_path = os.path.join(path, 'fire_metrics_agg.json')
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
        
        log.info(f"Enhanced fire metrics saved to {path}")
        
        # Print summary to console
        self._print_forecast_summary(forecast_summary)

    def _print_forecast_summary(self, summary: Dict):
        """Print forecast summary to console."""
        print("\n=== MULTI-DAY FIRE FORECAST PERFORMANCE ===")
        print(f"Forecast horizon: {summary['forecast_days']} days")
        
        for neighborhood, metrics in summary['neighborhoods'].items():
            print(f"\n{neighborhood} neighborhood:")
            print(f"  Mean CSI: {metrics['mean_csi']:.4f}")
            print(f"  CSI degradation: {metrics['csi_degradation']['total_degradation']:.4f} total")
            print(f"  Daily degradation: {metrics['csi_degradation']['daily_degradation']:.4f} per day")
            
            print("  Day-by-day CSI:")
            for day, csi in metrics['csi_by_day'].items():
                threshold = metrics['best_thresholds'].get(day, 0.5)
                print(f"    Day {day + 1}: CSI={csi:.4f}, Threshold={threshold:.4f}")

    def calc_stats(self) -> Dict:
        """Calculate aggregate statistics for compatibility."""
        stats = {}
        
        if len(self.csi) > 0:
            metrics_dict = {
                'csi': self.csi,
                'pod': self.pod,
                'far': self.far,
                'brier_score': self.brier_score,
            }
            
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
        
        return stats

    @property
    def ngrid(self) -> int:
        """Number of grid cells."""
        return self.pred.shape[-2] if self.pred.ndim >= 2 else len(self.pred)

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return self.pred.shape[-1] if self.pred.ndim >= 2 else 1