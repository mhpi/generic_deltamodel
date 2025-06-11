from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from numpy.typing import NDArray


class HydroLoss(torch.nn.Module):
    """ hydrological loss function  integrating RMSE, NSE, and FDC metrics.
    Parameters
    ----------
    target : np.ndarray
        The target data array.
    config : dict
        The configuration dictionary.
    device : str, optional
        The device to use for the loss function object. The default is 'cpu'.
        
    Optional Parameters: (Set in config)
    --------------------
    rmse_weight : float
        Weight for RMSE component. Default is 0.3.
    nse_weight : float
        Weight for NSE component. Default is 0.3.
    fdc_weight : float
        Weight for FDC component. Default is 0.4.
    fdc_points : List[float]
        Percentile points for FDC calculation. Default is [0.05, 0.2, 0.7, 0.95].
        These points segment the FDC into high flows (0-0.05), wet conditions (0.05-0.2),
        mid-range flows (0.2-0.7), dry conditions (0.7-0.95), and low flows (0.95-1.0).
    segment_weights : List[float]
        Weights for each FDC segment. Default is [0.15, 0.2, 0.3, 0.2, 0.15].
        Must have length = len(fdc_points) + 1.
    epsilon : float
        Stability term to prevent division by zero. Default is 1e-6.
    """
    
    def __init__(
        self,
        target: NDArray[np.float32],
        config: Dict[str, Any],
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        
        # Get weights for each loss component
        self.rmse_weight = config.get('rmse_weight', 0.3)
        self.nse_weight = config.get('nse_weight', 0.3)
        self.fdc_weight = config.get('fdc_weight', 0.4)
        
        # FDC parameters
        self.fdc_points = config.get('fdc_points', [0.05, 0.2, 0.7, 0.95])
        self.segment_weights = config.get('segment_weights', [0.15, 0.2, 0.3, 0.2, 0.15])
        
        # Stability terms
        self.epsilon = config.get('epsilon', 1e-6)
        
        # Store target statistics for NSE calculation
        self.std = np.nanstd(target[:, :, 0], axis=0)
        self.mean = np.nanmean(target[:, :, 0], axis=0)
        
        # Pre-compute target FDC for each basin
        self._precompute_target_fdc(target)
        
    def _precompute_target_fdc(self, target: NDArray[np.float32]) -> None:
        """Pre-compute target FDC percentiles for each basin.
        
        Parameters
        ----------
        target : np.ndarray
            The target data array.
        """
        # Get flow data from target (first variable)
        flow_data = target[:, :, 0]  # shape: [time, basins]
        n_basins = flow_data.shape[1]
        
        # Initialize storage for FDC percentiles
        self.target_fdc_percentiles = np.zeros((len(self.fdc_points), n_basins))
        
        # Calculate percentiles for each basin
        for basin_idx in range(n_basins):
            basin_flow = flow_data[:, basin_idx]
            valid_flow = basin_flow[torch.isnan(basin_flow) == False]
            
            if len(valid_flow) > 0:
                self.target_fdc_percentiles[:, basin_idx] = np.percentile(
                    valid_flow, 
                    [p * 100 for p in self.fdc_points]
                )
            else:
                self.target_fdc_percentiles[:, basin_idx] = np.zeros(len(self.fdc_points))
    
    def _compute_rmse_loss(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute RMSE loss.
        
        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
        target : torch.Tensor
            The observed values.
            
        Returns
        -------
        torch.Tensor
            The RMSE loss.
        """
        # Mask where observations are valid (not NaN)
        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]
        
        if len(t_sub) > 0:
            rmse = torch.sqrt(((p_sub - t_sub) ** 2).mean())
        else:
            rmse = torch.tensor(0.0, device=self.device)
            
        return rmse
    
    def _compute_nse_loss(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        n_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute NSE loss (1 - NSE, so lower is better).
        
        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
        target : torch.Tensor
            The observed values.
        n_samples : torch.Tensor
            The indices of basins in the batch.
            
        Returns
        -------
        torch.Tensor
            The NSE loss.
        """
        n_samples = n_samples.astype(int)
        n_timesteps = target.shape[0]
        
        # Prepare basin-specific mean and std for NSE calculation
        mean_batch = torch.tensor(
            np.tile(self.mean[n_samples].T, (n_timesteps, 1)),
            dtype=torch.float32,
            requires_grad=False,
            device=self.device
        )
        
        std_batch = torch.tensor(
            np.tile(self.std[n_samples].T, (n_timesteps, 1)),
            dtype=torch.float32,
            requires_grad=False,
            device=self.device
        )
        
        # Mask where observations are valid (not NaN)
        mask = ~torch.isnan(target)
        p_sub = prediction[mask]
        t_sub = target[mask]
        mean_sub = mean_batch[mask]
        std_sub = std_batch[mask]
        
        if len(t_sub) > 0:
            # Compute NSE components
            numerator = ((p_sub - t_sub) ** 2).sum()
            denominator = ((t_sub - mean_sub) ** 2).sum() + self.epsilon
            
            # 1 - NSE (lower is better)
            nse_loss = numerator / denominator
        else:
            nse_loss = torch.tensor(0.0, device=self.device)
            
        return nse_loss
    
    def _compute_fdc_percentiles(
        self, 
        flow_series: torch.Tensor
    ) -> torch.Tensor:
        """Compute FDC percentiles for a flow series.
        
        Parameters
        ----------
        flow_series : torch.Tensor
            Flow time series.
            
        Returns
        -------
        torch.Tensor
            FDC percentiles at the specified points.
        """
        # Remove NaN values
        valid_flow = flow_series[~torch.isnan(flow_series)]
        
        if len(valid_flow) > 0:
            # Sort flow values in descending order
            sorted_flow, _ = torch.sort(valid_flow, descending=True)
            
            # Calculate percentile indices
            percentile_indices = [int(p * len(sorted_flow)) for p in self.fdc_points]
            percentile_indices = [min(i, len(sorted_flow) - 1) for i in percentile_indices]
            
            # Extract percentile values
            percentiles = torch.tensor([sorted_flow[i] for i in percentile_indices], device=self.device)
        else:
            percentiles = torch.zeros(len(self.fdc_points), device=self.device)
            
        return percentiles
    
    def _compute_fdc_loss(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        n_samples: torch.Tensor
    ) -> torch.Tensor:
        """Compute FDC-based loss.
        
        Parameters
        ----------
        prediction : torch.Tensor
            The predicted values.
        target : torch.Tensor
            The observed values.
        n_samples : torch.Tensor
            The indices of basins in the batch.
            
        Returns
        -------
        torch.Tensor
            The FDC-based loss.
        """
        n_samples = n_samples.astype(int)
        batch_size = len(n_samples)
        
        # Initialize loss
        fdc_loss = torch.tensor(0.0, device=self.device)
        
        for b in range(batch_size):
            # Get flow series for this basin
            pred_flow = prediction[:, b]
            obs_flow = target[:, b]
            
            # Compute FDC percentiles for predicted flow
            pred_percentiles = self._compute_fdc_percentiles(pred_flow)
            
            # Get pre-computed target percentiles for this basin
            obs_percentiles = torch.tensor(
                self.target_fdc_percentiles[:, n_samples[b]],
                dtype=torch.float32,
                device=self.device
            )
            
            # Calculate segment losses (using RMSE between percentiles)
            segment_losses = []
            
            # First segment (0 to first percentile)
            if obs_percentiles[0] > 0:
                segment_losses.append(
                    torch.abs(pred_percentiles[0] - obs_percentiles[0]) / obs_percentiles[0]
                )
            else:
                segment_losses.append(torch.abs(pred_percentiles[0]))
            
            # Middle segments
            for i in range(len(self.fdc_points) - 1):
                if obs_percentiles[i] > 0 and obs_percentiles[i+1] > 0:
                    segment_loss = torch.abs(
                        pred_percentiles[i+1] / pred_percentiles[i] - 
                        obs_percentiles[i+1] / obs_percentiles[i]
                    )
                    segment_losses.append(segment_loss)
                else:
                    # Fallback for zero values
                    segment_losses.append(
                        torch.abs(pred_percentiles[i+1] - obs_percentiles[i+1] + self.epsilon)
                    )
            
            # Last segment (beyond last percentile to min flow)
            min_pred = torch.min(pred_flow[~torch.isnan(pred_flow)])
            
            # Fixed: Handle target dimension correctly
            # The error occurs because target is 2D [time, batch], not 3D [time, batch, features]
            min_obs_array = obs_flow[~torch.isnan(obs_flow)]
            if len(min_obs_array) > 0:
                min_obs = torch.min(min_obs_array)
            else:
                min_obs = torch.tensor(0.0, device=self.device)
            
            if min_obs > 0 and obs_percentiles[-1] > 0:
                segment_losses.append(
                    torch.abs(min_pred / pred_percentiles[-1] - min_obs / obs_percentiles[-1])
                )
            else:
                segment_losses.append(torch.abs(min_pred - min_obs + self.epsilon))
            
            # Combine segment losses with weights
            basin_fdc_loss = sum(w * l for w, l in zip(self.segment_weights, segment_losses))
            fdc_loss += basin_fdc_loss
        
        # Average across basins
        if batch_size > 0:
            fdc_loss /= batch_size
            
        return fdc_loss
        
    def forward(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_obs : torch.Tensor
            The observed values.
        n_samples : torch.Tensor
            The batch indices.
            
        Returns
        -------
        torch.Tensor
            The combined loss value.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]
        
        # Initialize loss components
        rmse_loss = torch.tensor(0.0, device=self.device)
        nse_loss = torch.tensor(0.0, device=self.device)
        fdc_loss = torch.tensor(0.0, device=self.device)
        
        if len(target) > 0:
            # Compute individual loss components
            rmse_loss = self._compute_rmse_loss(prediction, target)
            nse_loss = self._compute_nse_loss(prediction, target, n_samples)
            fdc_loss = self._compute_fdc_loss(prediction, target, n_samples)
            
            # Combine losses with weights
            combined_loss = (
                self.rmse_weight * rmse_loss + 
                self.nse_weight * nse_loss + 
                self.fdc_weight * fdc_loss
            )
        else:
            combined_loss = torch.tensor(0.0, device=self.device)
            
        return combined_loss
    
    def compute_metrics(
        self,
        y_pred: torch.Tensor,
        y_obs: torch.Tensor,
        n_samples: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute and return individual metrics for analysis.
        
        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y_obs : torch.Tensor
            The observed values.
        n_samples : torch.Tensor
            The batch indices.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with individual metrics.
        """
        prediction = y_pred.squeeze()
        target = y_obs[:, :, 0]
        
        metrics = {}
        
        if len(target) > 0:
            metrics['rmse'] = self._compute_rmse_loss(prediction, target)
            metrics['nse_loss'] = self._compute_nse_loss(prediction, target, n_samples)
            metrics['fdc_loss'] = self._compute_fdc_loss(prediction, target, n_samples)
            
            # Calculate true NSE (not loss version)
            mask = ~torch.isnan(target)
            if mask.any():
                p_sub = prediction[mask]
                t_sub = target[mask]
                
                n_timesteps = target.shape[0]
                mean_batch = torch.tensor(
                    np.tile(self.mean[n_samples].T, (n_timesteps, 1)),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device
                )
                mean_sub = mean_batch[mask]
                
                numerator = ((p_sub - t_sub) ** 2).sum()
                denominator = ((t_sub - mean_sub) ** 2).sum() + self.epsilon
                metrics['nse'] = 1 - (numerator / denominator)
            else:
                metrics['nse'] = torch.tensor(0.0, device=self.device)
                
            # Calculate combined loss
            metrics['combined_loss'] = (
                self.rmse_weight * metrics['rmse'] + 
                self.nse_weight * metrics['nse_loss'] + 
                self.fdc_weight * metrics['fdc_loss']
            )
        
        return metrics