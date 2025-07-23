import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)


class BceLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for fire occurrence prediction.
    
    This follows the EXACT interface as your existing loss functions 
    to work with your factory system.
    """
    
    def __init__(self, config, device, y_obs=None, **kwargs):
        """
        Initialize BCE loss function.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing loss function parameters
        device : str or torch.device
            Device to run on
        y_obs : array-like, optional
            Observed data (not used for BCE but kept for interface compatibility)
        """
        super().__init__()
        
        # Extract pos_weight from config
        self.pos_weight = config.get('pos_weight', None)
        
        # Extract reduction from config 
        self.reduction = config.get('reduction', 'mean')
        
        # Handle device parameter
        if isinstance(device, str):
            self.device = device
        elif hasattr(device, 'type'):
            self.device = str(device)
        else:
            self.device = 'cpu'
        
        # Validate and convert pos_weight
        if self.pos_weight is not None:
            try:
                self.pos_weight = float(self.pos_weight)
                log.info(f"Using pos_weight from config: {self.pos_weight}")
            except (ValueError, TypeError):
                log.warning(f"Could not convert pos_weight '{self.pos_weight}' to float, using None")
                self.pos_weight = None
        
        # Validate reduction parameter
        if self.reduction not in ['none', 'mean', 'sum']:
            log.warning(f"Invalid reduction '{self.reduction}', using 'mean'")
            self.reduction = 'mean'
        
        # Log initialization
        log.info(f"Initialized BceLoss with pos_weight={self.pos_weight}, reduction={self.reduction}, device={self.device}")
        
        # Calculate dynamic pos_weight from data if not provided
        if self.pos_weight is None and y_obs is not None:
            self._calculate_pos_weight_from_data(y_obs)
    
    def _calculate_pos_weight_from_data(self, y_obs):
        """Calculate pos_weight from observed data distribution."""
        try:
            # Convert to numpy if it's a tensor
            if hasattr(y_obs, 'cpu'):
                y_data = y_obs.cpu().numpy()
            else:
                y_data = np.array(y_obs)
            
            # Remove NaN values
            valid_data = y_data[~np.isnan(y_data)]
            
            if len(valid_data) > 0:
                # Count fires vs non-fires
                fire_count = np.sum(valid_data > 0.5)
                no_fire_count = np.sum(valid_data <= 0.5)
                
                if fire_count > 0:
                    calculated_pos_weight = no_fire_count / fire_count
                    # Cap at reasonable values
                    calculated_pos_weight = min(max(calculated_pos_weight, 1.0), 1000.0)
                    
                    self.pos_weight = calculated_pos_weight
                    log.info(f"Calculated pos_weight from data: {self.pos_weight:.2f}")
                    log.info(f"Fire rate in training data: {fire_count / len(valid_data):.6f}")
                else:
                    log.warning("No fires found in training data, using pos_weight=1.0")
                    self.pos_weight = 1.0
            else:
                log.warning("No valid data found for pos_weight calculation")
                self.pos_weight = 1.0
                
        except Exception as e:
            log.warning(f"Error calculating pos_weight from data: {e}")
            self.pos_weight = 1.0
        
    def forward(self, predictions, targets):
        """
        Compute BCE loss for fire occurrence predictions.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions (logits) with shape [time, cells, 1] or [time, cells]
        targets : torch.Tensor  
            Ground truth fire occurrence with shape [time, cells, 1] or [time, cells]
            
        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Ensure predictions and targets have the same shape
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if targets.dim() == 3 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
        
        # Flatten to handle batch processing
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove NaN values (common in gridded data)
        valid_mask = ~(torch.isnan(target_flat) | torch.isnan(pred_flat))
        
        if valid_mask.sum() == 0:
            log.warning("No valid values found in BCE loss computation")
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Ensure targets are in [0, 1] range
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Use binary_cross_entropy_with_logits for numerical stability and pos_weight support
        if self.pos_weight is not None and self.pos_weight != 1.0:
            pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)
            loss = F.binary_cross_entropy_with_logits(
                pred_valid, target_valid, 
                pos_weight=pos_weight_tensor,
                reduction=self.reduction
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                pred_valid, target_valid, 
                reduction=self.reduction
            )
        
        return loss

