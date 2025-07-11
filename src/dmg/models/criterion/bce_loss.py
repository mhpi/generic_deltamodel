import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)

class BceLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for fire occurrence prediction.
    
    This follows the same interface as your existing loss functions (like RmseLoss)
    so it can be a drop-in replacement in your loss function factory.
    """
    
    def __init__(self, pos_weight=None, reduction='mean', **kwargs):
        super().__init__()
        
        # Handle case where pos_weight is passed as config dict or other non-numeric type
        if pos_weight is not None:
            if isinstance(pos_weight, dict):
                # If it's a config dict, ignore it and use None
                self.pos_weight = None
                log.info("Ignoring pos_weight config dict, using None")
            elif isinstance(pos_weight, (int, float)):
                self.pos_weight = float(pos_weight)
            else:
                # Try to convert to float, fall back to None
                try:
                    self.pos_weight = float(pos_weight)
                except (ValueError, TypeError):
                    log.warning(f"Could not convert pos_weight {pos_weight} to float, using None")
                    self.pos_weight = None
        else:
            self.pos_weight = None
            
        # Handle case where reduction is passed as device or other invalid type
        if isinstance(reduction, torch.device):
            self.reduction = 'mean'
            log.info("Got device as reduction parameter, using 'mean' instead")
        elif isinstance(reduction, str) and reduction in ['none', 'mean', 'sum']:
            self.reduction = reduction
        elif str(reduction).startswith('cuda:'):
            self.reduction = 'mean'
            log.info(f"Got device string '{reduction}' as reduction parameter, using 'mean' instead")
        else:
            log.warning(f"Invalid reduction parameter '{reduction}', using 'mean' instead")
            self.reduction = 'mean'
        
        # Log the loss function initialization
        log.info(f"Initialized BceLoss with pos_weight={self.pos_weight}, reduction={self.reduction}")
        
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
        if self.pos_weight is not None:
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



class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance in fire occurrence prediction.
    
    This is particularly useful for fire data where positive cases (fires) are rare.
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        log.info(f"Initialized FocalLoss with alpha={alpha}, gamma={gamma}, reduction={reduction}")
        
    def forward(self, predictions, targets):
        """
        Compute focal loss for fire occurrence predictions.
        
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
        # Ensure same shape
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if targets.dim() == 3 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
        
        # Flatten
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove NaN values
        valid_mask = ~(torch.isnan(target_flat) | torch.isnan(pred_flat))
        
        if valid_mask.sum() == 0:
            log.warning("No valid values found in Focal loss computation")
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        # Ensure targets are in [0, 1] range
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Apply sigmoid to get probabilities from logits
        pred_probs = torch.sigmoid(pred_valid)
        
        # Compute binary cross-entropy
        ce_loss = F.binary_cross_entropy(pred_probs, target_valid, reduction='none')
        
        # Compute pt (probability of true class)
        pt = torch.where(target_valid == 1, pred_probs, 1 - pred_probs)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss