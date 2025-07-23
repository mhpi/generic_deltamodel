import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """
    Improved Focal Loss for handling severe class imbalance in fire occurrence prediction.
    Compatible with your factory system.
    """
    
    def __init__(self, config, device, y_obs=None, **kwargs):
        super().__init__()
        
        # Extract parameters from config with better defaults for fire data
        self.alpha = config.get('alpha', 0.75)  # Higher alpha for rare positive class
        self.gamma = config.get('gamma', 3.0)   # Higher gamma for more focus on hard examples
        self.reduction = config.get('reduction', 'mean')
        
        # Scale factor to prevent loss from being too small
        self.scale_factor = config.get('scale_factor', 100.0)
        
        # Handle device
        if isinstance(device, str):
            self.device = device
        elif hasattr(device, 'type'):
            self.device = str(device)
        else:
            self.device = 'cpu'
        
        # Statistics tracking
        self.call_count = 0
        
        log.info(f"Initialized FocalLoss with alpha={self.alpha}, gamma={self.gamma}, scale_factor={self.scale_factor}, reduction={self.reduction}")
        
    def forward(self, predictions, targets):
        """
        Compute focal loss for fire occurrence predictions.
        """
        self.call_count += 1
        
        # Debug logging for first few calls
        if self.call_count <= 3:
            log.info(f"FocalLoss Call #{self.call_count}:")
            log.info(f"  Predictions shape: {predictions.shape}, range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
            log.info(f"  Targets shape: {targets.shape}, range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
        
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
        
        # Debug class distribution on first few calls
        if self.call_count <= 3:
            fire_count = torch.sum(target_valid > 0.5).item()
            total_count = len(target_valid)
            fire_rate = fire_count / total_count if total_count > 0 else 0
            log.info(f"  Fire rate: {fire_rate:.6f} ({fire_count}/{total_count})")
        
        # Ensure targets are in [0, 1] range
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Use binary_cross_entropy_with_logits for numerical stability
        # Then apply focal weighting
        ce_loss = F.binary_cross_entropy_with_logits(pred_valid, target_valid, reduction='none')
        
        # Get probabilities for focal weight calculation
        pred_probs = torch.sigmoid(pred_valid)
        
        # Compute pt (probability of true class)
        pt = torch.where(target_valid == 1, pred_probs, 1 - pred_probs)
        
        # Compute focal weight with numerical stability
        # Use different alpha for positive and negative classes
        alpha_factor = torch.where(target_valid == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * torch.pow(torch.clamp(1 - pt, min=1e-8), self.gamma)
        
        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss
        
        # Scale the loss to prevent it from being too small
        focal_loss = focal_loss * self.scale_factor
        
        # Debug loss values on first few calls
        if self.call_count <= 3:
            log.info(f"  CE loss range: [{ce_loss.min().item():.6f}, {ce_loss.max().item():.6f}]")
            log.info(f"  Focal weight range: [{focal_weight.min().item():.6f}, {focal_weight.max().item():.6f}]")
            log.info(f"  Final focal loss range: [{focal_loss.min().item():.6f}, {focal_loss.max().item():.6f}]")
        
        if self.reduction == 'mean':
            final_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            final_loss = focal_loss.sum()
        else:
            final_loss = focal_loss
        
        # Debug final loss value
        if self.call_count <= 3:
            log.info(f"  Final loss value: {final_loss.item():.6f}")
            
            # Check if loss seems reasonable
            if final_loss.item() < 0.1:
                log.warning(f"Loss value {final_loss.item():.6f} seems too low for fire prediction")
            elif final_loss.item() > 100.0:
                log.warning(f"Loss value {final_loss.item():.6f} seems too high, consider reducing scale_factor")
        
        return final_loss