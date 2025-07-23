import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

log = logging.getLogger(__name__)

class CalibrationBceLoss(nn.Module):
    """
    BCE Loss with calibration penalty to reduce overconfidence.
    """
    
    def __init__(self, config, device, y_obs=None, **kwargs):
        super().__init__()
        
        self.pos_weight = config.get('pos_weight', 80.0)
        self.calibration_weight = config.get('calibration_weight', 0.1)  # Weight for calibration term
        self.confidence_penalty = config.get('confidence_penalty', 0.05)  # Penalty for high confidence
        self.reduction = config.get('reduction', 'mean')
        
        self.device = device if isinstance(device, str) else str(device)
        
        log.info(f"CalibrationBCELoss: pos_weight={self.pos_weight}, "
                f"calibration_weight={self.calibration_weight}, "
                f"confidence_penalty={self.confidence_penalty}")
    
    def forward(self, predictions, targets):
        """Compute calibration-aware BCE loss"""
        
        # Handle shape differences
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if targets.dim() == 3 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
        
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove NaN values
        valid_mask = ~(torch.isnan(target_flat) | torch.isnan(pred_flat))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Standard BCE loss with pos_weight
        pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_valid, target_valid, 
            pos_weight=pos_weight_tensor,
            reduction=self.reduction
        )
        
        # Calibration penalty: discourage extreme confidence
        probs = torch.sigmoid(pred_valid)
        
        # Penalty for being too confident when wrong
        confidence_penalty = 0
        if self.confidence_penalty > 0:
            # High confidence on wrong predictions gets penalized
            wrong_positive = (probs > 0.8) & (target_valid < 0.5)  # Confident fire, but no fire
            wrong_negative = (probs < 0.2) & (target_valid > 0.5)  # Confident no fire, but fire
            
            confidence_penalty = self.confidence_penalty * (
                torch.sum(probs[wrong_positive] ** 2) + 
                torch.sum((1 - probs[wrong_negative]) ** 2)
            ) / len(pred_valid)
        
        # Entropy regularization to prevent overconfidence
        entropy_reg = 0
        if self.calibration_weight > 0:
            # Encourage more uncertain predictions
            entropy = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
            entropy_reg = -self.calibration_weight * torch.mean(entropy)  # Negative because we want to maximize entropy
        
        total_loss = bce_loss + confidence_penalty + entropy_reg
        
        return total_loss


class DistributionLoss(nn.Module):
    """
    Loss that explicitly matches the distribution of predictions to observations.
    This helps with calibration by ensuring prediction distribution matches reality.
    """
    
    def __init__(self, config, device, y_obs=None, **kwargs):
        super().__init__()
        
        self.pos_weight = config.get('pos_weight', 80.0)
        self.distribution_weight = config.get('distribution_weight', 0.2)
        self.n_bins = config.get('n_bins', 10)
        self.reduction = config.get('reduction', 'mean')
        
        self.device = device if isinstance(device, str) else str(device)
        
        log.info(f"DistributionLoss: pos_weight={self.pos_weight}, "
                f"distribution_weight={self.distribution_weight}, n_bins={self.n_bins}")
    
    def forward(self, predictions, targets):
        """Compute distribution-aware loss"""
        
        # Handle shape differences
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if targets.dim() == 3 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
        
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove NaN values
        valid_mask = ~(torch.isnan(target_flat) | torch.isnan(pred_flat))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Standard BCE loss
        pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_valid, target_valid, 
            pos_weight=pos_weight_tensor,
            reduction=self.reduction
        )
        
        # Distribution matching loss
        probs = torch.sigmoid(pred_valid)
        
        # Create bins for calibration
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=predictions.device)
        
        distribution_loss = 0
        for i in range(self.n_bins):
            # Find predictions in this bin
            if i == self.n_bins - 1:
                bin_mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            else:
                bin_mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            
            if bin_mask.sum() > 10:  # Only if we have enough samples
                bin_preds = probs[bin_mask]
                bin_targets = target_valid[bin_mask]
                
                # Expected vs actual fire rate in this bin
                expected_fire_rate = torch.mean(bin_preds)
                actual_fire_rate = torch.mean(bin_targets)
                
                # Penalize difference between expected and actual
                distribution_loss += torch.abs(expected_fire_rate - actual_fire_rate)
        
        distribution_loss = distribution_loss / self.n_bins * self.distribution_weight
        
        total_loss = bce_loss + distribution_loss
        
        return total_loss


class LabelSmoothingBCE(nn.Module):
    """
    Label smoothing for fire prediction to reduce overconfidence.
    """
    
    def __init__(self, config, device, y_obs=None, **kwargs):
        super().__init__()
        
        self.pos_weight = config.get('pos_weight', 80.0)
        self.smoothing = config.get('label_smoothing', 0.1)  # Smoothing factor
        self.reduction = config.get('reduction', 'mean')
        
        self.device = device if isinstance(device, str) else str(device)
        
        log.info(f"LabelSmoothingBCE: pos_weight={self.pos_weight}, smoothing={self.smoothing}")
    
    def forward(self, predictions, targets):
        """Compute label-smoothed BCE loss"""
        
        # Handle shape differences
        if predictions.dim() != targets.dim():
            if predictions.dim() == 3 and predictions.shape[-1] == 1:
                predictions = predictions.squeeze(-1)
            if targets.dim() == 3 and targets.shape[-1] == 1:
                targets = targets.squeeze(-1)
        
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        # Remove NaN values
        valid_mask = ~(torch.isnan(target_flat) | torch.isnan(pred_flat))
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        target_valid = torch.clamp(target_valid, 0.0, 1.0)
        
        # Apply label smoothing
        # Fire labels (1) become (1 - smoothing)
        # No-fire labels (0) become (smoothing)
        smoothed_targets = target_valid * (1 - self.smoothing) + self.smoothing * 0.5
        
        # BCE loss with smoothed targets
        pos_weight_tensor = torch.tensor(self.pos_weight, device=predictions.device)
        loss = F.binary_cross_entropy_with_logits(
            pred_valid, smoothed_targets, 
            pos_weight=pos_weight_tensor,
            reduction=self.reduction
        )
        
        return loss