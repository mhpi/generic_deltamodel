import torch
import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

def create_variable_mapping(pretrained_vars: List[str], 
                           finetuning_vars: List[str]) -> Tuple[Dict[int, int], List[bool]]:
    """
    Create mapping between pretrained and fine-tuning variables.
    
    Args:
        pretrained_vars: List of variable names from pretrained model
        finetuning_vars: List of variable names available for fine-tuning
        
    Returns:
        mapping: Dict mapping pretrained indices to fine-tuning indices
        missing_mask: Boolean list indicating which pretrained vars are missing
    """
    mapping = {}
    missing_mask = []
    
    for i, pretrained_var in enumerate(pretrained_vars):
        if pretrained_var in finetuning_vars:
            j = finetuning_vars.index(pretrained_var)
            mapping[i] = j
            missing_mask.append(False)
        else:
            missing_mask.append(True)
    
    log.info(f"Variable mapping: {len(mapping)}/{len(pretrained_vars)} variables available")
    missing_vars = [var for i, var in enumerate(pretrained_vars) if missing_mask[i]]
    if missing_vars:
        log.info(f"Missing variables: {missing_vars}")
    
    return mapping, missing_mask

def prepare_masked_input(finetuning_data: torch.Tensor, 
                        variable_mapping: Dict[int, int],
                        missing_mask: List[bool],
                        n_pretrained_vars: int) -> torch.Tensor:
    """
    Convert fine-tuning data to pretrained model format with proper masking.
    
    Args:
        finetuning_data: Input data in fine-tuning format
        variable_mapping: Mapping from pretrained to fine-tuning indices
        missing_mask: Boolean mask for missing variables
        n_pretrained_vars: Number of variables expected by pretrained model
        
    Returns:
        masked_data: Data in pretrained format with NaN for missing variables
    """
    if finetuning_data.dim() == 3:  # Time series data [batch, seq, vars]
        batch_size, seq_len, _ = finetuning_data.shape
        masked_data = torch.full(
            (batch_size, seq_len, n_pretrained_vars),
            float('nan'),
            device=finetuning_data.device,
            dtype=finetuning_data.dtype
        )
    else:  # Static data [batch, vars]
        batch_size, _ = finetuning_data.shape
        masked_data = torch.full(
            (batch_size, n_pretrained_vars),
            float('nan'),
            device=finetuning_data.device,
            dtype=finetuning_data.dtype
        )
    
    # Fill in available variables
    for pretrained_idx, finetuning_idx in variable_mapping.items():
        if finetuning_data.dim() == 3:
            masked_data[:, :, pretrained_idx] = finetuning_data[:, :, finetuning_idx]
        else:
            masked_data[:, pretrained_idx] = finetuning_data[:, finetuning_idx]
    
    return masked_data

