import os
import re
import torch
import torch.nn as nn
import logging
import warnings
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union, Optional

from models.neural_networks.cudnn_lstm import CudnnLstm
from models.neural_networks.transformer.MFFormer import Model as MFFormer
from models.neural_networks.transformer.MFFormerTFT import Model as MFFormerTFT
from dmg.core.utils.variable_mapping import create_variable_mapping, prepare_masked_input

warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")
log = logging.getLogger(__name__)

class MaskFinetuneing(nn.Module):
    """
    Fine-tuning module that uses masked variables instead of skipping embedding layers.
    This approach preserves the pretrained model's learned representations.
    """
    
    def __init__(self, config: Union[Dict, DictConfig], ny) -> None:
        super().__init__()
        
        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config
            
        if 'nn_model' in config_dict:
            dpl_config = config_dict
            nn_config = config_dict['nn_model']
        else:
            dpl_config = config_dict.get('delta_model', {})
            nn_config = dpl_config.get('nn_model', {})
        
        # Get pretrained variables from config
        self.pretrained_ts_vars = nn_config.get('pretrained_time_series_vars', [])
        self.pretrained_static_vars = nn_config.get('pretrained_static_vars', [])
        
        if not self.pretrained_ts_vars or not self.pretrained_static_vars:
            raise ValueError("Must specify 'pretrained_time_series_vars' and 'pretrained_static_vars' in config")
        
        # Get fine-tuning variables from config
        self.finetuning_ts_vars = nn_config.get('forcings', [])
        self.finetuning_static_vars = nn_config.get('attributes', [])
        
        if not self.finetuning_ts_vars or not self.finetuning_static_vars:
            raise ValueError("Missing forcings or attributes in configuration")
        
        # Create variable mappings
        self.ts_mapping, self.ts_missing_mask = create_variable_mapping(
            self.pretrained_ts_vars, self.finetuning_ts_vars
        )
        self.static_mapping, self.static_missing_mask = create_variable_mapping(
            self.pretrained_static_vars, self.finetuning_static_vars
        )
        
        # Model configuration - use pretrained variable counts for model architecture
        self.model_config = {
            'd_model': nn_config.get('hidden_size', 256),
            'num_heads': nn_config.get('num_heads', 4),
            'dropout': nn_config.get('dropout', 0.1),
            'num_enc_layers': nn_config.get('num_enc_layers', 4),
            'num_dec_layers': nn_config.get('num_dec_layers', 2),
            'd_ffd': nn_config.get('d_ffd', 512),
            'pred_len': dpl_config.get('rho', 365),
            'pretrained_model': nn_config.get('pretrained_model'),
            'target_variables': ny,
            'pretrained_type': nn_config.get('pretrained_type', 'mfformer'),
            'freeze_pretrained': nn_config.get('freeze_pretrained', True),
            'adapter_type': nn_config.get('adapter_type', 'none'),
            'use_residual_lstm': nn_config.get('use_residual_lstm', False),
            'adapter_params': nn_config.get('adapter_params', {}),
            'use_temporal_features': nn_config.get('use_temporal_features', True),
            'temporal_feature_dim': nn_config.get('temporal_feature_dim', 7),
        }
        
        # Extract test mode information
        self.test_mode = config_dict.get('test', {})
        self.is_spatial_test = (self.test_mode and 
                               self.test_mode.get('type') == 'spatial')
        
        # Initialize pretrained model with original variable configuration
        self.pretrained_model = self._initialize_pretrained_model()
        
        # Initialize adapter
        self.adapter = self._initialize_adapter()
        
        # Initialize LSTM decoder
        self.decoder = CudnnLstm(
            nx=self.model_config['d_model'],  
            hidden_size=self.model_config['d_model'],  
            dr=self.model_config['dropout']
        )
        
        # Initialize residual LSTM components if needed
        if self.model_config['use_residual_lstm']:
            n_time_features = len(self.finetuning_ts_vars)     
            n_static_features = len(self.finetuning_static_vars)
            
            self.lstm_input_dim = self.model_config['d_model'] + n_time_features + n_static_features
            self.pre_lstm = nn.Linear(self.lstm_input_dim, self.model_config['d_model'])
            self.post_lstm = nn.Linear(
                self.model_config['d_model'] + n_time_features + n_static_features,
                self.model_config['d_model']
            )
        
        # Direct input projection for non-pretrained case
        n_time_series = len(self.finetuning_ts_vars)  # Use fine-tuning counts for direct case
        n_static_features = len(self.finetuning_static_vars)
        self.direct_input_projection = nn.Linear(
            n_time_series + n_static_features, 
            self.model_config['d_model']
        )
        
        # Final projection
        self.projection = nn.Linear(self.model_config['d_model'], self.model_config['target_variables'])

        # Normalization and scaling for embeddings
        if self.model_config['pretrained_type'] in ['mfformer', 'mfformer_tft']:
            self.embedding_norm = nn.LayerNorm(self.model_config['d_model'])
            self.embedding_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        log.info(f"Initialized MaskedFineTuning: {self.model_config['pretrained_type']} + {self.model_config['adapter_type']}")
        log.info(f"Time series variables: {len(self.ts_mapping)}/{len(self.pretrained_ts_vars)} available")
        log.info(f"Static variables: {len(self.static_mapping)}/{len(self.pretrained_static_vars)} available")

    def _initialize_pretrained_model(self):
        """Initialize pretrained model with original variable configuration."""
        pretrained_type = self.model_config['pretrained_type']
        
        if pretrained_type == 'mfformer':
            mfformer_config = type('Config', (), {
                'd_model': self.model_config['d_model'],
                'num_heads': self.model_config['num_heads'],
                'dropout': self.model_config['dropout'],
                'num_enc_layers': self.model_config['num_enc_layers'],
                'num_dec_layers': self.model_config['num_dec_layers'],
                'd_ffd': self.model_config['d_ffd'],
                'time_series_variables': self.pretrained_ts_vars,  # Use pretrained variables!
                'static_variables': self.pretrained_static_vars,   # Use pretrained variables!
                'static_variables_category': [],
                'static_variables_category_dict': {},
                'mask_ratio_time_series': 0.5,
                'mask_ratio_static': 0.5,
                'min_window_size': 12,
                'max_window_size': 36,
                'init_weight': 0.02,
                'init_bias': 0.02,
                'warmup_train': False,
                'add_input_noise': False
            })
            
            built_model = MFFormer(mfformer_config).float()
            return self._load_pretrained_weights(built_model)
            
        elif pretrained_type == 'mfformer_tft':
            mfformer_tft_config = type('Config', (), {
                'd_model': self.model_config['d_model'],
                'num_heads': self.model_config['num_heads'],
                'dropout': self.model_config['dropout'],
                'num_enc_layers': self.model_config['num_enc_layers'],
                'num_dec_layers': self.model_config['num_dec_layers'],
                'd_ffd': self.model_config['d_ffd'],
                'time_series_variables': self.pretrained_ts_vars,  # Use pretrained variables!
                'static_variables': self.pretrained_static_vars,   # Use pretrained variables!
                'static_variables_category': [],
                'static_variables_category_dict': {},
                'mask_ratio_time_series': 0.5,
                'mask_ratio_static': 0.5,
                'min_window_size': 12,
                'max_window_size': 36,
                'init_weight': 0.02,
                'init_bias': 0.02,
                'warmup_train': False,
                'add_input_noise': False
            })
            
            built_model = MFFormerTFT(mfformer_tft_config).float()
            return self._load_pretrained_weights(built_model)
            
        elif pretrained_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported pretrained_type: {pretrained_type}")
    
    def _initialize_adapter(self):
        """Initialize adapter layer."""
        adapter_type = self.model_config['adapter_type']
        d_model = self.model_config['d_model']
        n_time_features = len(self.finetuning_ts_vars)    
        n_static_features = len(self.finetuning_static_vars)
        adapter_params = self.model_config['adapter_params']
        
        if adapter_type == 'dual_residual':
            from models.neural_networks.adapters.dual_residual_adapter import DualResidualAdapter
            return DualResidualAdapter(d_model, n_time_features, n_static_features)
            
        elif adapter_type == 'gated':
            from models.neural_networks.adapters.gated_adapter import GatedAdapter
            return GatedAdapter(d_model, n_time_features)
            
        elif adapter_type == 'feedforward':
            from models.neural_networks.adapters.feedforward_adapter import FeedforwardAdapter
            hidden_multiplier = adapter_params.get('hidden_multiplier', 2)
            return FeedforwardAdapter(d_model, n_time_features, hidden_multiplier)
            
        elif adapter_type == 'conv':
            from models.neural_networks.adapters.conv_adapter import ConvAdapter
            kernel_size = adapter_params.get('kernel_size', 3)
            return ConvAdapter(d_model, n_time_features, kernel_size)
            
        elif adapter_type == 'attention':
            from models.neural_networks.adapters.attention_adapter import AttentionAdapter
            num_heads = adapter_params.get('num_heads', 4)
            return AttentionAdapter(d_model, n_time_features, num_heads)
            
        elif adapter_type == 'bottleneck':
            from models.neural_networks.adapters.bottleneck_adapter import BottleneckAdapter
            bottleneck_size = adapter_params.get('bottleneck_size', 64)
            return BottleneckAdapter(d_model, n_time_features, bottleneck_size)
            
        elif adapter_type == 'moe':
            from models.neural_networks.adapters.moe_adapter import MoEAdapter
            num_experts = adapter_params.get('num_experts', 4)
            expert_size = adapter_params.get('expert_size', d_model)
            return MoEAdapter(d_model, n_time_features, num_experts, expert_size)
            
        elif adapter_type == 'none':
            return nn.Identity()
            
        else:
            raise ValueError(f"Invalid adapter type: {adapter_type}")
    
    def _load_pretrained_weights(self, model):
        """Load pretrained weights - now we can load ALL weights since we keep all layers."""
        pretrained_model_path = self.model_config['pretrained_model']
        if not pretrained_model_path:
            log.warning("No pretrained model path provided")
            return model
            
        try:
            if os.path.isdir(pretrained_model_path):
                checkpoint_list = [f for f in os.listdir(pretrained_model_path) if re.search(r'^.+_[\d]*.pt$', f)]
                if not checkpoint_list:
                    raise ValueError(f"No checkpoint files found in {pretrained_model_path}")
                checkpoint_list.sort()
                checkpoint_file = os.path.join(pretrained_model_path, checkpoint_list[-1])
            elif os.path.isfile(pretrained_model_path):
                checkpoint_file = pretrained_model_path
            else:
                raise ValueError('pretrained_model is not a valid file or directory')

            import numpy as np
            from torch.serialization import safe_globals, add_safe_globals
            
            add_safe_globals([np.core.multiarray.scalar])
            
            with safe_globals([np.core.multiarray.scalar]):
                checkpoint_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

            pretrained_state_dict = checkpoint_dict['model_state_dict']
            
            # Clean state dict
            cleaned_state_dict = {}
            for name, param in pretrained_state_dict.items():
                clean_name = name[7:] if name.startswith('module.') else name
                cleaned_state_dict[clean_name] = param
            
            # Load weights - now most should match!
            current_state_dict = model.state_dict()
            loaded_keys = []
            skipped_keys = []
            
            for name, param in cleaned_state_dict.items():
                if name in current_state_dict and current_state_dict[name].size() == param.size():
                    current_state_dict[name].copy_(param)
                    loaded_keys.append(name)
                else:
                    skipped_keys.append(name)
                    log.debug(f"Skipping parameter {name}: size mismatch or not found")

            model.load_state_dict(current_state_dict)

            log.info(f"Loaded {len(loaded_keys)} parameters from pretrained model")
            if skipped_keys:
                log.info(f"Skipped {len(skipped_keys)} parameters")

            # Freeze pretrained weights if specified
            if self.model_config['freeze_pretrained']:
                for param in model.parameters():
                    param.requires_grad = False
                    
            return model
            
        except Exception as e:
            log.error(f"Error loading pretrained model: {str(e)}")
            raise

    def forward(self, xc_nn_norm, temporal_features=None, station_ids=None) -> torch.Tensor:
        """
        Forward pass with masked variable handling.
        
        Parameters
        ----------
        xc_nn_norm : torch.Tensor
            Normalized input features [time, stations, features]
        temporal_features : torch.Tensor, optional
            Temporal features for TFT models
        station_ids : torch.Tensor, optional
            Station IDs for localized adaptation [batch_size]
        """
        if self.model_config['pretrained_type'] in ['mfformer', 'mfformer_tft']:
            return self._forward_with_mfformer(xc_nn_norm, temporal_features, station_ids)
        elif self.model_config['pretrained_type'] == 'none':
            return self._forward_direct(xc_nn_norm, station_ids)
    
    def _forward_with_mfformer(self, xc_nn_norm, temporal_features=None, station_ids=None):
        # Split fine-tuning input into time series and static components
        n_ts_vars = len(self.finetuning_ts_vars)
        n_static_vars = len(self.finetuning_static_vars)
        
        # Extract ALL fine-tuning variables (this is what we want to use for adaptation)
        batch_x_ft = xc_nn_norm[..., :n_ts_vars]  # ALL 5 time series variables
        batch_c_ft = xc_nn_norm[..., n_ts_vars:n_ts_vars+n_static_vars][..., 0, :]  # ALL 26 static variables
        
        # Convert to pretrained model format with masking (only for pretrained model)
        batch_x_pretrained = prepare_masked_input(
            batch_x_ft, self.ts_mapping, self.ts_missing_mask, len(self.pretrained_ts_vars)
        )
        batch_c_pretrained = prepare_masked_input(
            batch_c_ft, self.static_mapping, self.static_missing_mask, len(self.pretrained_static_vars)
        )
        
        
        # Create masks for the pretrained model (True = missing/masked)
        time_series_mask = torch.tensor(
            self.ts_missing_mask, 
            device=batch_x_pretrained.device
        ).unsqueeze(0).unsqueeze(0).expand_as(batch_x_pretrained)
        
        static_mask = torch.tensor(
            self.static_missing_mask,
            device=batch_c_pretrained.device  
        ).unsqueeze(0).expand_as(batch_c_pretrained)
        
        # Run through pretrained model with masks
        with torch.amp.autocast('cuda', enabled=False):
            batch_data_dict = {
                'batch_x': batch_x_pretrained,
                'batch_c': batch_c_pretrained,
                'batch_time_series_mask_index': time_series_mask,
                'batch_static_mask_index': static_mask,
                'mode': 'test'
            }
            
            if temporal_features is not None:
                batch_data_dict['temporal_features'] = temporal_features
            
            # Get embeddings from pretrained model (using masked data)
            if self.model_config['pretrained_type'] == 'mfformer_tft':
                enc_x = self.pretrained_model.time_series_embedding(
                    batch_x_pretrained, 
                    feature_order=self.pretrained_ts_vars,
                    masked_index=time_series_mask
                )
                enc_c = self.pretrained_model.static_embedding(
                    batch_c_pretrained, 
                    feature_order=self.pretrained_static_vars,
                    masked_index=static_mask
                )
                
                # Process through model
                enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
                enc_combined = self.pretrained_model.positional_encoding(
                    enc_combined, temporal_features=temporal_features
                )
                hidden_states = self.pretrained_model.encoder(enc_combined)
                hidden_states = self.pretrained_model.encoder_norm(hidden_states)
                hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
                hidden_states = hidden_states[:, :batch_x_pretrained.size(1), :]
                
            else:  # mfformer
                enc_x = self.pretrained_model.time_series_embedding(
                    batch_x_pretrained, 
                    feature_order=self.pretrained_ts_vars,
                    masked_index=time_series_mask
                )
                enc_c = self.pretrained_model.static_embedding(
                    batch_c_pretrained, 
                    feature_order=self.pretrained_static_vars,
                    masked_index=static_mask
                )
                
                enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
                enc_combined = self.pretrained_model.positional_encoding(enc_combined)
                hidden_states = self.pretrained_model.encoder(enc_combined)
                hidden_states = self.pretrained_model.encoder_norm(hidden_states)
                hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
                hidden_states = hidden_states[:, :batch_x_pretrained.size(1), :]
            
            
            # Normalize and scale embeddings
            hidden_states = self.embedding_norm(hidden_states)
            hidden_states = hidden_states * self.embedding_scale
            
            # Apply adapter using FULL fine-tuning data (not masked data!)
            adapted = self._apply_adapter(hidden_states, batch_x_ft, batch_c_ft)
            
            if self.model_config['use_residual_lstm']:
                outputs = self._decode_with_residual_lstm(adapted, batch_x_ft, batch_c_ft)
            else:
                outputs = self._decode_simple(adapted)
            
            # Final projection
            outputs = self.projection(outputs)
            
            return outputs
    
    def _apply_adapter(self, hidden_states, orig_time_features, orig_static_features):
        adapter_type = self.model_config['adapter_type']            
        if adapter_type in ['gated', 'feedforward', 'conv', 'attention', 'bottleneck', 'moe', 'dual_residual']:
            return self.adapter(hidden_states, orig_time_features, orig_static_features)
        elif adapter_type == 'none':
            return self.adapter(hidden_states)
        else:
            raise ValueError(f"Invalid adapter type: {adapter_type}")
    
    def _decode_with_residual_lstm(self, adapted, orig_time_features, orig_static_features):
        # Prepare LSTM input with residual connections
        static_expanded = orig_static_features.unsqueeze(1).expand(-1, adapted.size(1), -1)
        
        # Concatenate adapted features with original inputs for rich representation
        lstm_input_combined = torch.cat([adapted, orig_time_features, static_expanded], dim=-1)
        
        # Project to LSTM dimension
        lstm_input = self.pre_lstm(lstm_input_combined)
        
        # Process through LSTM
        lstm_output, _ = self.decoder(
            lstm_input,
            do_drop_mc=False,
            dr_false=(not self.training)
        )
        
        # Reintroduce time series and static information
        post_lstm_combined = torch.cat([lstm_output, orig_time_features, static_expanded], dim=-1)
        
        # Process through post-LSTM layer with residual connection
        post_output = self.post_lstm(post_lstm_combined)
        final_output = post_output + lstm_output  # Residual connection
        
        return final_output
    
    def _decode_simple(self, adapted):
        lstm_input = adapted
        lstm_output, _ = self.decoder(lstm_input, do_drop_mc=False, dr_false=(not self.training))
        return lstm_output
    
    def _forward_direct(self, xc_nn_norm, station_ids=None):
        # LSTM-only case using fine-tuning variables directly
        n_time_series = len(self.finetuning_ts_vars)
        batch_x = xc_nn_norm[..., :n_time_series]
        batch_c = xc_nn_norm[..., n_time_series:n_time_series+len(self.finetuning_static_vars)][..., 0, :]
        
        static_expanded = batch_c.unsqueeze(1).expand(-1, batch_x.size(1), -1)
        combined_input = torch.cat([batch_x, static_expanded], dim=-1)
        
        lstm_input = self.direct_input_projection(combined_input)
        dec_output = self._decode_simple(lstm_input)
        outputs = self.projection(dec_output)
        return outputs