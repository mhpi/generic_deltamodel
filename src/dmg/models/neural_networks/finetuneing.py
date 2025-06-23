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
from models.neural_networks.adapters.localization_adapter import LocalizedStationAdapter, AdaptiveLocalizedAdapter

warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")
log = logging.getLogger(__name__)

class Finetuneing(nn.Module):
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
        
        self.model_config = {
            'd_model': nn_config.get('hidden_size', 256),
            'num_heads': 4,
            'dropout': nn_config.get('dropout', 0.1),
            'num_enc_layers': nn_config.get('num_enc_layers', 4),
            'num_dec_layers': nn_config.get('num_dec_layers', 2),
            'd_ffd': nn_config.get('d_ffd', 512),
            'time_series_variables': nn_config.get('forcings', []),
            'static_variables': nn_config.get('attributes', []),
            'pred_len': dpl_config.get('rho', 365),
            'pretrained_model': nn_config.get('pretrained_model'),
            'target_variables': ny,
            'pretrained_type': nn_config.get('pretrained_type', 'mfformer'),
            'freeze_pretrained': nn_config.get('freeze_pretrained', True),
            'adapter_type': nn_config.get('adapter_type', 'none'),
            'use_residual_lstm': nn_config.get('use_residual_lstm', False),
            'adapter_params': nn_config.get('adapter_params', {}),
            # New TFT-specific configurations
            'use_temporal_features': nn_config.get('use_temporal_features', True),
            'temporal_feature_dim': nn_config.get('temporal_feature_dim', 7),
            # Localized adapter configurations
            'use_localized_adapter': nn_config.get('use_localized_adapter', False),
            'localized_adapter_params': nn_config.get('localized_adapter_params', {}),
        }
        
        # Extract test mode information for spatial testing detection
        self.test_mode = config_dict.get('test_mode', {})
        self.is_spatial_test = (self.test_mode and 
                               self.test_mode.get('type') == 'spatial')
        
        missing_configs = []
        if not self.model_config['time_series_variables']:
            missing_configs.append("time_series_variables (forcings)")
        if not self.model_config['target_variables']:
            missing_configs.append("target_variables")
        if missing_configs:
            raise ValueError(f"Missing configuration: {', '.join(missing_configs)}")

        # Initialize pretrained model
        self.pretrained_model = self._initialize_pretrained_model()
        
        # Initialize standard adapter
        self.adapter = self._initialize_adapter()
        
        # Initialize localized station adapter
        self.localized_adapter = self._initialize_localized_adapter(config_dict)
        
        # Initialize LSTM components
        self.decoder = CudnnLstm(
            nx=self.model_config['d_model'],  
            hidden_size=self.model_config['d_model'],  
            dr=self.model_config['dropout']
        )
        
        # Initialize residual LSTM components if needed
        if self.model_config['use_residual_lstm']:
            n_time_features = len(self.model_config['time_series_variables'])
            n_static_features = len(self.model_config['static_variables'])
            
            self.lstm_input_dim = self.model_config['d_model'] + n_time_features + n_static_features
            self.pre_lstm = nn.Linear(self.lstm_input_dim, self.model_config['d_model'])
            self.post_lstm = nn.Linear(
                self.model_config['d_model'] + n_time_features + n_static_features,
                self.model_config['d_model']
            )
        
        # Final projection
        self.projection = nn.Linear(self.model_config['d_model'], self.model_config['target_variables'])
        
        log.info(f"Initialized FineTuner: {self.model_config['pretrained_type']} + {self.model_config['adapter_type']}" + 
                (f" + localized_adapter" if self.model_config['use_localized_adapter'] else ""))

    def _initialize_localized_adapter(self, config_dict: Dict) -> Optional[nn.Module]:
        """Initialize the localized station adapter if enabled."""
        if not self.model_config['use_localized_adapter']:
            return None
        
        # Get station information from config
        localized_params = self.model_config['localized_adapter_params']
        
        # Determine number of stations from dataset configuration
        n_stations = self._get_number_of_stations(config_dict)
        
        if n_stations is None:
            log.warning("Cannot determine number of stations, disabling localized adapter")
            return None
        
        # Extract parameters with defaults
        embedding_dim = localized_params.get('embedding_dim', 64)
        dropout_rate = localized_params.get('dropout_rate', 0.2)
        use_positional = localized_params.get('use_positional', True)
        adapter_variant = localized_params.get('variant', 'standard')  # 'standard' or 'adaptive'
        
        # Initialize appropriate adapter variant
        if adapter_variant == 'adaptive':
            use_adaptive_weighting = localized_params.get('use_adaptive_weighting', True)
            context_window = localized_params.get('context_window', 7)
            
            localized_adapter = AdaptiveLocalizedAdapter(
                d_model=self.model_config['d_model'],
                n_stations=n_stations,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                use_positional=use_positional,
                use_adaptive_weighting=use_adaptive_weighting,
                context_window=context_window
            )
        else:
            localized_adapter = LocalizedStationAdapter(
                d_model=self.model_config['d_model'],
                n_stations=n_stations,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                use_positional=use_positional
            )
        
        log.info(f"Initialized {adapter_variant} localized adapter for {n_stations} stations")
        return localized_adapter
    
    def _get_number_of_stations(self, config_dict: Dict) -> Optional[int]:
        """Determine the number of stations from configuration."""
        try:
            # Try to get from subset file if available
            observations = config_dict.get('observations', {})
            subset_path = observations.get('subset_path')
            
            if subset_path and os.path.exists(subset_path):
                try:
                    with open(subset_path, 'r') as f:
                        content = f.read().strip()
                        if content.startswith('[') and content.endswith(']'):
                            content = content.strip('[]')
                            stations = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                        else:
                            stations = [line.strip() for line in content.split() if line.strip()]
                    return len(stations)
                except Exception as e:
                    log.warning(f"Error reading subset file: {e}")
            
            # Fallback: try to get from localized adapter params if explicitly specified
            localized_params = self.model_config['localized_adapter_params']
            if 'n_stations' in localized_params:
                return localized_params['n_stations']
            
            # If all else fails, return None (will disable localized adapter)
            log.warning("Could not determine number of stations from config")
            return None
            
        except Exception as e:
            log.warning(f"Error determining number of stations: {e}")
            return None

    def _initialize_pretrained_model(self):
        pretrained_type = self.model_config['pretrained_type']
        
        if pretrained_type == 'mfformer':
            # Original MFFormer configuration
            mfformer_config = type('Config', (), {
                'd_model': self.model_config['d_model'],
                'num_heads': self.model_config['num_heads'],
                'dropout': self.model_config['dropout'],
                'num_enc_layers': self.model_config['num_enc_layers'],
                'num_dec_layers': self.model_config['num_dec_layers'],
                'd_ffd': self.model_config['d_ffd'],
                'time_series_variables': self.model_config['time_series_variables'],
                'static_variables': self.model_config['static_variables'],
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
            # New MFFormer with TFT-style temporal encoding
            mfformer_tft_config = type('Config', (), {
                'd_model': self.model_config['d_model'],
                'num_heads': self.model_config['num_heads'],
                'dropout': self.model_config['dropout'],
                'num_enc_layers': self.model_config['num_enc_layers'],
                'num_dec_layers': self.model_config['num_dec_layers'],
                'd_ffd': self.model_config['d_ffd'],
                'time_series_variables': self.model_config['time_series_variables'],
                'static_variables': self.model_config['static_variables'],
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
            raise ValueError(f"Unsupported pretrained_type: {pretrained_type}. Available types: ['mfformer', 'mfformer_tft', 'none']")
    
    def _initialize_adapter(self):
        adapter_type = self.model_config['adapter_type']
        d_model = self.model_config['d_model']
        n_time_features = len(self.model_config['time_series_variables'])
        n_static_features = len(self.model_config['static_variables'])
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
            raise ValueError(f"Invalid adapter type: {adapter_type} available types: ['none', 'gated', 'feedforward', 'conv', 'attention', 'bottleneck', 'moe','dual_residual']")
    
    def _load_pretrained_weights(self, model):
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
            
            cleaned_state_dict = {}
            for name, param in pretrained_state_dict.items():
                # Strip 'module.' prefix if it exists
                clean_name = name[7:] if name.startswith('module.') else name
                cleaned_state_dict[clean_name] = param
            
            # Now use the cleaned state dict
            current_state_dict = model.state_dict()
            
            loaded_keys = []
            for name, param in cleaned_state_dict.items():
                if name in current_state_dict and current_state_dict[name].size() == param.size():
                    current_state_dict[name].copy_(param)
                    loaded_keys.append(name)
                else:
                    log.warning(f"Skipping parameter {name}: size mismatch or not found in current model")

            # Load the updated state dict
            model.load_state_dict(current_state_dict)

            log.info(f"Loaded {len(loaded_keys)} parameters from pretrained model")

            if self.model_config['freeze_pretrained']:
                for param in model.parameters():
                    param.requires_grad = False
                    
            return model
            
        except Exception as e:
            log.error(f"Error loading pretrained model: {str(e)}")
            raise

    def forward(self, xc_nn_norm, temporal_features=None, station_ids=None) -> torch.Tensor:
        """
        Forward pass with optional localized station adaptation.
        
        Parameters
        ----------
        xc_nn_norm : torch.Tensor
            Normalized input features
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
        n_time_series = len(self.model_config['time_series_variables'])
        batch_x = xc_nn_norm[..., :n_time_series]
        batch_c = xc_nn_norm[..., n_time_series:][..., 0, :]
        
        # Handle missing values
        x_mask = torch.isnan(batch_x)
        c_mask = torch.isnan(batch_c)
        x_median = torch.nanmedian(batch_x) if not torch.isnan(batch_x).all() else torch.tensor(0.0)
        c_median = torch.nanmedian(batch_c) if not torch.isnan(batch_c).all() else torch.tensor(0.0)
        batch_x = batch_x.masked_fill(x_mask, x_median)
        batch_c = batch_c.masked_fill(c_mask, c_median)
        
        with torch.amp.autocast('cuda', enabled=False):
            # Prepare batch data dictionary for both model types
            batch_data_dict = {
                'batch_x': batch_x,
                'batch_c': batch_c,
                'batch_time_series_mask_index': torch.zeros_like(batch_x, dtype=torch.bool),
                'batch_static_mask_index': torch.zeros_like(batch_c, dtype=torch.bool),
                'mode': 'test'
            }
            
            # Add temporal features for TFT model if available
            if self.model_config['pretrained_type'] == 'mfformer_tft' and temporal_features is not None:
                batch_data_dict['temporal_features'] = temporal_features
            
            # Get embeddings from pretrained model
            if self.model_config['pretrained_type'] == 'mfformer_tft':
                # Use the new TFT model's forward method
                model_output = self.pretrained_model(batch_data_dict, is_mask=False)
                
                # Extract encoded representations from the model
                # We need to manually extract the embeddings since the TFT model has a different structure
                enc_x = self.pretrained_model.time_series_embedding(
                    batch_x, feature_order=self.model_config['time_series_variables']
                )
                enc_c = self.pretrained_model.static_embedding(
                    batch_c, feature_order=self.model_config['static_variables']
                )
                
                # Process through TFT positional encoding
                enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
                temporal_features_for_encoding = batch_data_dict.get('temporal_features', None)
                enc_combined = self.pretrained_model.positional_encoding(enc_combined, temporal_features=temporal_features_for_encoding)
                
                # Process through transformer encoder
                hidden_states = self.pretrained_model.encoder(enc_combined)
                hidden_states = self.pretrained_model.encoder_norm(hidden_states)
                hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
                hidden_states = hidden_states[:, :batch_x.size(1), :]
                
            else:
                # Original MFFormer processing
                enc_x = self.pretrained_model.time_series_embedding(
                    batch_x, feature_order=self.model_config['time_series_variables']
                )
                enc_c = self.pretrained_model.static_embedding(
                    batch_c, feature_order=self.model_config['static_variables']
                )
                
                # Process through transformer encoder
                enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
                enc_combined = self.pretrained_model.positional_encoding(enc_combined)
                hidden_states = self.pretrained_model.encoder(enc_combined)
                hidden_states = self.pretrained_model.encoder_norm(hidden_states)
                hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
                hidden_states = hidden_states[:, :batch_x.size(1), :]
            
            orig_time_features = batch_x
            orig_static_features = batch_c
            
            # Apply standard adapter first
            adapted = self._apply_adapter(hidden_states, orig_time_features, orig_static_features)
            
            # Apply localized station adapter if enabled
            if self.localized_adapter is not None:
                adapted = self.localized_adapter(
                    adapted, 
                    station_ids=station_ids,
                    is_spatial_test=self.is_spatial_test,
                    is_training=self.training
                )
            
            if self.model_config['use_residual_lstm']:
                outputs = self._decode_with_residual_lstm(adapted, orig_time_features, orig_static_features)
            else:
                outputs = self._decode_simple(adapted)
            
            # Final projection
            outputs = self.projection(outputs)
            
            return outputs
    
    def _apply_adapter(self, hidden_states, orig_time_features, orig_static_features):
        adapter_type = self.model_config['adapter_type']            
        if adapter_type in ['gated', 'feedforward', 'conv', 'attention', 'bottleneck', 'moe','dual_residual']:
            return self.adapter(hidden_states, orig_time_features, orig_static_features)
            
        elif adapter_type == 'none':
            return self.adapter(hidden_states)
            
        else:
            raise ValueError(f"Invalid adapter type: {adapter_type} available types:['none', 'gated', 'feedforward', 'conv', 'attention', 'bottleneck', 'moe','dual_residual']")
    
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
        # LSTM-only case
        n_time_series = len(self.model_config['time_series_variables'])
        batch_x = xc_nn_norm[..., :n_time_series]
        batch_c = xc_nn_norm[..., n_time_series:][..., 0, :]
        
        # Handle NaN
        x_mask = torch.isnan(batch_x)
        c_mask = torch.isnan(batch_c)
        x_median = torch.nanmedian(batch_x) if not torch.isnan(batch_x).all() else torch.tensor(0.0)
        c_median = torch.nanmedian(batch_c) if not torch.isnan(batch_c).all() else torch.tensor(0.0)
        batch_x = batch_x.masked_fill(x_mask, x_median)
        batch_c = batch_c.masked_fill(c_mask, c_median)
        
        static_expanded = batch_c.unsqueeze(1).expand(-1, batch_x.size(1), -1)
        combined_input = torch.cat([batch_x, static_expanded], dim=-1)
        
        lstm_input = nn.Linear(combined_input.size(-1), self.model_config['d_model']).to(combined_input.device)(combined_input)
        
        # Apply localized station adapter if enabled (even for direct LSTM)
        if self.localized_adapter is not None:
            lstm_input = self.localized_adapter(
                lstm_input,
                station_ids=station_ids,
                is_spatial_test=self.is_spatial_test,
                is_training=self.training
            )
        
        dec_output = self._decode_simple(lstm_input)
        outputs = self.projection(dec_output)
        return outputs
    
    def get_station_similarities(self, station_ids: torch.Tensor = None) -> Optional[torch.Tensor]:
        """
        Get station similarity matrix from localized adapter.
        Useful for analyzing station groupings.
        """
        if self.localized_adapter is None:
            log.warning("No localized adapter available for similarity analysis")
            return None
        
        return self.localized_adapter.get_station_similarities(station_ids)
    
    def set_localized_dropout(self, rate: float) -> None:
        """Update the dropout rate for localized adapter."""
        if self.localized_adapter is not None:
            self.localized_adapter.set_dropout_rate(rate)
        else:
            log.warning("No localized adapter available to set dropout rate")
    
    def get_parameters(self):
        """Get trainable parameters, respecting frozen pretrained weights."""
        trainable_params = []
        
        # Add adapter parameters (always trainable)
        if hasattr(self, 'adapter') and self.adapter is not None:
            trainable_params.extend(list(self.adapter.parameters()))
        
        # Add localized adapter parameters (always trainable)
        if hasattr(self, 'localized_adapter') and self.localized_adapter is not None:
            trainable_params.extend(list(self.localized_adapter.parameters()))
        
        # Add LSTM decoder parameters (always trainable)
        if hasattr(self, 'decoder'):
            trainable_params.extend(list(self.decoder.parameters()))
        
        # Add residual LSTM parameters if present
        if hasattr(self, 'pre_lstm'):
            trainable_params.extend(list(self.pre_lstm.parameters()))
        if hasattr(self, 'post_lstm'):
            trainable_params.extend(list(self.post_lstm.parameters()))
        
        # Add final projection parameters
        if hasattr(self, 'projection'):
            trainable_params.extend(list(self.projection.parameters()))
        
        # Add pretrained model parameters only if not frozen
        if (hasattr(self, 'pretrained_model') and 
            self.pretrained_model is not None and 
            not self.model_config['freeze_pretrained']):
            trainable_params.extend(list(self.pretrained_model.parameters()))
        
        log.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")
        
        return trainable_params