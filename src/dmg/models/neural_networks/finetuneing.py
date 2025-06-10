import os
import re
import torch
import torch.nn as nn
import logging
import warnings
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union

from models.neural_networks.lstm_models import CudnnLstm
from models.neural_networks.transformer.MFFormer import Model as MFFormer

warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")
log = logging.getLogger(__name__)

class FineTuner(nn.Module):
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
            dpl_config = config_dict.get('dpl_model', {})
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
            'pretrained_model': dpl_config.get('pretrained_model'),
            'target_variables': ny,
            'pretrained_type': nn_config.get('pretrained_type', 'mfformer'),
            'freeze_pretrained': nn_config.get('freeze_pretrained', True),
            'adapter_type': nn_config.get('adapter_type', 'none'),
            'use_residual_lstm': nn_config.get('use_residual_lstm', False)
        }
        
        missing_configs = []
        if not self.model_config['time_series_variables']:
            missing_configs.append("time_series_variables (forcings)")
        if not self.model_config['target_variables']:
            missing_configs.append("target_variables")
        if missing_configs:
            raise ValueError(f"Missing configuration: {', '.join(missing_configs)}")

        # Initialize pretrained model
        self.pretrained_model = self._initialize_pretrained_model()
        
        # Initialize adapter
        self.adapter = self._initialize_adapter()
        
        # Initialize LSTM components
        self.decoder = CudnnLstm(
            inputSize=self.model_config['d_model'],
            hiddenSize=self.model_config['d_model'],
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
        
        log.info(f"Initialized FineTuner: {self.model_config['pretrained_type']} + {self.model_config['adapter_type']}")

    def _initialize_pretrained_model(self):
        pretrained_type = self.model_config['pretrained_type']
        
        if pretrained_type == 'mfformer':
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
        elif pretrained_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported pretrained_type: {pretrained_type}")
    
    def _initialize_adapter(self):
        adapter_type = self.model_config['adapter_type']
        
        if adapter_type == 'dual_residual':
            from models.neural_networks.adapters import DualResidualAdapter
            d_model = self.model_config['d_model']
            n_time_features = len(self.model_config['time_series_variables'])
            n_static_features = len(self.model_config['static_variables'])
            return DualResidualAdapter(d_model, n_time_features, n_static_features)
        elif adapter_type == 'feedforward':
            d_model = self.model_config['d_model']
            n_features = len(self.model_config['time_series_variables'])
            return nn.Sequential(
                nn.Linear(d_model + n_features, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
        elif adapter_type == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
    
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
            current_state_dict = model.state_dict()
            
            for name, param in pretrained_state_dict.items():
                if name in current_state_dict and current_state_dict[name].size() == param.size():
                    current_state_dict[name].copy_(param)

            if list(current_state_dict.keys())[0].startswith('module.'):
                new_state_dict = {k.replace('module.', ''): v for k, v in current_state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(current_state_dict)

            if self.model_config['freeze_pretrained']:
                for param in model.parameters():
                    param.requires_grad = False
                    
            return model
            
        except Exception as e:
            log.error(f"Error loading pretrained model: {str(e)}")
            raise

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model_config['pretrained_type'] == 'mfformer':
            return self._forward_with_mfformer(data_dict)
        elif self.model_config['pretrained_type'] == 'none':
            return self._forward_direct(data_dict)
    
    def _forward_with_mfformer(self, data_dict):
        xc_nn_norm = data_dict['xc_nn_norm']
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
            # Get embeddings from pretrained model
            enc_x = self.pretrained_model.time_series_embedding(
                batch_x, feature_order=self.model_config['time_series_variables']
            )
            enc_c = self.pretrained_model.static_embedding(
                batch_c, feature_order=self.model_config['static_variables']
            )
            
            # Save original embeddings for residual connections (YOUR EXACT PATTERN)
            orig_time_features = batch_x
            orig_static_features = batch_c
            
            # Process through transformer encoder
            enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
            enc_combined = self.pretrained_model.positional_encoding(enc_combined)
            hidden_states = self.pretrained_model.encoder(enc_combined)
            hidden_states = self.pretrained_model.encoder_norm(hidden_states)
            hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
            hidden_states = hidden_states[:, :batch_x.size(1), :]
            
            # Apply adapter (preserving your exact call signature)
            if self.model_config['adapter_type'] == 'dual_residual':
                adapted = self.adapter(hidden_states, orig_time_features, orig_static_features)
            elif self.model_config['adapter_type'] == 'feedforward':
                adapter_input = torch.cat([hidden_states, orig_time_features], dim=-1)
                adapted = self.adapter(adapter_input)
            else:
                adapted = self.adapter(hidden_states)
            
            # Apply LSTM with your exact residual pattern
            if self.model_config['use_residual_lstm']:
                outputs = self._decode_with_residual_lstm(adapted, orig_time_features, orig_static_features)
            else:
                outputs = self._decode_simple(adapted)
            
            # Final projection
            outputs = self.projection(outputs)
            
            if len(outputs.shape) == 3:
                outputs = outputs.permute(1, 0, 2)
                
            return outputs
    
    def _decode_with_residual_lstm(self, adapted, orig_time_features, orig_static_features):
        # YOUR EXACT RESIDUAL LSTM PATTERN
        # Prepare LSTM input with residual connections
        static_expanded = orig_static_features.unsqueeze(1).expand(-1, adapted.size(1), -1)
        
        # Concatenate adapted features with original inputs for rich representation
        lstm_input_combined = torch.cat([adapted, orig_time_features, static_expanded], dim=-1)
        
        # Project to LSTM dimension
        lstm_input = self.pre_lstm(lstm_input_combined)
        
        # Permute for LSTM [seq, batch, features]
        lstm_input = lstm_input.permute(1, 0, 2)
        
        # Process through LSTM
        lstm_output, _ = self.decoder(
            lstm_input,
            doDropMC=False,
            dropoutFalse=(not self.training)
        )
        
        # Convert back to [batch, seq, features]
        lstm_output = lstm_output.permute(1, 0, 2)
        
        # Add final residual connections
        # Reintroduce time series and static information
        post_lstm_combined = torch.cat([lstm_output, orig_time_features, static_expanded], dim=-1)
        
        # Process through post-LSTM layer with residual connection
        post_output = self.post_lstm(post_lstm_combined)
        final_output = post_output + lstm_output  # YOUR EXACT RESIDUAL CONNECTION
        
        return final_output
    
    def _decode_simple(self, adapted):
        lstm_input = adapted.permute(1, 0, 2)
        lstm_output, _ = self.decoder(lstm_input, doDropMC=False, dropoutFalse=(not self.training))
        return lstm_output.permute(1, 0, 2)
    
    def _forward_direct(self, data_dict):
        # LSTM-only case
        xc_nn_norm = data_dict['xc_nn_norm']
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
        dec_output = self._decode_simple(lstm_input)
        outputs = self.projection(dec_output)
        return outputs.permute(1, 0, 2)