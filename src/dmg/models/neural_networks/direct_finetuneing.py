import os
import re
import torch
import torch.nn as nn
import logging
import warnings
import numpy as np
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union

from models.neural_networks.cudnn_lstm import CudnnLstm
from models.neural_networks.transformer.StefaLand_PatchTokens_TFT import Model as StefaLandPatchTFT

warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")
log = logging.getLogger(__name__)


class DirectFinetuneing(nn.Module):
    """
    Fine-tuning module that uses pretrained data directly (xc_pretrained_norm)
    through a frozen pretrained encoder, then adapts with trainable adapter + LSTM.

    Expects NnDualLoader which provides both xc_nn_norm and xc_pretrained_norm.
    All data flows through dpl_model as a dict.
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

        # Pretrained variable lists (defines pretrained model architecture)
        self.pretrained_ts_vars = nn_config.get('pretrained_time_series_vars', [])
        self.pretrained_static_vars = nn_config.get('pretrained_static_vars', [])
        if not self.pretrained_ts_vars or not self.pretrained_static_vars:
            raise ValueError("Must specify 'pretrained_time_series_vars' and 'pretrained_static_vars'")

        # Fine-tuning variable lists (for adapter and LSTM)
        self.finetuning_ts_vars = nn_config.get('forcings', [])
        self.finetuning_static_vars = nn_config.get('attributes', [])
        if not self.finetuning_ts_vars or not self.finetuning_static_vars:
            raise ValueError("Must specify 'forcings' and 'attributes'")

        n_ft_ts = len(self.finetuning_ts_vars)
        n_ft_static = len(self.finetuning_static_vars)

        # Model config
        self.d_model = nn_config.get('hidden_size', 256)
        self.dropout = nn_config.get('dropout', 0.1)
        self.freeze_pretrained = nn_config.get('freeze_pretrained', True)
        self.use_residual_lstm = nn_config.get('use_residual_lstm', False)
        self.target_variables = ny

        self.model_config = {
            'd_model': self.d_model,
            'num_heads': nn_config.get('num_heads', 4),
            'dropout': self.dropout,
            'num_enc_layers': nn_config.get('num_enc_layers', 4),
            'd_ffd': nn_config.get('d_ffd', 512),
            'pretrained_model': nn_config.get('pretrained_model'),
            'adapter_type': nn_config.get('adapter_type', 'dual_residual'),
            'adapter_params': nn_config.get('adapter_params', {}),
        }

        # Initialize pretrained encoder
        self.pretrained_model = self._build_pretrained_model()

        # Initialize adapter
        self.adapter = self._build_adapter(
            self.model_config['adapter_type'],
            self.d_model, n_ft_ts, n_ft_static,
            self.model_config['adapter_params'],
        )

        # LSTM decoder
        self.decoder = CudnnLstm(
            nx=self.d_model,
            hidden_size=self.d_model,
            dr=self.dropout,
        )

        # Residual LSTM components
        if self.use_residual_lstm:
            self.pre_lstm = nn.Linear(self.d_model + n_ft_ts + n_ft_static, self.d_model)
            self.post_lstm = nn.Linear(self.d_model + n_ft_ts + n_ft_static, self.d_model)

        # Embedding normalization and scaling
        self.embedding_norm = nn.LayerNorm(self.d_model)
        self.embedding_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Final projection to target
        self.projection = nn.Linear(self.d_model, self.target_variables)

        log.info(f"Initialized DirectFinetuneing: adapter={self.model_config['adapter_type']}")
        log.info(f"Pretrained vars: {len(self.pretrained_ts_vars)} ts, {len(self.pretrained_static_vars)} static")
        log.info(f"Fine-tuning vars: {n_ft_ts} ts, {n_ft_static} static")

    # ------------------------------------------------------------------
    # Pretrained model
    # ------------------------------------------------------------------
    def _build_pretrained_model(self):
        cfg = type('Config', (), {
            'd_model': self.d_model,
            'num_heads': self.model_config['num_heads'],
            'dropout': self.dropout,
            'num_enc_layers': self.model_config['num_enc_layers'],
            'd_ffd': self.model_config['d_ffd'],
            'time_series_variables': self.pretrained_ts_vars,
            'static_variables': self.pretrained_static_vars,
            'static_variables_category': [],
            'static_variables_category_dict': {},
            'mask_ratio_time_series': 0.5,
            'mask_ratio_static': 0.5,
            'min_window_size': 12,
            'max_window_size': 36,
            'init_weight': 0.02,
            'init_bias': 0.02,
            'warmup_train': False,
            'add_input_noise': False,
            'use_patches': True,
            'patch_len': 16,
            'patch_stride': 8,
            'group_mask_dict': {},
        })
        model = StefaLandPatchTFT(cfg).float()
        return self._load_weights(model)

    def _load_weights(self, model):
        path = self.model_config['pretrained_model']
        if not path:
            log.warning("No pretrained model path; using random init")
            return model

        try:
            if os.path.isdir(path):
                files = [f for f in os.listdir(path) if re.search(r'^.+_[\d]*.pt$', f)]
                if not files:
                    raise ValueError(f"No checkpoint files in {path}")
                files.sort()
                path = os.path.join(path, files[-1])

            from torch.serialization import safe_globals, add_safe_globals
            add_safe_globals([np.core.multiarray.scalar])
            with safe_globals([np.core.multiarray.scalar]):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)

            state = ckpt["model_state_dict"]
            # Strip "module." prefix
            cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

            current = model.state_dict()
            loaded = 0
            for name, param in cleaned.items():
                if name in current and current[name].size() == param.size():
                    current[name].copy_(param)
                    loaded += 1
            model.load_state_dict(current)
            log.info(f"Loaded {loaded} parameters from pretrained model")

            if self.freeze_pretrained:
                for p in model.parameters():
                    p.requires_grad = False

            return model

        except Exception as e:
            log.error(f"Error loading pretrained weights: {e}. Using random init.")
            return model

    # ------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------
    @staticmethod
    def _build_adapter(adapter_type, d_model, n_time, n_static, params):
        if adapter_type == 'dual_residual':
            from models.neural_networks.adapters.dual_residual_adapter import DualResidualAdapter
            return DualResidualAdapter(d_model, n_time, n_static)
        elif adapter_type == 'gated':
            from models.neural_networks.adapters.gated_adapter import GatedAdapter
            return GatedAdapter(d_model, n_time)
        elif adapter_type == 'feedforward':
            from models.neural_networks.adapters.feedforward_adapter import FeedforwardAdapter
            return FeedforwardAdapter(d_model, n_time, params.get('hidden_multiplier', 2))
        elif adapter_type == 'conv':
            from models.neural_networks.adapters.conv_adapter import ConvAdapter
            return ConvAdapter(d_model, n_time, params.get('kernel_size', 3))
        elif adapter_type == 'attention':
            from models.neural_networks.adapters.attention_adapter import AttentionAdapter
            return AttentionAdapter(d_model, n_time, params.get('num_heads', 4))
        elif adapter_type == 'bottleneck':
            from models.neural_networks.adapters.bottleneck_adapter import BottleneckAdapter
            return BottleneckAdapter(d_model, n_time, params.get('bottleneck_size', 64))
        elif adapter_type == 'moe':
            from models.neural_networks.adapters.moe_adapter import MoEAdapter
            return MoEAdapter(d_model, n_time, params.get('num_experts', 4), params.get('expert_size', d_model))
        elif adapter_type == 'kriging_locality':
            from models.neural_networks.adapters.kriging_locality_adapter import KrigingLocalityAdapter
            return KrigingLocalityAdapter(
                d_model           = d_model,
                n_static          = n_static,
                n_time            = n_time,
                key_dim           = params.get('key_dim', 64),
                max_bank_size     = params.get('max_bank_size', 1024),
                bank_subsample    = params.get('bank_subsample', 256),
                bank_dropout      = params.get('bank_dropout', 0.3),
                dropout           = params.get('dropout', 0.1),
                use_fm_embeddings = params.get('use_fm_embeddings', False),
            )
        elif adapter_type == 'kriging_relational':
            from models.neural_networks.adapters.kriging_relational_adapter import RelationalKrigingAdapter
            return RelationalKrigingAdapter(
                d_model           = d_model,
                n_static          = n_static,
                n_time            = n_time,
                key_dim           = params.get('key_dim', 64),
                max_bank_size     = params.get('max_bank_size', 1024),
                bank_subsample    = params.get('bank_subsample', 256),
                bank_dropout      = params.get('bank_dropout', 0.3),
                dropout           = params.get('dropout', 0.1),
                use_fm_embeddings = params.get('use_fm_embeddings', False),
            )
        elif adapter_type == 'kriging_obs':
            from models.neural_networks.adapters.kriging_obs_adapter import KrigingObsAdapter
            return KrigingObsAdapter(
                d_model           = d_model,
                n_static          = n_static,
                n_time            = n_time,
                key_dim           = params.get('key_dim', 64),
                phi_hidden        = tuple(params.get('phi_hidden', [128, 128])),
                rho_hidden        = tuple(params.get('rho_hidden', [128])),
                weight_hidden     = tuple(params.get('weight_hidden', [64, 64])),
                max_bank_size     = params.get('max_bank_size', 1024),
                bank_subsample    = params.get('bank_subsample', 256),
                bank_dropout      = params.get('bank_dropout', 0.3),
                dropout           = params.get('dropout', 0.1),
                use_gate          = params.get('use_gate', True),
                use_fm_embeddings = params.get('use_fm_embeddings', False),
            )
        elif adapter_type == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

    # ------------------------------------------------------------------
    # Pretrained encoder
    # ------------------------------------------------------------------
    def _encode_pretrained(self, xc_pretrained_norm, temporal_features=None):
        """Run pretrained data through frozen encoder.

        Parameters
        ----------
        xc_pretrained_norm : [T, B, n_pre_ts + n_pre_static]
        temporal_features : [T, B, K] or None
        """
        n_ts = len(self.pretrained_ts_vars)
        n_st = len(self.pretrained_static_vars)

        batch_x = xc_pretrained_norm[..., :n_ts].permute(1, 0, 2)         # [B, T, n_ts]
        batch_c = xc_pretrained_norm[0, :, n_ts:n_ts + n_st]              # [B, n_st]

        B, T, F = batch_x.shape

        bd = {
            'batch_x': batch_x,
            'batch_c': batch_c,
            'batch_time_series_mask_index': torch.zeros(B, T, F, dtype=torch.bool, device=batch_x.device),
            'batch_static_mask_index': torch.zeros(B, n_st, dtype=torch.bool, device=batch_c.device),
            'mode': 'test',
        }

        if temporal_features is not None:
            if temporal_features.ndim == 3:
                bd['temporal_features'] = temporal_features.permute(1, 0, 2)  # [B, T, K]
            elif temporal_features.ndim == 2:
                bd['temporal_features'] = temporal_features.unsqueeze(0).expand(B, -1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            od = self.pretrained_model(bd, is_mask=False)

        # Project encoder output to d_model
        if not hasattr(self, "_enc_proj"):
            in_dim = od['outputs_time_series'].size(-1)
            self._enc_proj = nn.Linear(in_dim, self.d_model).to(od['outputs_time_series'].device)

        return self._enc_proj(od['outputs_time_series'])  # [B, T, d_model]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, xc_nn_norm, temporal_features=None, station_ids=None):
        """
        Parameters
        ----------
        xc_nn_norm : dict or torch.Tensor
            When dict (from dpl_model): contains 'xc_nn_norm', 'xc_pretrained_norm',
            'temporal_features'. When tensor: [T, B, F] fine-tuning features only.
        """
        # Unpack dict from dpl_model
        xc_pretrained_norm = None
        obs      = None
        obs_mask = None
        if isinstance(xc_nn_norm, dict):
            data_dict = xc_nn_norm
            xc_nn_norm = data_dict['xc_nn_norm']
            temporal_features = data_dict.get('temporal_features', temporal_features)
            xc_pretrained_norm = data_dict.get('xc_pretrained_norm', None)
            obs      = data_dict.get('obs', None)       # [B] or [B,1] observed target
            obs_mask = data_dict.get('obs_mask', None)  # [B] or [B,1] validity flag

        # Extract fine-tuning features (convert to batch-first)
        n_ts = len(self.finetuning_ts_vars)
        n_st = len(self.finetuning_static_vars)
        batch_x_ft = xc_nn_norm[..., :n_ts].permute(1, 0, 2)            # [B, T, n_ts]
        batch_c_ft = xc_nn_norm[0, :, n_ts:n_ts + n_st]                 # [B, n_st]

        # Encode with pretrained model
        if xc_pretrained_norm is not None:
            hidden = self._encode_pretrained(xc_pretrained_norm, temporal_features)
        else:
            raise ValueError(
                "DirectFinetuneing requires xc_pretrained_norm. "
                "Use NnDualLoader as data_loader and ensure pretrained_path is set."
            )

        # Normalize + scale
        hidden = self.embedding_norm(hidden) * self.embedding_scale

        # Adapter (uses fine-tuning features)
        adapter_type = self.model_config['adapter_type']
        if adapter_type in ['gated', 'feedforward', 'conv', 'attention', 'bottleneck', 'moe',
                            'dual_residual', 'kriging_locality', 'kriging_relational']:
            adapted = self.adapter(hidden, batch_x_ft, batch_c_ft)
        elif adapter_type == 'kriging_obs':
            adapted = self.adapter(hidden, batch_x_ft, batch_c_ft,
                                   obs=obs, obs_mask=obs_mask)
        elif adapter_type == 'none':
            adapted = self.adapter(hidden)
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")

        # Decode
        # CudnnLstm expects time-major [T, B, features]; permute from [B, T, d_model]
        if self.use_residual_lstm:
            static_exp = batch_c_ft.unsqueeze(1).expand(-1, adapted.size(1), -1)
            lstm_in = self.pre_lstm(torch.cat([adapted, batch_x_ft, static_exp], dim=-1))
            lstm_in_t = lstm_in.permute(1, 0, 2)  # [T, B, d_model]
            lstm_out_t, _ = self.decoder(lstm_in_t, do_drop_mc=False, dr_false=(not self.training))
            lstm_out = lstm_out_t.permute(1, 0, 2)  # [B, T, d_model]
            post = self.post_lstm(torch.cat([lstm_out, batch_x_ft, static_exp], dim=-1))
            output = (post + lstm_out).permute(1, 0, 2)  # [T, B, d_model]
        else:
            adapted_t = adapted.permute(1, 0, 2)  # [T, B, d_model]
            output, _ = self.decoder(adapted_t, do_drop_mc=False, dr_false=(not self.training))  # [T, B, hidden]

        return self.projection(output)  # [T, B, 1]
