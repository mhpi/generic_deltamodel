import logging
import os
import re
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from models.neural_networks.adapters.build_adapter import apply_adapter, build_adapter
from models.neural_networks.cudnn_lstm import CudnnLstm
from models.neural_networks.transformer.StefaLandDecLSTM import (
    Model as StefaLandDecLSTM,
)
from omegaconf import DictConfig, OmegaConf

warnings.filterwarnings(
    "ignore", message=".*weights are not part of single contiguous chunk.*"
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Frozen-encoder helpers, factored out of DirectFinetuneing so both the
# on-the-fly model and scripts/generate_embeddings.py (offline batch
# encoding to precomputed embedding files) share one checkpoint-loading
# and encoding implementation. Keeping this in one place means an offline
# embedding file and a live DirectFinetuneing forward pass can never drift
# out of numerical sync with each other.
# ----------------------------------------------------------------------
def build_pretrained_encoder(
    d_model: int,
    num_heads: int,
    dropout: float,
    num_enc_layers: int,
    d_ffd: int,
    pretrained_ts_vars: List[str],
    pretrained_static_vars: List[str],
    pretrained_model_path: Optional[str],
    freeze: bool = True,
    pretrained_type: str = 'stefaland_dec_lstm',
) -> nn.Module:
    """Build a frozen pretrained encoder and load checkpoint weights into it.

    The architecture is StefaLandDecLSTM: a TransformerBackbone encoder
    with a single-layer LSTM decoder.
    """
    cfg = type(
        'Config',
        (),
        {
            'd_model': d_model,
            'num_heads': num_heads,
            'dropout': dropout,
            'num_enc_layers': num_enc_layers,
            # Gates whether StefaLandDecLSTM.forward() runs its decoder LSTM
            # before the masked-reconstruction output heads; the decoder
            # submodule is always registered regardless (needed either way
            # for the checkpoint's weights to load), and encode_with_pretrained
            # only reads the pre-decoder encoder_hidden_time_series/static, so
            # this has no effect on what we actually use -- a constant, not
            # worth threading through as a real config knob.
            'num_dec_layers': 1,
            'd_ffd': d_ffd,
            'time_series_variables': pretrained_ts_vars,
            'static_variables': pretrained_static_vars,
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
        },
    )
    if pretrained_type != 'stefaland_dec_lstm':
        raise ValueError(
            f"Unknown pretrained_type '{pretrained_type}'. This release "
            "ships one frozen encoder, 'stefaland_dec_lstm'."
        )
    model = StefaLandDecLSTM(cfg).float()
    return _load_pretrained_weights(model, pretrained_model_path, freeze)


def _load_pretrained_weights(
    model: nn.Module, path: Optional[str], freeze: bool
) -> nn.Module:
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

        from torch.serialization import add_safe_globals, safe_globals

        add_safe_globals([np.core.multiarray.scalar])
        with safe_globals([np.core.multiarray.scalar]):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        state = ckpt["model_state_dict"]
        # Strip "module." prefix
        cleaned = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state.items()
        }

        current = model.state_dict()
        loaded = 0
        for name, param in cleaned.items():
            if name in current and current[name].size() == param.size():
                current[name].copy_(param)
                loaded += 1
        model.load_state_dict(current)
        log.info(f"Loaded {loaded} parameters from pretrained model")

        if freeze:
            for p in model.parameters():
                p.requires_grad = False

        return model

    except Exception as e:
        log.error(f"Error loading pretrained weights: {e}. Using random init.")
        return model


def encode_with_pretrained(
    model: nn.Module,
    pretrained_ts_vars: List[str],
    pretrained_static_vars: List[str],
    xc_pretrained_norm: torch.Tensor,
    temporal_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run pretrained-format inputs through a frozen encoder.

    Parameters
    ----------
    xc_pretrained_norm : [T, B, n_pre_ts + n_pre_static]
    temporal_features : [T, B, K] or None

    Returns
    -------
    torch.Tensor
        [B, T, d_model] -- raw encoder embeddings (not the depatcher's
        reconstructed [B, T, F] output), fusing the per-timestep dynamic
        embedding with the pooled static embedding so both are reflected
        downstream.
    """
    n_ts = len(pretrained_ts_vars)
    n_st = len(pretrained_static_vars)

    batch_x = xc_pretrained_norm[..., :n_ts].permute(1, 0, 2)  # [B, T, n_ts]
    batch_c = xc_pretrained_norm[0, :, n_ts : n_ts + n_st]  # [B, n_st]

    B, T, F = batch_x.shape

    bd = {
        'batch_x': batch_x,
        'batch_c': batch_c,
        'batch_time_series_mask_index': torch.zeros(
            B, T, F, dtype=torch.bool, device=batch_x.device
        ),
        'batch_static_mask_index': torch.zeros(
            B, n_st, dtype=torch.bool, device=batch_c.device
        ),
        'mode': 'test',
    }

    if temporal_features is not None:
        if temporal_features.ndim == 3:
            tf = temporal_features.permute(1, 0, 2)  # [B, T', K]
        elif temporal_features.ndim == 2:
            tf = temporal_features.unsqueeze(0).expand(B, -1, -1)  # [B, T', K]
        else:
            tf = None
        if tf is not None:
            T_in = tf.shape[1]
            if T_in > T:
                tf = tf[:, :T, :]
            elif T_in < T:
                pad = tf.new_zeros(B, T - T_in, tf.shape[2])
                tf = torch.cat([tf, pad], dim=1)
            bd['temporal_features'] = tf

    with torch.amp.autocast('cuda', enabled=False):
        od = model(bd, is_mask=False)

    hidden_ts = od['encoder_hidden_time_series']  # [B, T, d_model]
    hidden_static = od['encoder_hidden_static']  # [B, d_model]

    return hidden_ts + hidden_static.unsqueeze(1)  # [B, T, d_model]


class DirectFinetuneing(nn.Module):
    """Fine-tuning module that uses pretrained data directly (xc_pretrained_norm)
    through a frozen pretrained encoder, then adapts with trainable adapter + LSTM.

    Expects NnDualLoader which provides both xc_nn_norm and xc_pretrained_norm.
    All data flows through dpl_model as a Dict.

    ACCEPTS_BATCH_DICT = True signals NnModel to pass the full batch Dict rather
    than extracting xc_nn_norm, since this model also needs xc_pretrained_norm.
    """

    ACCEPTS_BATCH_DICT = True

    def __init__(self, config: Union[Dict, DictConfig], ny) -> None:
        super().__init__()

        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        if 'nn' in config_dict:
            dpl_config = config_dict
            nn_config = config_dict['nn']
        else:
            dpl_config = config_dict.get('model', {})
            nn_config = dpl_config.get('nn', {})

        # Pretrained variable lists (defines pretrained model architecture)
        self.pretrained_ts_vars = nn_config.get('pretrained_time_series_vars', [])
        self.pretrained_static_vars = nn_config.get('pretrained_static_vars', [])
        if not self.pretrained_ts_vars or not self.pretrained_static_vars:
            raise ValueError(
                "Must specify 'pretrained_time_series_vars' and 'pretrained_static_vars'"
            )

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
        self.lstm_hidden_size = nn_config.get('lstm_hidden_size', self.d_model)
        self.target_variables = ny

        self.model_config = {
            'd_model': self.d_model,
            'num_heads': nn_config.get('num_heads', 4),
            'dropout': self.dropout,
            'num_enc_layers': nn_config.get('num_enc_layers', 4),
            'd_ffd': nn_config.get('d_ffd', 512),
            'pretrained_model': nn_config.get('pretrained_model'),
            'pretrained_type': nn_config.get('pretrained_type', 'stefaland_dec_lstm'),
            'adapter_type': nn_config.get('adapter_type', 'dual_residual'),
            'adapter_params': nn_config.get('adapter_params', {}),
        }

        # Initialize pretrained encoder
        self.pretrained_model = self._build_pretrained_model()

        # Initialize adapter
        self.adapter = build_adapter(
            self.model_config['adapter_type'],
            self.d_model,
            n_ft_ts,
            n_ft_static,
            self.model_config['adapter_params'],
        )

        # LSTM decoder
        # nx stays at d_model (matches the adapter output the encoder/adapter
        # produce); hidden_size is independently configurable so the decoder
        # can be swept without touching d_model, which must stay fixed to
        # keep the frozen pretrained encoder's weights loadable.
        self.decoder = CudnnLstm(
            nx=self.d_model,
            hidden_size=self.lstm_hidden_size,
            dr=self.dropout,
        )

        # Residual LSTM components
        if self.use_residual_lstm:
            self.pre_lstm = nn.Linear(
                self.d_model + n_ft_ts + n_ft_static, self.d_model
            )
            self.post_lstm = nn.Linear(
                self.lstm_hidden_size + n_ft_ts + n_ft_static, self.lstm_hidden_size
            )

        # Embedding normalization and scaling
        self.embedding_norm = nn.LayerNorm(self.d_model)
        self.embedding_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Final projection to target
        self.projection = nn.Linear(self.lstm_hidden_size, self.target_variables)

        log.info(
            f"Initialized DirectFinetuneing: adapter={self.model_config['adapter_type']}"
        )
        log.info(
            f"Pretrained vars: {len(self.pretrained_ts_vars)} ts, {len(self.pretrained_static_vars)} static"
        )
        log.info(f"Fine-tuning vars: {n_ft_ts} ts, {n_ft_static} static")

        if self.freeze_pretrained:
            self.pretrained_model.eval()

    def train(self, mode: bool = True) -> 'DirectFinetuneing':
        """Set training mode; keep the frozen pretrained encoder in eval.

        nn.Module.train() recursively puts all submodules into train mode,
        which would re-enable dropout inside the pretrained encoder every
        time the outer model calls .train() (once per epoch). A frozen
        encoder should produce deterministic embeddings, so force it back
        to eval whenever freeze_pretrained is set.
        """
        super().train(mode)
        if self.freeze_pretrained:
            self.pretrained_model.eval()
        return self

    # ------------------------------------------------------------------
    # Pretrained model
    # ------------------------------------------------------------------
    def _build_pretrained_model(self):
        return build_pretrained_encoder(
            self.d_model,
            self.model_config['num_heads'],
            self.dropout,
            self.model_config['num_enc_layers'],
            self.model_config['d_ffd'],
            self.pretrained_ts_vars,
            self.pretrained_static_vars,
            self.model_config['pretrained_model'],
            freeze=self.freeze_pretrained,
            pretrained_type=self.model_config['pretrained_type'],
        )

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
        return encode_with_pretrained(
            self.pretrained_model,
            self.pretrained_ts_vars,
            self.pretrained_static_vars,
            xc_pretrained_norm,
            temporal_features,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, xc_nn_norm, temporal_features=None, station_ids=None):
        """
        Parameters
        ----------
        xc_nn_norm : Dict or torch.Tensor
            When Dict (from dpl_model): contains 'xc_nn_norm', 'xc_pretrained_norm',
            'temporal_features'. When tensor: [T, B, F] fine-tuning features only.
        """
        # Unpack Dict from dpl_model
        xc_pretrained_norm = None
        obs = None
        obs_mask = None
        if isinstance(xc_nn_norm, Dict):
            data_dict = xc_nn_norm
            xc_nn_norm = data_dict['xc_nn_norm']
            temporal_features = data_dict.get('temporal_features', temporal_features)
            xc_pretrained_norm = data_dict.get('xc_pretrained_norm', None)
            obs = data_dict.get('obs', None)  # [B] or [B,1] observed target
            obs_mask = data_dict.get('obs_mask', None)  # [B] or [B,1] validity flag

        # Extract fine-tuning features (convert to batch-first)
        n_ts = len(self.finetuning_ts_vars)
        n_st = len(self.finetuning_static_vars)
        batch_x_ft = xc_nn_norm[..., :n_ts].permute(1, 0, 2)  # [B, T, n_ts]
        batch_c_ft = xc_nn_norm[0, :, n_ts : n_ts + n_st]  # [B, n_st]

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

        # Align T: pretrained data may cover fewer timesteps than the task window
        # (e.g. pretrained NetCDF starts later, giving a shorter warmup slice).
        # During training the sampler always pads, so T always matches.
        # During eval we slice directly, so T may differ.
        T_hidden = hidden.shape[1]
        T_task = batch_x_ft.shape[1]
        if T_hidden < T_task:
            pad = hidden.new_zeros(hidden.shape[0], T_task - T_hidden, hidden.shape[2])
            hidden = torch.cat([hidden, pad], dim=1)
        elif T_hidden > T_task:
            hidden = hidden[:, :T_task, :]

        # Adapter (uses fine-tuning features)
        adapter_type = self.model_config['adapter_type']
        adapted = apply_adapter(
            self.adapter,
            adapter_type,
            hidden,
            batch_x_ft,
            batch_c_ft,
            obs=obs,
            obs_mask=obs_mask,
        )

        # Decode
        # CudnnLstm expects time-major [T, B, features]; permute from [B, T, d_model]
        if self.use_residual_lstm:
            static_exp = batch_c_ft.unsqueeze(1).expand(-1, adapted.size(1), -1)
            lstm_in = self.pre_lstm(
                torch.cat([adapted, batch_x_ft, static_exp], dim=-1)
            )
            lstm_in_t = lstm_in.permute(1, 0, 2)  # [T, B, d_model]
            lstm_out_t, _ = self.decoder(
                lstm_in_t, do_drop_mc=False, dr_false=(not self.training)
            )
            lstm_out = lstm_out_t.permute(1, 0, 2)  # [B, T, d_model]
            post = self.post_lstm(torch.cat([lstm_out, batch_x_ft, static_exp], dim=-1))
            output = (post + lstm_out).permute(1, 0, 2)  # [T, B, d_model]
        else:
            adapted_t = adapted.permute(1, 0, 2)  # [T, B, d_model]
            output, _ = self.decoder(
                adapted_t, do_drop_mc=False, dr_false=(not self.training)
            )  # [T, B, hidden]

        return self.projection(output)  # [T, B, 1]
