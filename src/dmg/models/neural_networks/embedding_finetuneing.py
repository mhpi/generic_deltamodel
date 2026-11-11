import logging
from typing import Dict, Union

import torch
import torch.nn as nn
from models.neural_networks.adapters.build_adapter import apply_adapter, build_adapter
from models.neural_networks.cudnn_lstm import CudnnLstm
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class EmbeddingFinetuneing(nn.Module):
    """
     this model has no frozen encoder architecture
    to match, so it makes no assumption about embedding width. 'embedding_size'
    in config must equal the last dimension of the precomputed embeddings
    (e.g. 256, 1024, ...); every downstream layer is sized from that single
    value, so pointing at a differently-sized foundation-model release only
    requires updating that one number.

    ACCEPTS_BATCH_DICT = True signals NnModel to pass the full batch Dict
    rather than extracting xc_nn_norm, since this model also needs the
    embedding stream (carried under the 'xc_pretrained_norm' key for
    compatibility with the existing samplers).
    """

    ACCEPTS_BATCH_DICT = True

    def __init__(self, config: Union[Dict, DictConfig], ny) -> None:
        super().__init__()

        if isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            config_dict = config

        if 'nn' in config_dict:
            nn_config = config_dict['nn']
        else:
            nn_config = config_dict.get('model', {}).get('nn', {})

        # Fine-tuning variable lists (for adapter and LSTM)
        self.finetuning_ts_vars = nn_config.get('forcings', [])
        self.finetuning_static_vars = nn_config.get('attributes', [])
        if not self.finetuning_ts_vars or not self.finetuning_static_vars:
            raise ValueError("Must specify 'forcings' and 'attributes'")

        n_ft_ts = len(self.finetuning_ts_vars)
        n_ft_static = len(self.finetuning_static_vars)

        # Embedding width: must match whatever EmbeddingFinetuneLoader loaded
        # from the precomputed embedding file. Not inferred from a checkpoint
        # (there isn't one) -- it comes straight from config.
        self.d_model = nn_config.get('embedding_size') or nn_config.get('hidden_size')
        if not self.d_model:
            raise ValueError(
                "Must specify 'embedding_size' (or 'hidden_size') matching the "
                "last dimension of the precomputed embeddings this model will "
                "receive (e.g. 256, 1024, ...)."
            )

        self.dropout = nn_config.get('dropout', 0.1)
        self.use_residual_lstm = nn_config.get('use_residual_lstm', False)
        self.lstm_hidden_size = nn_config.get('lstm_hidden_size', self.d_model)
        self.target_variables = ny

        self.adapter_type = nn_config.get('adapter_type', 'dual_residual')
        adapter_params = nn_config.get('adapter_params', {})

        # Architecture ablations. 'none' is the full model and is the only
        # mode that touches the adapter / residual-LSTM machinery, so leaving
        # this key out of a config reproduces the pre-ablation behaviour
        # exactly. The two ablation modes deliberately bypass parts of the
        # architecture rather than zeroing them out, so their parameter counts
        # (not just their outputs) reflect what is actually being tested:
        #
        #   'linear_probe'       -- Linear(LayerNorm(embedding)) -> ny, and
        #       nothing else: no adapter, no LSTM decoder, and no task
        #       forcings/attributes reach the network at all. Measures how
        #       much of the parameterization the frozen embedding determines
        #       on its own. (Under a delta model the HBV physics still gets
        #       its own x_phy forcings; only the NN is restricted.)
        #
        #   'embedding_as_input' -- the embedding is concatenated to the task
        #       forcings as ordinary time-varying LSTM input channels, and the
        #       static attributes are dropped. Structurally this is the
        #       CudnnLstmModel baseline with its static-attribute block
        #       replaced by the 64-d daily embedding, which is what makes it
        #       comparable to LSTMHBVPUB_*.
        #
        #
        # The no-foundation-model control (raw FM input variables fed straight
        # to the adapter) lives in its own model, RawFmInputsFinetuneing --
        # see raw_fm_inputs_finetuneing.py -- because it reads a different data
        # source and never touches an embedding at all.
        self.ablation_mode = nn_config.get('ablation_mode', 'none')
        valid_modes = ('none', 'linear_probe', 'embedding_as_input')
        if self.ablation_mode not in valid_modes:
            raise ValueError(
                f"Unsupported ablation_mode '{self.ablation_mode}'. "
                f"Expected one of {valid_modes}."
            )

        # Embedding normalization. The 0.1-initialised learnable scale exists
        # to stop the hidden stream from swamping the adapter's residual path,
        # so it only applies to the full model -- see forward().
        self.embedding_norm = nn.LayerNorm(self.d_model)

        if self.ablation_mode == 'linear_probe':
            self.projection = nn.Linear(self.d_model, self.target_variables)

        elif self.ablation_mode == 'embedding_as_input':
            self.decoder = CudnnLstm(
                nx=self.d_model + n_ft_ts,
                hidden_size=self.lstm_hidden_size,
                dr=self.dropout,
            )
            self.projection = nn.Linear(self.lstm_hidden_size, self.target_variables)

        else:
            # Adapter
            self.adapter = build_adapter(
                self.adapter_type,
                self.d_model,
                n_ft_ts,
                n_ft_static,
                adapter_params,
            )

            # LSTM decoder
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

            self.embedding_scale = nn.Parameter(torch.ones(1) * 0.1)

            # Final projection to target
            self.projection = nn.Linear(self.lstm_hidden_size, self.target_variables)

        log.info(
            f"Initialized EmbeddingFinetuneing: ablation_mode={self.ablation_mode}, "
            f"adapter={self.adapter_type if self.ablation_mode == 'none' else 'bypassed'}, "
            f"embedding_size={self.d_model}"
        )
        log.info(f"Fine-tuning vars: {n_ft_ts} ts, {n_ft_static} static")
        if self.ablation_mode == 'linear_probe':
            log.info(
                "linear_probe: task forcings/attributes are loaded but NOT fed "
                "to the network."
            )
        elif self.ablation_mode == 'embedding_as_input':
            log.info(
                f"embedding_as_input: LSTM sees {n_ft_ts} forcings + "
                f"{self.d_model} embedding channels; the {n_ft_static} static "
                "attributes are loaded but NOT fed to the network."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, xc_nn_norm, temporal_features=None):
        """
        Parameters
        ----------
        xc_nn_norm : Dict or torch.Tensor
            When Dict (from dpl_model/NnModel): contains 'xc_nn_norm' and the
            precomputed embedding stream under 'xc_pretrained_norm', shape
            [T, B, embedding_size]. When tensor: [T, B, F] fine-tuning
            features only (embeddings must then be unavailable, which raises).
        """
        embedding = None
        obs = None
        obs_mask = None
        if isinstance(xc_nn_norm, Dict):
            data_dict = xc_nn_norm
            xc_nn_norm = data_dict['xc_nn_norm']
            embedding = data_dict.get('xc_pretrained_norm', None)
            obs = data_dict.get('obs', None)
            obs_mask = data_dict.get('obs_mask', None)

        # Extract fine-tuning features (convert to batch-first)
        n_ts = len(self.finetuning_ts_vars)
        n_st = len(self.finetuning_static_vars)
        batch_x_ft = xc_nn_norm[..., :n_ts].permute(1, 0, 2)  # [B, T, n_ts]
        batch_c_ft = xc_nn_norm[0, :, n_ts : n_ts + n_st]  # [B, n_st]

        if embedding is None:
            raise ValueError(
                "EmbeddingFinetuneing requires precomputed embeddings. "
                "Use EmbeddingFinetuneLoader as data_loader and set "
                "'embedding_path' in the config."
            )

        hidden = embedding.permute(1, 0, 2)  # [T, B, D] -> [B, T, D]

        if hidden.shape[-1] != self.d_model:
            raise ValueError(
                f"Precomputed embedding width ({hidden.shape[-1]}) does not "
                f"match configured 'embedding_size' ({self.d_model}). Update "
                "the config to match the embedding file actually being loaded."
            )

        # Normalize + scale, same treatment DirectFinetuneing gives its
        # on-the-fly encoder output, so adapters see consistently-scaled input
        # whichever model produced it. The 0.1 scale is an adapter-path
        # concern; in the ablation modes the embedding is consumed as an
        # ordinary input feature, where LayerNorm's unit scale is what matches
        # the z-scored task inputs it sits beside.
        hidden = self.embedding_norm(hidden)
        if self.ablation_mode == 'none':
            hidden = hidden * self.embedding_scale

        # Align T: precomputed embeddings may cover fewer timesteps than the
        # task window (e.g. embedding file starts later than the task data).
        T_hidden = hidden.shape[1]
        T_task = batch_x_ft.shape[1]
        if T_hidden < T_task:
            pad = hidden.new_zeros(hidden.shape[0], T_task - T_hidden, hidden.shape[2])
            hidden = torch.cat([hidden, pad], dim=1)
        elif T_hidden > T_task:
            hidden = hidden[:, :T_task, :]

        # --- Ablation short-circuits (see __init__ for what each isolates) ---
        if self.ablation_mode == 'linear_probe':
            # [B, T, D] -> [B, T, ny] -> [T, B, ny]
            return self.projection(hidden).permute(1, 0, 2)

        if self.ablation_mode == 'embedding_as_input':
            # Embedding rides in as extra time-varying channels alongside the
            # forcings; batch_c_ft is intentionally unused.
            lstm_in_t = torch.cat([hidden, batch_x_ft], dim=-1).permute(1, 0, 2)
            lstm_out_t, _ = self.decoder(
                lstm_in_t, do_drop_mc=False, dr_false=(not self.training)
            )  # [T, B, lstm_hidden_size]
            return self.projection(lstm_out_t)  # [T, B, ny]

        # Adapter (uses fine-tuning features)
        adapted = apply_adapter(
            self.adapter,
            self.adapter_type,
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

        return self.projection(output)  # [T, B, ny]
