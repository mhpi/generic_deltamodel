import torch
import torch.nn as nn
import torch.nn.functional as F
from dmg.models.neural_networks.informerLayers.Transformer_EncDec import (
    Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer
)
from dmg.models.neural_networks.informerLayers.SelfAttention_Family import ReformerLayer, AttentionLayer, ProbAttention
from dmg.models.neural_networks.informerLayers.features_embedding import DataEmbedding, LazyLinearEmbedding
from dmg.models.neural_networks.informerLayers.validation_tools import Validator



class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.config = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

        configs.distil = False

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.num_heads),
                    configs.d_model,
                    configs.d_ffd,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.num_enc_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.num_enc_layers - 1)
            ] if configs.distil and ('forecast' in configs.task_name) else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.num_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.num_heads),
                    configs.d_model,
                    configs.d_ffd,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.num_dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out  # [B, L, D]

    def regression(self, x_enc, x_mark_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, data_dict):

        x_enc = data_dict['batch_x']
        x_mark_enc = data_dict['batch_x_time_stamp']
        x_dec = data_dict['batch_y']
        x_mark_dec = data_dict['batch_y_time_stamp']
        batch_c = data_dict['batch_c']

        if (batch_c is not None) and (len(batch_c) > 0):
            x_enc = Validator.combine_timeseries_and_statics(x_enc, batch_c)

        if self.task_name in ['forecast']:
            x_dec, x_mark_dec = Validator.generate_dec_inp(self.config, data_dict)
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name in ['regression']:
            dec_out = self.regression(x_enc, x_mark_enc)

        return {"outputs_time_series": dec_out, }


# ------------------------------------------------------------------------------
# Minimal adapter to your package’s model format (no structural changes above)
# ------------------------------------------------------------------------------

class _SimpleConfigs:
    """Tiny namespace to satisfy the existing Informer `Model` without touching it."""
    def __init__(
        self,
        *,
        task_name: str,
        enc_in: int,
        c_out: int,
        d_model: int,
        dropout: float,
        factor: int = 5,
        output_attention: bool = False,
        num_heads: int = 4,
        num_enc_layers: int = 2,
        num_dec_layers: int = 1,
        d_ffd: int = None,
        activation: str = 'gelu',
        pred_len: int = 0,
        label_len: int = 0,
        distil: bool = False,
    ):
        self.task_name = task_name
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.dropout = dropout
        self.factor = factor
        self.output_attention = output_attention
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.d_ffd = d_ffd if d_ffd is not None else 4 * d_model
        self.activation = activation
        self.pred_len = pred_len
        self.label_len = label_len
        self.distil = distil


class Informer(nn.Module):
    """
    Adapter wrapper:
      - __init__(*, nx, ny, hidden_size, dr, ...)  (keyword-only like your LstmModel)
      - forward(x, do_drop_mc=False, dr_false=False)
      - x: [T, N, nx]  ->  y: [T, N, ny]  (regression mode)
    """
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: float = 0.0,
        # Optional Informer knobs with safe defaults (won’t change structure)
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 1,
        factor: int = 5,
        activation: str = 'gelu',
    ) -> None:
        super().__init__()
        self.name = "InformerModel"
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dr = dr

        # We run in "regression" mode to mirror LSTM’s time-aligned output.
        cfg = _SimpleConfigs(
            task_name='regression',
            enc_in=nx,
            c_out=ny,
            d_model=hidden_size,
            dropout=dr,
            factor=factor,
            output_attention=False,
            num_heads=n_heads,
            num_enc_layers=n_enc_layers,
            num_dec_layers=n_dec_layers,
            activation=activation,
            pred_len=0,
            label_len=0,
            distil=False,
        )
        self.core = Model(cfg)

    def forward(
        self,
        x: torch.Tensor,
        do_drop_mc: bool = False,   # kept for API parity; not used by Informer
        dr_false: bool = False,     # kept for API parity; not used by Informer
    ) -> torch.Tensor:
        """
        x: [T, N, nx]
        returns: [T, N, ny]
        """
        if x.ndim != 3 or x.shape[2] != self.nx:
            raise ValueError(f"InformerModel expected [T, N, {self.nx}], got {tuple(x.shape)}")

        T, N, _ = x.shape

        # Build the Chronos data_dict expected by your unchanged `Model`.
        # We provide zero time-stamps if none exist in this package format.
        # Shapes for Chronos are batch-first: [B, L, C] so we permute.
        x_enc = x.permute(1, 0, 2).contiguous()  # [B=N, L=T, C=nx]
        x_mark_enc = torch.zeros(N, T, 1, device=x.device, dtype=x.dtype)  # minimal placeholder
        data_dict = {
            'batch_x'            : x_enc,
            'batch_x_time_stamp' : x_mark_enc,
            'batch_y'            : None,   # not used in regression path
            'batch_y_time_stamp' : None,   # not used in regression path
            'batch_c'            : [],     # no statics by default
        }

        out = self.core(data_dict)["outputs_time_series"]   # [B, L, ny]
        y = out.permute(1, 0, 2).contiguous()               # -> [T, N, ny]
        return y
