import torch
import torch.nn as nn
from dmg.models.neural_networks.informerLayers.Transformer_EncDec import Encoder, EncoderLayer
from dmg.models.neural_networks.informerLayers.SelfAttention_Family import ReformerLayer
from dmg.models.neural_networks.informerLayers.features_embedding import DataEmbedding
from dmg.models.neural_networks.informerLayers.validation_tools import Validator


# ---------------------------------------------------------------------
# Minimal config container (mirrors Informer adapter style)
# ---------------------------------------------------------------------
class _SimpleConfigs:
    def __init__(
        self,
        *,
        task_name: str,
        enc_in: int,
        c_out: int,
        d_model: int,
        dropout: float,
        num_heads: int = 4,
        num_enc_layers: int = 2,
        d_ffd: int = None,
        activation: str = "gelu",
        pred_len: int = 0,
        seq_len: int = 0,
    ):
        self.task_name = task_name
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.d_ffd = d_ffd if d_ffd is not None else 4 * d_model
        self.activation = activation
        self.pred_len = pred_len
        self.seq_len = seq_len


# ---------------------------------------------------------------------
# Core Reformer model (unchanged structure)
# ---------------------------------------------------------------------
class _ReformerCore(nn.Module):
    """Reformer with O(L log L) complexity."""

    def __init__(self, configs, bucket_size=4, n_hashes=4):
        super().__init__()
        self.config = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # embeddings
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.c_out, configs.d_model, configs.dropout)

        # Encoder: stack of Reformer layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(
                        None,
                        configs.d_model,
                        configs.num_heads,
                        bucket_size=bucket_size,
                        n_hashes=n_hashes,
                    ),
                    configs.d_model,
                    configs.d_ffd,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.num_enc_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        self.projection0 = nn.Linear(configs.c_out, configs.enc_in, bias=True)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def regression(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)

    def forward(self, data_dict):
        x_enc = data_dict["batch_x"]
        x_mark_enc = data_dict["batch_x_time_stamp"]
        batch_c = data_dict["batch_c"]

        if (batch_c is not None) and (len(batch_c) > 0):
            x_enc = Validator.combine_timeseries_and_statics(x_enc, batch_c)

        out = self.regression(x_enc, x_mark_enc)
        return {"outputs_time_series": out}


# ---------------------------------------------------------------------
# External adapter to unify interface with LSTM/Linear/etc.
# ---------------------------------------------------------------------
class Reformer(nn.Module):
    """
    Adapter wrapper for Reformer baseline.
    Compatible with:
        model = Reformer(nx=nx, ny=ny, hidden_size=hidden_size, dr=dr)
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: float = 0.0,
    ):
        super().__init__()
        self.name = "ReformerModel"
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dr = dr

        # create simple config for the internal core model
        cfg = _SimpleConfigs(
            task_name="regression",
            enc_in=nx,
            c_out=ny,
            d_model=hidden_size,
            dropout=dr,
        )

        self.core = _ReformerCore(cfg)

    def forward(self, x: torch.Tensor, do_drop_mc: bool = False, dr_false: bool = False):
        """
        x: [T, N, nx] → y: [T, N, ny]
        """
        if x.ndim != 3 or x.shape[2] != self.nx:
            raise ValueError(f"Expected input [T, N, {self.nx}], got {tuple(x.shape)}")

        T, N, _ = x.shape
        x_enc = x.permute(1, 0, 2).contiguous()  # [B, L, C]
        x_mark_enc = torch.zeros_like(x_enc[..., :1])  # placeholder time encoding

        data_dict = {
            "batch_x": x_enc,
            "batch_x_time_stamp": x_mark_enc,
            "batch_y": None,
            "batch_y_time_stamp": None,
            "batch_c": [],
        }

        out = self.core(data_dict)["outputs_time_series"]  # [B, L, ny]
        return out.permute(1, 0, 2).contiguous()           # [T, N, ny]
