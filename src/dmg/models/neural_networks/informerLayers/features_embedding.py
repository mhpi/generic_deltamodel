import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class FlexiblePositionalEmbedding(nn.Module):
    """
    Sine-cos positional encoding that auto-extends to any sequence length
    and matches device/dtype on the fly.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len, d_model), persistent=False)

    @staticmethod
    def _build_pe(max_len, d_model, device=None, dtype=None):
        pe = torch.zeros(max_len, d_model, device=device, dtype=dtype if dtype is not None else torch.float32)
        position = torch.arange(0, max_len, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, L, C]
        B, L = x.size(0), x.size(1)
        if self.pe.size(1) < L or self.pe.device != x.device or self.pe.dtype != x.dtype:
            # rebuild with at least L and on the correct device/dtype
            new_len = max(L, self.pe.size(1))
            self.pe = self._build_pe(new_len, self.d_model, device=x.device, dtype=x.dtype)
        return self.pe[:, :L]  # [1, L, d_model]


class LinearEmbedding(nn.Module):
    """
    Fixed-width linear embedding (for known input size).
    """
    def __init__(self, c_in, d_model, initrange=None, bias=False):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(c_in, d_model, bias=bias)
        self._reset_parameters(initrange)

    def _reset_parameters(self, initrange):
        if initrange is None:
            initrange = 1.0 / math.sqrt(self.d_model)
        with torch.no_grad():
            for p in self.parameters():
                p.uniform_(-initrange, initrange)

    def forward(self, x):
        # x: [B, L, c_in]
        return self.linear(x)


class LazyLinearEmbedding(nn.Module):
    """
    Lazy linear embedding that adapts to ANY number of input time features.
    Uses LazyLinear so in_features are inferred on first forward.
    """
    def __init__(self, d_model, initrange=None, bias=False):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.LazyLinear(d_model, bias=bias)
        self.initrange = initrange
        self._initialized = False

    def _maybe_init(self):
        if self._initialized:
            return
        # Only safe to initialize after the first forward created weights.
        if self.linear.weight is not None and self.linear.weight.numel() > 0:
            initrange = self.initrange if self.initrange is not None else (1.0 / math.sqrt(self.d_model))
            with torch.no_grad():
                self.linear.weight.uniform_(-initrange, initrange)
                if self.linear.bias is not None:
                    self.linear.bias.zero_()
            self._initialized = True

    def forward(self, x):
        # x: [B, L, K] with arbitrary K
        y = self.linear(x)
        self._maybe_init()
        return y


class DataEmbedding(nn.Module):
    """
    Value embedding (fixed c_in) + time embedding (ANY K) + positional encoding.
    """
    def __init__(self, c_in, d_model, dropout=0.05):
        super().__init__()
        self.value_embedding = LinearEmbedding(c_in=c_in, d_model=d_model, bias=False)
        self.time_embedding  = LazyLinearEmbedding(d_model=d_model, bias=False)   # <-- accepts any x_mark last-dim
        self.position_embedding = FlexiblePositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x:       [B, L, c_in]
        x_mark:  [B, L, K] or None (ANY K)
        """
        out = self.value_embedding(x) + self.position_embedding(x)
        if x_mark is not None:
            out = out + self.time_embedding(x_mark)
        return self.dropout(out)


class DataEmbedding_wo_pos(nn.Module):
    """
    Same as DataEmbedding but without positional encoding.
    """
    def __init__(self, c_in, d_model, dropout=0.05):
        super().__init__()
        self.value_embedding = LinearEmbedding(c_in=c_in, d_model=d_model, bias=False)
        self.time_embedding  = LazyLinearEmbedding(d_model=d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        out = self.value_embedding(x)
        if x_mark is not None:
            out = out + self.time_embedding(x_mark)
        return self.dropout(out)
