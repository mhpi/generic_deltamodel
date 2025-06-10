import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerBackbone(nn.Module):
    def __init__(self, d_model, num_layers, d_ffd, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(d_model, num_heads, d_ffd, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        src = src * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1)
        return output
