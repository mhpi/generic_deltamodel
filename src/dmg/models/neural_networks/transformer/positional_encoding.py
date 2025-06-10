import torch
from torch import nn
import numpy as np


# positional encoding it can learn the position information by itself
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, d_model), requires_grad=True)

    def forward(self, input_data, index=None):
        if index is None:
            index = np.arange(input_data.size(1))

        pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        return input_data
