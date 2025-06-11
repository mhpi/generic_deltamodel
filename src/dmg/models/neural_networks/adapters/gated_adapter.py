#Custom code but concept from  https://assets.amazon.science/a1/76/7d5695df4d4092e150b6f01a230c/gated-contextual-adapters-for-selective-contextual-biasing-in-neural-transducers.pdf
import torch
import torch.nn as nn

class GatedAdapter(nn.Module):
    def __init__(self, hidden_size: int, input_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size + input_size, hidden_size)
        self.transform = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, h: torch.Tensor, x: torch.Tensor, static_features=None):
        combined = torch.cat([h, x], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        transformed = self.transform(combined)
        return h + gate * transformed