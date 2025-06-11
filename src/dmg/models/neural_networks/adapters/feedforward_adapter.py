import torch
import torch.nn as nn

class FeedforwardAdapter(nn.Module):
    def __init__(self, d_model: int, input_size: int, hidden_multiplier: int = 2):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(d_model + input_size, d_model * hidden_multiplier),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * hidden_multiplier, d_model)
        )
    
    def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
        combined = torch.cat([hidden_states, time_features], dim=-1)
        return self.adapter(combined)