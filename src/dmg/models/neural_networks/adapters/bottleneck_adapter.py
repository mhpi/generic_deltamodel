import torch
import torch.nn as nn

class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, input_size: int, bottleneck_size: int = 64):
        super().__init__()
        
        # Down-project to bottleneck
        self.down_project = nn.Sequential(
            nn.Linear(d_model + input_size, bottleneck_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Up-project back to model dimension
        self.up_project = nn.Sequential(
            nn.Linear(bottleneck_size, d_model),
            nn.Dropout(0.1)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
        combined = torch.cat([hidden_states, time_features], dim=-1)
        
        # Bottleneck transformation
        bottleneck = self.down_project(combined)
        adapted = self.up_project(bottleneck)
        
        # Residual connection
        output = self.layer_norm(hidden_states + adapted)
        return output