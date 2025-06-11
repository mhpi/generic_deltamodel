import torch
import torch.nn as nn

class ConvAdapter(nn.Module):
    def __init__(self, d_model: int, input_size: int, kernel_size: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(d_model + input_size, d_model)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(d_model * 2, d_model, kernel_size=kernel_size, padding=kernel_size//2)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
        combined = torch.cat([hidden_states, time_features], dim=-1)
        projected = self.input_projection(combined)
        
        # Conv1d expects [batch, channels, seq_len]
        conv_input = projected.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)
        conv_output = conv_output.transpose(1, 2)
        
        # Residual connection
        output = self.layer_norm(hidden_states + conv_output)
        return output