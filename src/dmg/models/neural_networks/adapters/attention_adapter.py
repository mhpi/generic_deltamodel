import torch
import torch.nn as nn

class AttentionAdapter(nn.Module):
    def __init__(self, d_model: int, input_size: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
        # Project time features to model dimension
        time_proj = self.input_projection(time_features)
        
        # Self-attention on hidden states
        attn_output, _ = self.self_attention(hidden_states, hidden_states, hidden_states)
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Cross-attention between hidden states and time features
        cross_attn_output, _ = self.cross_attention(hidden_states, time_proj, time_proj)
        hidden_states = self.layer_norm2(hidden_states + cross_attn_output)
        
        # Feedforward
        ff_output = self.feedforward(hidden_states)
        output = self.layer_norm3(hidden_states + ff_output)
        
        return output