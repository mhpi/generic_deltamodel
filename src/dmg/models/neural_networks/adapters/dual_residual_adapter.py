#stefgen paper custom adapter split track for time and static features
import torch
import torch.nn as nn

class DualResidualAdapter(nn.Module):
    def __init__(self, d_model: int, n_time_features: int, n_static_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_time_features = n_time_features
        self.n_static_features = n_static_features
        
        self.time_transform = nn.Sequential(
            nn.Linear(n_time_features, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.static_transform = nn.Sequential(
            nn.Linear(n_static_features, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.combined_transform = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states, time_features, static_features):
        batch_size, seq_len, _ = hidden_states.shape
        
        time_proj = self.time_transform(time_features)
        static_proj = self.static_transform(static_features)
        static_expanded = static_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        combined = torch.cat([hidden_states, time_proj, static_expanded], dim=-1)
        combined_proj = self.combined_transform(combined)
        
        output = self.layer_norm1(hidden_states + combined_proj)
        output = self.layer_norm2(output + time_proj + static_expanded)
        
        return output