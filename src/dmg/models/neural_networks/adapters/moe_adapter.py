import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEAdapter(nn.Module):
    def __init__(self, d_model: int, input_size: int, num_experts: int = 4, expert_size: int = None):
        super().__init__()
        self.num_experts = num_experts
        self.expert_size = expert_size or d_model
        
        # Gating network
        self.gate = nn.Linear(d_model + input_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + input_size, self.expert_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.expert_size, d_model)
            ) for _ in range(num_experts)
        ])
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
        combined = torch.cat([hidden_states, time_features], dim=-1)
        
        # Compute gating weights
        gate_weights = F.softmax(self.gate(combined), dim=-1)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(combined))
        
        # Weight and combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [..., d_model, num_experts]
        gate_weights = gate_weights.unsqueeze(-2)  # [..., 1, num_experts]
        
        # Weighted combination
        adapted = torch.sum(expert_outputs * gate_weights, dim=-1)
        
        # Residual connection
        output = self.layer_norm(hidden_states + adapted)
        return output
        