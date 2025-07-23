import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

#------------------------------------------#
# NOTE: Suppress this warning until we can implement a proper pytorch nn.LSTM.
warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")
#------------------------------------------#


class FireLstm(torch.nn.Module):
    """Fire-optimized LSTM model for fire occurrence prediction.

    Parameters
    ----------
    nx
        Number of input features.
    ny
        Number of output features.
    hidden_size
        Number of hidden units.
    dr
        Dropout rate.
    """
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
    ) -> None:
        super().__init__()
        self.name = 'FireLstmModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.ct = 0
        self.n_layers = 1

        # Input normalization and projection
        self.input_norm = nn.LayerNorm(nx)
        self.linear_in = torch.nn.Linear(nx, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            batch_first=False,
            dropout=dr if dr > 0 else 0
        )
        
        # Attention mechanism for fire-relevant features
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Output layers
        self.linear_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(hidden_size // 2, ny)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(
        self,
        x,
        do_drop_mc: Optional[bool] = False,
        dr_false: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x
            The input tensor.
        do_drop_mc
            Flag for applying dropout.
        dr_false
            Flag for applying dropout.
        """
        # Input normalization and projection
        x = self.input_norm(x)
        x0 = F.relu(self.linear_in(x))
        
        # LSTM processing
        lstm_out, _ = self.lstm(x0)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attended_out = lstm_out * attention_weights
        
        # Final output
        return self.linear_out(attended_out)