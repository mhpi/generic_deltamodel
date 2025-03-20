from typing import Any

import torch
import torch.nn as nn


class MlpModel(nn.Module):
    """Multi-layer perceptron (MLP) model.
    
    Parameters
    ----------
    config
        Configuration dictionary with model settings.
    """
    def __init__(self, config):
        super().__init__()
        self.name = 'MlpModel'
        self.L1 = nn.Linear(
            len(config['optData']['varC']), config['seq_lin_layers']['hidden_size']
        )
        self.L2 = nn.Linear(
            config['seq_lin_layers']['hidden_size'], config['seq_lin_layers']['hidden_size']
        )
        self.L3 = nn.Linear(
            config['seq_lin_layers']['hidden_size'], config['seq_lin_layers']['hidden_size']
        )

        self.L4 = nn.Linear(config['seq_lin_layers']['hidden_size'], 23)

        # 6 for alpha and beta of surface/subsurface/groundwater flow
        # 3 for conv bias,
        # 2 for scaling and bias of final answer,
        # 1 for shade_factor_riparian
        # 3 for surface/subsurface/groundwater flow percentage
        # 1 for albedo
        # 1 for solar shade factor
        # 4 for width coefficient nominator, width coefficient denominator, width A coefficient, and width exponent
        # 2 for p & q

        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x
            Input tensor.
        
        Returns
        -------
        out
            Output tensor.
        """
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation_sigmoid(out)
        return out


class MlpMulModel(nn.Module):
    """Multi-layer perceptron (MLP) model with multiple layers.
    
    Parameters
    ----------
    config
        Configuration dictionary with model settings.
    nx
        Number of input features.
    ny
        Number of output features.
    """
    def __init__(self, config: dict[str, Any], nx: int, ny: int) -> None:
        super().__init__()
        self.name = 'MlpMulModel'
        self.config = config
        self.L1 = nn.Linear(
            nx,  self.config['hidden_size'],
        )
        self.L2 = nn.Linear(
            self.config['hidden_size'], self.config['hidden_size']
        )
        self.L3 = nn.Linear(
            self.config['hidden_size'], self.config['hidden_size']
        )

        self.L4 = nn.Linear(self.config['hidden_size'], ny)
        self.activation_sigmoid = torch.nn.Sigmoid()
        self.activation_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x
            Input tensor.
        
        Returns
        -------
        out
            Output tensor.
        """
        # out = self.seq_lin_layers(x)
        out = self.L1(x)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        # out1 = torch.abs(out)
        out = self.activation_sigmoid(out)
        return out
