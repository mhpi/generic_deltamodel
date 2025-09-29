from typing import Any

import torch
import torch.nn as nn


class MlpModel(nn.Module):
    """Multi-layer perceptron (MLP) model.

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
            nx,
            self.config['hidden_size'],
        )
        self.L2 = nn.Linear(
            self.config['hidden_size'],
            self.config['hidden_size'],
        )
        self.L3 = nn.Linear(
            self.config['hidden_size'],
            self.config['hidden_size'],
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
