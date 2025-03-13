from typing import Optional, Tuple

import torch

from dMG.models.neural_networks.ann import AnnModel
from dMG.models.neural_networks.cudnn_lstm import CudnnLstmModel
from dMG.models.neural_networks.lstm import LstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.
    
    Supports GPU and CPU forwarding.
    
    Paramters
    ---------
    nx1 : int
        Number of LSTM input features.
    ny1 : int
        Number of LSTM output features.
    hiddeninv1 : int
        LSTM hidden size.
    nx2 : int
        Number of MLP input features.
    ny2 : int
        Number of MLP output features.
    hiddeninv2 : int
        MLP hidden size.
    dr1 : float, optional
        Dropout rate for LSTM. Default is 0.5.
    dr2 : float, optional
        Dropout rate for MLP. Default is 0.5.
    device : str, optional
        Device to run the model on. Default is 'cpu'.
    """
    def __init__(
        self,
        *,
        nx1: int,
        ny1: int,
        hiddeninv1: int,
        nx2: int,
        ny2: int,
        hiddeninv2: int,
        dr1: Optional[float] = 0.5,
        dr2: Optional[float] = 0.5,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        
        if device == 'cpu':
            # CPU-compatible LSTM model.
            self.lstminv = LstmModel(
                nx=nx1, ny=ny1, hidden_size=hiddeninv1, dr=dr1
            )
        else:
            # GPU-only HydroDL LSTM.
            self.lstminv = CudnnLstmModel(
                nx=nx1, ny=ny1, hidden_size=hiddeninv1, dr=dr1
            )
        
        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z1 : torch.Tensor
            The LSTM input tensor. 
        z2 : torch.Tensor
            The MLP input tensor.
        
        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        lstm_out = self.lstminv(z1)  # dim: timesteps, gages, params

        ann_out = self.ann(z2)

        return [torch.sigmoid(lstm_out), ann_out]
