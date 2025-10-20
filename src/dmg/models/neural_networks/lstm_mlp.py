from typing import Optional

import torch

from dmg.models.neural_networks.ann import AnnModel
from dmg.models.neural_networks.cudnn_lstm import CudnnLstmModel
from dmg.models.neural_networks.lstm import LstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.

    Supports GPU and CPU forwarding.

    Parameters
    ----------
    nx1
        Number of LSTM input features.
    ny1
        Number of LSTM output features.
    hiddeninv1
        LSTM hidden size.
    nx2
        Number of MLP input features.
    ny2
        Number of MLP output features.
    hiddeninv2
        MLP hidden size.
    dr1
        Dropout rate for LSTM. Default is 0.5.
    dr2
        Dropout rate for MLP. Default is 0.5.
    device
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
            # CPU-compatible PyTorch LSTM.
            self.lstminv = LstmModel(
                nx=nx1,
                ny=ny1,
                hidden_size=hiddeninv1,
                dr=dr1,
            )
        else:
            # GPU-only HydroDL LSTM.
            self.lstminv = CudnnLstmModel(
                nx=nx1,
                ny=ny1,
                hidden_size=hiddeninv1,
                dr=dr1,
            )

        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

        # LSTM states
        self.hn = None
        self.cn = None

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x1
            The LSTM input tensor.
        x2
            The MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        self.lstminv.load_states((self.hn, self.cn))
        lstm_out = self.lstminv(x1)
        self.hn, self.cn = self.lstminv.get_states()

        ann_out = self.ann(x2)

        return [torch.sigmoid(lstm_out), ann_out]

    def get_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden and cell states."""
        return self.hn, self.cn

    def load_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Load hidden and cell states."""
        self.hn, self.cn = states
