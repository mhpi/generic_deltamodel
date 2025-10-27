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
    cache_states
        Whether to cache hidden and cell states for LSTM.
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
        cache_states: Optional[bool] = False,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        self.nx1 = nx1
        self.ny1 = ny1
        self.hiddeninv1 = hiddeninv1
        self.nx2 = nx2
        self.ny2 = ny2
        self.hiddeninv2 = hiddeninv2
        self.dr1 = dr1
        self.dr2 = dr2
        self.cache_states = cache_states
        self.device = device

        self.hn, self._hn_cache = None, None  # hidden state
        self.cn, self._cn_cache = None, None  # cell state

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

        self.activation = torch.nn.Sigmoid()

        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    def get_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden and cell states."""
        return self._hn_cache, self._cn_cache

    def load_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Load hidden and cell states."""
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        if not (isinstance(states, tuple) and len(states) == 2):
            raise ValueError("`states` must be a tuple of 2 tensors.")

        self.hn = states[0].detach()
        self.cn = states[1].detach()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        NOTE (caching): Hidden states are always cached so that they can be
        accessed by `get_states`, but they are only available to the LSTM if
        `cache_states` is set to True.

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
        act_out = self.activation(lstm_out)
        ann_out = self.ann(x2)

        self._hn_cache, self._cn_cache = self.lstminv.get_states()

        if self.cache_states:
            self.hn = self._hn_cache.to(x1.device)
            self.cn = self._cn_cache.to(x1.device)

        return (act_out, ann_out)
