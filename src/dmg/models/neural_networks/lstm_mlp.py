from typing import Any, Optional

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

        if torch.device(device).type == 'cpu':
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
            if state and not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        if not (isinstance(states, tuple) and len(states) == 2):
            raise ValueError("`states` must be a tuple of 2 tensors.")

        device = next(self.parameters()).device
        self.hn = states[0].detach().to(device)
        self.cn = states[1].detach().to(device)

    def get_runtime_state(self) -> dict[str, Any]:
        """Snapshot the LSTM state stepwise inference carries between calls.

        With 'cache_states' on, the LSTM starts each forward pass from the
        hidden and cell states its previous pass ended on. The MLP output
        depends only on the static inputs, so it carries no state.

        Returns
        -------
        dict[str, Any]
            {'hn': ..., 'cn': ...}, references to the LSTM's
            cached states, each [1, ngrid, hidden_size], or None before the
            first forward pass.
        """
        return {'hn': self.lstminv.hn, 'cn': self.lstminv.cn}

    def load_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by 'get_runtime_state'.

        Parameters
        ----------
        state
            Snapshot returned by 'get_runtime_state'.
        """
        if set(state) != {'hn', 'cn'}:
            raise ValueError(
                f"Runtime state keys {sorted(state)} do not match the expected "
                "keys ['cn', 'hn'].",
            )
        if (state['hn'] is None) != (state['cn'] is None):
            raise ValueError("Runtime states 'hn' and 'cn' must both be set or None.")
        device = next(self.parameters()).device

        if state['hn'] is None:
            hn_cache = cn_cache = hn = cn = None
        else:
            # Mirror what a forward pass leaves behind: CPU copies in the
            # caches read by 'get_states', device copies used by the LSTM.
            hn_cache = state['hn'].detach().cpu()
            cn_cache = state['cn'].detach().cpu()
            hn = hn_cache.to(device)
            cn = cn_cache.to(device)

        for module in (self.lstminv, self):
            module._hn_cache, module._cn_cache = hn_cache, cn_cache
            module.hn, module.cn = hn, cn

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
        lstm_out = self.lstminv(x1)
        act_out = self.activation(lstm_out)
        ann_out = self.ann(x2)

        if self.cache_states:
            self._hn_cache, self._cn_cache = self.lstminv.get_states()
            self.hn = self._hn_cache.to(x1.device)
            self.cn = self._cn_cache.to(x1.device)

        return (act_out, ann_out)
