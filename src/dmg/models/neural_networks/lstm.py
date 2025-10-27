import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from dmg.core.calc.dropout import DropMask, createMask


class Lstm(torch.nn.Module):
    """LSTM using torch LSTM (GPU + CPU support).

    This replaces the HydroDL `CudnnLstm`, which uses CPU-incompatible
    torch cudnn_rnn backends.

    Parameters
    ----------
    nx
        Number of input features.
    hidden_size
        Number of hidden units.
    dr
        Dropout rate. Default is 0.5.

    NOTE: Not validated for training.
    """

    def __init__(
        self,
        nx: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
    ) -> None:
        super().__init__()
        self.name = 'Lstm'
        self.nx = nx
        self.hidden_size = hidden_size
        self.dr = dr

        # Initialize new torch LSTM; disable dropout (it's handled manually).
        self.lstm = torch.nn.LSTM(
            input_size=self.nx,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=False,
            bias=True,
            dropout=0,
            bidirectional=False,
        )
        # Remove default parameters. These are manually assigned in forward().
        delattr(self.lstm, 'weight_ih_l0')
        delattr(self.lstm, 'weight_hh_l0')
        delattr(self.lstm, 'bias_ih_l0')
        delattr(self.lstm, 'bias_hh_l0')

        # Name parameters to match CudannLstm.
        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, nx))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

        self._init_mask()
        self._init_parameters()

    def __setstate__(self, d: dict) -> None:
        super().__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def _init_mask(self):
        """Initialize dropout mask."""
        with torch.no_grad():
            self.mask_w_ih = createMask(self.w_ih, self.dr)
            self.mask_w_hh = createMask(self.w_hh, self.dr)

    def _init_parameters(self):
        """Initialize parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            if param.requires_grad:
                param.data.uniform_(-stdv, stdv)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
        cx: Optional[torch.Tensor] = None,
        do_drop_mc: bool = False,
        dr_false: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        input
            The input tensor.
        hx
            Hidden state tensor.
        cx
            Cell state tensor.
        do_drop_mc
            Flag for applying dropout.
        dr_false
            Flag for applying dropout.
        """
        # Ensure do_drop is False, unless do_drop_mc is True.
        if dr_false and (not do_drop_mc):
            do_drop = False
        elif self.dr > 0 and (do_drop_mc is True or self.training is True):
            do_drop = True
        else:
            do_drop = False

        batch_size = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1,
                batch_size,
                self.hidden_size,
                requires_grad=False,
            )
        if cx is None:
            cx = input.new_zeros(
                1,
                batch_size,
                self.hidden_size,
                requires_grad=False,
            )

        if do_drop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # Manually assign parameters to torch LSTM.
        self.lstm.weight_ih_l0 = torch.nn.Parameter(weight[0])
        self.lstm.weight_hh_l0 = torch.nn.Parameter(weight[1])
        self.lstm.bias_ih_l0 = torch.nn.Parameter(weight[2])
        self.lstm.bias_hh_l0 = torch.nn.Parameter(weight[3])
        output, (hy, cy) = self.lstm(input, (hx, cx))

        return output, (hy, cy)

    @property
    def all_weights(self):
        """Return all weights."""
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class LstmModel(torch.nn.Module):
    """LSTM model using torch LSTM (GPU + CPU support).

    This replaces `CudnnLstmModel`, which uses torch cudnn_rnn backends with no
    CPU support.

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
    cache_states
        Whether to cache hidden and cell states.

    NOTE: Not validated for training.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
        cache_states: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.name = 'LstmModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dr = dr
        self.cache_states = cache_states

        self.hn, self._hn_cache = None, None  # hidden state
        self.cn, self._cn_cache = None, None  # cell state

        self.linear_in = torch.nn.Linear(nx, hidden_size)
        self.lstm = Lstm(nx=hidden_size, hidden_size=hidden_size, dr=dr)
        self.linear_out = torch.nn.Linear(hidden_size, ny)

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

        device = next(self.parameters()).device
        self.hn = states[0].detach().to(device)
        self.cn = states[1].detach().to(device)

    def forward(
        self,
        x,
        do_drop_mc: Optional[bool] = False,
        dr_false: Optional[bool] = False,
    ) -> torch.Tensor:
        """Forward pass.

        NOTE (caching): Hidden states are always cached so that they can be
        accessed by `get_states`, but they are only available to the LSTM if
        `cache_states` is set to True.

        Parameters
        ----------
        x
            The input tensor.
        do_drop_mc
            Flag for applying mc dropout.
        dr_false
            Flag for applying dropout.
        """
        x0 = F.relu(self.linear_in(x))
        lstm_out, (hn, cn) = self.lstm(
            x0,
            self.hn,
            self.cn,
            do_drop_mc=do_drop_mc,
            dr_false=dr_false,
        )

        self._hn_cache = hn.detach().cpu()
        self._cn_cache = cn.detach().cpu()

        if self.cache_states:
            self.hn = self._hn_cache.to(x.device)
            self.cn = self._cn_cache.to(x.device)

        return self.linear_out(lstm_out)
