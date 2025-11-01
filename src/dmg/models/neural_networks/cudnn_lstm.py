import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from dmg.core.calc.dropout import DropMask, createMask

# ------------------------------------------#
# NOTE: Suppress this warning until we can implement a proper pytorch nn.LSTM.
warnings.filterwarnings(
    "ignore", message=".*weights are not part of single contiguous chunk.*"
)
# ------------------------------------------#


class CudnnLstm(torch.nn.Module):
    """HydroDL LSTM model using cuDNN backend (GPU only).

    Modification of original LSTM developed by Dapeng Feng, et al. (2022) for
    HydroDL.

    Parameters
    ----------
    nx
        Number of input features.
    hidden_size
        Number of hidden units.
    dr
        Dropout rate.
    """

    def __init__(
        self,
        nx: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
    ) -> None:
        super().__init__()
        self.name = 'CudnnLstm'
        self.nx = nx
        self.hidden_size = hidden_size
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hidden_size * 4, nx))
        self.w_hh = Parameter(torch.Tensor(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.Tensor(hidden_size * 4))
        self.b_hh = Parameter(torch.Tensor(hidden_size * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

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
        self.mask_w_ih = createMask(self.w_ih, self.dr)
        self.mask_w_hh = createMask(self.w_hh, self.dr)

    def _init_parameters(self):
        """Initialize parameters."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

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

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hidden_size, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hidden_size, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if do_drop is True:
            self._init_mask()
            weight = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hidden_size,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        """Return all weights."""
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class CudnnLstmModel(torch.nn.Module):
    """HydroDL LSTM model using torch cudnn_rnn backend (GPU only).

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
    dpl
        Flag for applying sigmoid activation to output. This is necessary if the
        model is used for differentiable parameter learning.
    cache_states
        Whether to cache hidden and cell states.
    """

    def __init__(
        self,
        *nx: int,
        ny: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
        dpl: Optional[bool] = False,
        cache_states: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.name = 'CudnnLstmModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dr = dr
        self.dpl = dpl
        self.cache_states = cache_states

        self.hn, self._hn_cache = None, None  # hidden state
        self.cn, self._cn_cache = None, None  # cell state

        self.linear_in = torch.nn.Linear(nx, hidden_size)
        self.lstm = CudnnLstm(nx=hidden_size, hidden_size=hidden_size, dr=dr)
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

        out = self.linear_out(lstm_out)
        if self.dpl:
            return torch.sigmoid(out)
        else:
            return out
