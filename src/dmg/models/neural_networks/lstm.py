from typing import Optional

import torch
import torch._VF as _VF
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

        # Name parameters to match CudnnLstm.
        self.w_ih = Parameter(torch.empty(hidden_size * 4, nx))
        self.w_hh = Parameter(torch.empty(hidden_size * 4, hidden_size))
        self.b_ih = Parameter(torch.empty(hidden_size * 4))
        self.b_hh = Parameter(torch.empty(hidden_size * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

        if torch.cuda.is_available():
            self.cuda()  # Ensures weight initialization matches CudnnLstm.

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
        stdv = 1.0 / self.hidden_size**0.5
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

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
        self.w_ih.contiguous()
        self.w_hh.contiguous()
        self.b_ih.contiguous()
        self.b_hh.contiguous()

        # Determine if dropout should be applied
        if dr_false and (not do_drop_mc):
            do_drop = False
        elif self.dr > 0 and (do_drop_mc is True or self.training is True):
            do_drop = True
        else:
            do_drop = False

        batch_size = input.size(1)

        if hx is None:
            hx = torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
                requires_grad=False,
            )
        if cx is None:
            cx = torch.zeros(
                1,
                batch_size,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
                requires_grad=False,
            )

        # Apply dropout mask if needed
        if do_drop is True:
            self._init_mask()
            weight_list = [
                DropMask.apply(self.w_ih, self.mask_w_ih, True),
                DropMask.apply(self.w_hh, self.mask_w_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight_list = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        output, hy, cy = _VF.lstm(
            input=input,
            hx=(hx, cx),
            params=weight_list,
            has_biases=True,
            num_layers=1,
            dropout=0.0,
            train=self.training,
            bidirectional=False,
            batch_first=False,
        )
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
    dpl
        Flag for applying sigmoid activation to output. This is necessary if the
        model is used for differentiable parameter learning.
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
        dpl: Optional[bool] = False,
        cache_states: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.name = 'LstmModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size
        self.dpl = dpl
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

        out = self.linear_out(lstm_out)
        if self.dpl:
            return torch.sigmoid(out)
        else:
            return out
