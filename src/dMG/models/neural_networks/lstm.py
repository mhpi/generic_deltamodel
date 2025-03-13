import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from dMG.core.calc.dropout import DropMask, createMask


class Lstm(torch.nn.Module):
    """Custom LSTM model using new PyTorch implementation.

    Supports GPU and CPU.
    
    This replaces `CudnnLstm`, which used PyTorch rnn backends with no
    CPU support.

    NOTE: Only mirrors inference functionality of `CudnnLstm`. Not validated
    for training.
    """
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW'):
        super().__init__()
        self.name = 'CudnnLstm'
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        # Initialize new torch LSTM; disable dropout (it's handled manually).
        self.lstm = torch.nn.LSTM(
            input_size=self.inputSize,
            hidden_size=self.hiddenSize,
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
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        
        self.reset_mask()
        self.reset_parameters()

    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        with torch.no_grad():
            self.maskW_ih = createMask(self.w_ih, self.dr)
            self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for param in self.parameters():
            if param.requires_grad:
                param.data.uniform_(-stdv, stdv)

    def flatten_parameters(self):
        """This method does nothing, just to bypass non-contiguous memory warning."""
        pass

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        self.device = input.device  # TODO: handle this better -- needs to be an argument in def.

        # Ensure doDrop is False, unless doDropMC is True.
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = torch.zeros(1, batchSize, self.hiddenSize, device=self.device, requires_grad=False)
        if cx is None:
            cx = torch.zeros(1, batchSize, self.hiddenSize, device=self.device, requires_grad=False)

        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        self.lstm.to(self.device)
        
        # Manually assign parameters to torch LSTM.
        self.lstm.weight_ih_l0 = torch.nn.Parameter(weight[0])
        self.lstm.weight_hh_l0 = torch.nn.Parameter(weight[1])
        self.lstm.bias_ih_l0 = torch.nn.Parameter(weight[2])
        self.lstm.bias_hh_l0 = torch.nn.Parameter(weight[3])
        output, (hy, cy) = self.lstm(input, (hx, cx))

        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class LstmModel(torch.nn.Module):
    """Custom LSTM model using new PyTorch implementation.
    
    Supports GPU and CPU.
    
    This replaces `CudnnLstmModel`, which used PyTorch rnn backends with no
    CPU support.

    NOTE: Only mirrors inference functionality of `CudnnLstm`. Not validated
    for training.
    """
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super().__init__()
        self.name = 'CudnnLstmModel'
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1

        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = Lstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))        
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(outLSTM)
        return out
    