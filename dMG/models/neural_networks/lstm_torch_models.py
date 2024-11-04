"""
Recharacterization of original hydroDL LSTM model with conditional dropout, using
torch.nn.LSTM as a base class. This adds pytorch support and inheritence, as well
as better readability and maintainability.

Model structure adapted from: Dapeng Feng, MHPI

TODO: debug and test against original.
"""
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.neural_networks.dropout import DropMask, createMask



class CudnnLstm(torch.nn.LSTM):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW"):
        super().__init__(input_size=inputSize, hidden_size=hiddenSize, num_layers=1, batch_first=True)
        self.name = 'CudnnLstm'
        self.dr = dr
        self.drMethod = drMethod
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def reset_mask(self):
        """Resets dropout masks for input and hidden weights."""
        if self.dr > 0:
            self.maskW_ih = createMask(self.weight_ih_l0, self.dr)
            self.maskW_hh = createMask(self.weight_hh_l0, self.dr)

    def forward(self, input, hx=None, doDropMC=False, dropoutFalse=False):
        """Custom forward pass with conditional dropout."""
        doDrop = (
            self.training or doDropMC
        ) and not dropoutFalse and self.dr > 0

        if doDrop:
            self.reset_mask()
            weight_ih = DropMask.apply(self.weight_ih_l0, self.maskW_ih, True)
            weight_hh = DropMask.apply(self.weight_hh_l0, self.maskW_hh, True)
        else:
            weight_ih = self.weight_ih_l0
            weight_hh = self.weight_hh_l0

        # Apply the LSTM forward pass with the selected weights
        output, (hn, cn) = super().forward(input, hx=hx)
        
        return output, (hn, cn)


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super().__init__()
        self.name = 'CudnnLstmModel'
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """Forward pass through the linear, LSTM, and output layers."""
        x0 = F.relu(self.linearIn(x))
        outLSTM, _ = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        out = self.linearOut(outLSTM)
        return out
