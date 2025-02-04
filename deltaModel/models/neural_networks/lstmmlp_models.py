import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neural_networks.dropout import DropMask, createMask
from torch.nn import Parameter

class LSTMMLP(torch.nn.Module):

    def __init__(self, *, nx1, ny1,hiddeninv1,nx2, ny2,hiddeninv2,dr1=0.5,dr2=0.5):
        super().__init__()
        self.lstminv = CudnnLstmModel(
            nx=nx1, ny=ny1, hiddenSize=hiddeninv1, dr=dr1)
        
        self.Ann = AnnModel(
            nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)


    def forward(self, z1,z2):
        Lstm_para = self.lstminv(z1) # dim: Time, Gage, Para

        Ann_para = self.Ann(z2)


        return [ torch.sigmoid(Lstm_para),Ann_para]

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize) 
        self.lstm = CudnnLstm(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)
        return out




class CudnnLstm(torch.nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod='drW',
                 gpu=0):
        super(CudnnLstm, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]
        self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLstm, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLstm, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [['w_ih', 'w_hh', 'b_ih', 'b_hh']]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(
                1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True), self.b_ih,
                self.b_hh
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 1, False, 0, self.training, False, (), None)
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input, weight, 4, None, hx, cx, 2,  # 2 means LSTM
                self.hiddenSize, 0, 1, False, 0, self.training, False, (), None)
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights]
                for weights in self._all_weights]
    


class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dropout_rate=0.5):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = nn.Linear(nx, hiddenSize)
        self.h2h1 = nn.Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h2 = nn.Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h3 = nn.Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h4 = nn.Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h5 = nn.Linear(hiddenSize, hiddenSize, bias=True)  # New layer 5
        self.h2h6 = nn.Linear(hiddenSize, hiddenSize, bias=True)  # New layer 6
        self.h2o = nn.Linear(hiddenSize, ny)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, y=None):
        # Assuming x is already in appropriate batch form
        ht = F.relu(self.i2h(x))
        ht = self.dropout(ht)  # Apply dropout after the first hidden layer activation
        
        ht1 = F.relu(self.h2h1(ht))
        ht1 = self.dropout(ht1)  # Apply dropout after the second hidden layer activation
        
        ht2 = F.relu(self.h2h2(ht1))
        ht2 = self.dropout(ht2)  # Apply dropout after the third hidden layer activation
        
        ht3 = F.relu(self.h2h3(ht2))
        ht3 = self.dropout(ht3)  # Apply dropout after the fourth hidden layer activation
        
        ht4 = F.relu(self.h2h4(ht3))
        ht4 = self.dropout(ht4)  # Apply dropout after the fifth hidden layer activation
        
        ht5 = F.relu(self.h2h5(ht4))
        ht5 = self.dropout(ht5)  # Apply dropout after the sixth hidden layer activation
        
        ht6 = F.relu(self.h2h6(ht5))
        ht6 = self.dropout(ht6)  # Optionally, apply dropout before the output layer

        yt = torch.sigmoid(self.h2o(ht6))  # Using sigmoid for binary classification or a logistic outcome
        return yt