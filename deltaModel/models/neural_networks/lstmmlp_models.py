import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neural_networks.lstm_models import CudnnLstm, CudnnLstmModel


class LSTMMLP(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning."""
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
