import torch
from torch.nn import Linear, Dropout
import torch.nn.functional as F



class AnnModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dropout_rate=0.5):
        super(AnnModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = Linear(nx, hiddenSize)
        self.h2h1 = Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h2 = Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h3 = Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h4 = Linear(hiddenSize, hiddenSize, bias=True)
        self.h2h5 = Linear(hiddenSize, hiddenSize, bias=True)  # New layer 5
        self.h2h6 = Linear(hiddenSize, hiddenSize, bias=True)  # New layer 6
        self.h2o = Linear(hiddenSize, ny)
        self.dropout = Dropout(dropout_rate)

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


class AnnCloseModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, fillObs=True):
        super(AnnCloseModel, self).__init__()
        self.hiddenSize = hiddenSize
        self.i2h = Linear(nx + 1, hiddenSize)
        self.h2h = Linear(hiddenSize, hiddenSize)
        self.h2o = Linear(hiddenSize, ny)
        self.fillObs = fillObs
        self.ny = ny

    def forward(self, x, y=None):
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs = y[t, :, :]
                mask = ytObs == ytObs
                yt[mask] = ytObs[mask]
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out
    