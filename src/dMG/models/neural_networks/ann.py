import torch
from typing import Optional
import torch.nn.functional as F
from torch.nn import Dropout, Linear


class AnnModel(torch.nn.Module):
    """Artificial neural network (ANN) model.
    
    Parameters
    ----------
    nx : int
        Number of input features.
    ny : int
        Number of output features.
    hidden_size : int
        Number of hidden units.
    dr : float, optional
        Dropout rate. Default is 0.5.
    """
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        dr: Optional[float] = 0.5,
    ) -> None:
        super().__init__()
        self.name = 'AnnModel'
        self.hidden_size = hidden_size
        self.i2h = Linear(nx, hidden_size)
        self.h2h1 = Linear(hidden_size, hidden_size, bias=True)
        self.h2h2 = Linear(hidden_size, hidden_size, bias=True)
        self.h2h3 = Linear(hidden_size, hidden_size, bias=True)
        self.h2h4 = Linear(hidden_size, hidden_size, bias=True)
        self.h2h5 = Linear(hidden_size, hidden_size, bias=True)
        self.h2h6 = Linear(hidden_size, hidden_size, bias=True)
        self.h2o = Linear(hidden_size, ny)
        self.dropout = Dropout(dr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Assuming x is already in appropriate batch form.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        ht = F.relu(self.i2h(x))
        ht = self.dropout(ht)  # Apply dropout after each hidden layer activation.
        
        ht1 = F.relu(self.h2h1(ht))
        ht1 = self.dropout(ht1)
        
        ht2 = F.relu(self.h2h2(ht1))
        ht2 = self.dropout(ht2)
        
        ht3 = F.relu(self.h2h3(ht2))
        ht3 = self.dropout(ht3)
        
        ht4 = F.relu(self.h2h4(ht3))
        ht4 = self.dropout(ht4)
        
        ht5 = F.relu(self.h2h5(ht4))
        ht5 = self.dropout(ht5)
        
        ht6 = F.relu(self.h2h6(ht5))
        ht6 = self.dropout(ht6)

        # Using sigmoid for binary classification or a logistic outcome.
        return torch.sigmoid(self.h2o(ht6))


class AnnCloseModel(torch.nn.Module):
    """Artificial neural network (ANN) model with close observations.

    Parameters
    ----------
    nx : int
        Number of input features.
    ny : int
        Number of output features.
    hidden_size : int
        Number of hidden units.
    fill_obs : bool, optional
        Whether to fill observations. Default is True.
    """
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int,
        fill_obs: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.name = 'AnnCloseModel'
        self.ny = ny
        self.hidden_size = hidden_size
        self.fill_obs = fill_obs

        self.i2h = Linear(nx + 1, hidden_size)
        self.h2h = Linear(hidden_size, hidden_size)
        self.h2o = Linear(hidden_size, ny)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Assuming x is already in appropriate batch form.
        y : torch.Tensor, optional
            Observations tensor. Default is None.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1).cuda()
        out = torch.zeros(nt, ngrid, self.ny).cuda()
        for t in range(nt):
            if self.fill_obs is True:
                yt_obs = y[t, :, :] 
                mask = yt_obs == yt_obs
                yt[mask] = yt_obs[mask]
            
            xt = torch.cat((x[t, :, :], yt), 1)
            ht = F.relu(self.i2h(xt))
            ht2 = self.h2h(ht)
            yt = self.h2o(ht2)
            out[t, :, :] = yt
        return out
