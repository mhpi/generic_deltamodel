
import torch
from typing import Optional
from kan import KAN
import torch.nn as nn
import torch.nn.functional as F


class KAN_2(nn.Module):
    """A Kolmogorov-Arnold Neural Network (KAN)"""

    def __init__(self, *, nx, ny, hiddenSize, k=3, grid=5, num_hidden_layers=2, dropout_rate=0.5, use_dropout=True):
        super(KAN_2, self).__init__()
        self.input_size = nx
        self.hidden_size = hiddenSize
        self.num_hidden_layers = num_hidden_layers
        self.use_dropout = use_dropout

        self.layers = nn.ModuleList()

        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

        # Input and output linear layers
        self.input = nn.Linear(self.input_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, ny, bias=False)

        # Initialize input/output layers
        self.xavier_normal_initializer(self.input.weight)
        self.xavier_normal_initializer(self.output.weight)
        nn.init.zeros_(self.input.bias)

        # Add hidden KAN layers
        for _ in range(self.num_hidden_layers):
            layer = KAN(
                width=[self.hidden_size,2*self.hidden_size+1,self.hidden_size],
                k=k,
                grid=grid
            )
            self.layers.append(layer)

        # Initialize parameters in KAN layers
        self.initialize_kan_layers()

    def xavier_normal_initializer(self, x) -> None:
        """Xavier normal initialization"""
        gain = nn.init.calculate_gain('relu')  # Adjust based on activation (ReLU here)
        nn.init.xavier_normal_(x, gain=gain)

    def initialize_kan_layers(self):
        """Apply Xavier initialization to all KAN layers"""
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name or 'coeff' in name:
                    self.xavier_normal_initializer(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x, y=None):
        """Forward pass of the neural network"""
        _x = F.relu(self.input(x))

        for layer in self.layers:
            _x = layer(_x)

        if self.use_dropout:
            _x = self.dropout(_x)

        _x = self.output(_x)
        y = torch.sigmoid(_x)
        return y



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
        sub_batch_size: Optional[int] = 500,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'

        self.lstm_inv = torch.nn.LSTM(
            input_size=nx1,
            hidden_size=hiddeninv1,
            dropout=dr1
        )
        self.linear_out = torch.nn.Linear(hiddeninv1, ny1)

        self.ann = torch.nn.Sequential(
            torch.nn.Linear(nx2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, ny2), torch.nn.Sigmoid()
        )
        # self.ann = KAN_2(nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)

        # chunk prediction for distributed modeling
        self.sub_batch_size = sub_batch_size
        self.sub_batch_mode = False

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z1
            The LSTM input tensor.
        z2
            The MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        if self.sub_batch_mode:  # output cpu tensor to save gpu memory
            device = next(self.parameters()).device
            total_size = z2.size(0)
            lstm_out_list = []
            ann_out_list = []
            for start in range(0, total_size, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size)
                batch_z1 = z1[:, start:end, :].to(device)
                batch_z2 = z2[start:end, :].to(device)
                lstm_out_sub, (_, _) = self.lstm_inv(batch_z1)
                lstm_out_sub = torch.sigmoid(self.linear_out(lstm_out_sub))
                lstm_out_list.append(lstm_out_sub.detach().cpu())
                ann_out_sub = self.ann(batch_z2)
                ann_out_list.append(ann_out_sub.detach().cpu())
            lstm_out = torch.cat(lstm_out_list, dim=1)
            ann_out = torch.cat(ann_out_list, dim=0)
        else:
            lstm_out, (_, _) = self.lstm_inv(z1)  # dim: timesteps, units, params
            lstm_out = torch.sigmoid(self.linear_out(lstm_out))
            ann_out = self.ann(z2)
        return [lstm_out, ann_out]


class LstmMlp2Model(torch.nn.Module):
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
    nx3
        Number of second MLP input features.
    ny3
        Number of second MLP output features.
    hiddeninv3
        Second MLP hidden size.
    dr1
        Dropout rate for LSTM. Default is 0.5.
    dr2
        Dropout rate for MLP. Default is 0.5.
    dr3
        Dropout rate for second MLP. Default is 0.5.
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
            nx3: int,
            ny3: int,
            hiddeninv3: int,
            dr1: Optional[float] = 0.5,
            dr2: Optional[float] = 0.5,
            dr3: Optional[float] = 0.5,
            sub_batch_size: Optional[int] = 500,
            device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'

        self.lstm_inv = torch.nn.LSTM(
            input_size=nx1,
            hidden_size=hiddeninv1,
            dropout=dr1
        )
        self.linear_out = torch.nn.Linear(hiddeninv1, ny1)

        self.ann1 = torch.nn.Sequential(
            torch.nn.Linear(nx2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, hiddeninv2), torch.nn.ReLU(), torch.nn.Dropout(dr2),
            torch.nn.Linear(hiddeninv2, ny2), torch.nn.Sigmoid()
        )
        self.ann2 = torch.nn.Sequential(
            torch.nn.Linear(nx3, hiddeninv3), torch.nn.ReLU(), torch.nn.Dropout(dr3),
            torch.nn.Linear(hiddeninv3, hiddeninv3), torch.nn.ReLU(), torch.nn.Dropout(dr3),
            torch.nn.Linear(hiddeninv3, hiddeninv3), torch.nn.ReLU(), torch.nn.Dropout(dr3),
            torch.nn.Linear(hiddeninv3, ny3), torch.nn.Sigmoid()
        )
        # self.ann1 = KAN_2(nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)
        # self.ann2 = KAN_2(nx=nx3, ny=ny3, hiddenSize=hiddeninv3, dropout_rate=dr3)

        # chunk prediction for distributed modeling
        self.sub_batch_size = sub_batch_size
        self.sub_batch_mode = False

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
            z3: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z1
            The LSTM input tensor.
        z2
            The MLP input tensor.
        z3
            The second MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        if self.sub_batch_mode:  # output cpu tensor to save gpu memory
            device = next(self.parameters()).device
            total_size_static_1 = z2.size(0)
            total_size_static_2 = z3.size(0)
            lstm_out_list = []
            ann_out1_list = []
            ann_out2_list = []
            for start in range(0, total_size_static_1, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size_static_1)
                batch_z1 = z1[:, start:end, :].to(device)
                batch_z2 = z2[start:end, :].to(device)
                lstm_out_sub, (_, _) = self.lstm_inv(batch_z1)
                lstm_out_sub = torch.sigmoid(self.linear_out(lstm_out_sub))
                lstm_out_list.append(lstm_out_sub.detach().cpu())
                ann_out1_sub = self.ann1(batch_z2)
                ann_out1_list.append(ann_out1_sub.detach().cpu())
            for start in range(0, total_size_static_2, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size_static_2)
                batch_z3 = z3[start:end, :].to(device)
                ann_out2_sub = self.ann2(batch_z3)
                ann_out2_list.append(ann_out2_sub.detach().cpu())
            lstm_out = torch.cat(lstm_out_list, dim=1)
            ann_out1 = torch.cat(ann_out1_list, dim=0)
            ann_out2 = torch.cat(ann_out2_list, dim=0)
        else:
            lstm_out, (_, _) = self.lstm_inv(z1)  # dim: timesteps, gages, params
            lstm_out = torch.sigmoid(self.linear_out(lstm_out))
            ann_out1 = self.ann1(z2)
            ann_out2 = self.ann2(z3)
        return [lstm_out, ann_out1, ann_out2]


class StackLstmMlpModel(torch.nn.Module):

    def __init__(self, lstm_mlp: LstmMlpModel, lstm_mlp2: LstmMlp2Model):
        super(StackLstmMlpModel, self).__init__()
        self.lstm_mlp = lstm_mlp
        self.lstm_mlp2 = lstm_mlp2

    def set_mode(self, is_simulate: bool):
        if is_simulate:
            self.lstm_mlp.sub_batch_mode = True
            self.lstm_mlp2.sub_batch_mode = True
        else:
            self.lstm_mlp.sub_batch_mode = False
            self.lstm_mlp2.sub_batch_mode = False

    def forward(self,
                input1: tuple[torch.Tensor, torch.Tensor],
                input2: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        input1 and input2 can be in cpu for sub_batch_mode
        """
        return self.lstm_mlp(*input1), self.lstm_mlp2(*input2)
