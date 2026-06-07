from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

USE_KAN = False


class KAN_2(nn.Module):
    """A Kolmogorov-Arnold Neural Network (KAN)."""

    def __init__(
        self,
        *,
        nx,
        ny,
        hiddenSize,
        k=3,
        grid=5,
        num_hidden_layers=2,
        dropout_rate=0.5,
        use_dropout=True,
    ):
        super().__init__()
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
                width=[self.hidden_size, 2 * self.hidden_size + 1, self.hidden_size],
                k=k,
                grid=grid,
            )
            self.layers.append(layer)

        # Initialize parameters in KAN layers
        self.initialize_kan_layers()

    def xavier_normal_initializer(self, x) -> None:
        """Xavier normal initialization."""
        gain = nn.init.calculate_gain('relu')  # Adjust based on activation (ReLU here)
        nn.init.xavier_normal_(x, gain=gain)

    def initialize_kan_layers(self):
        """Apply Xavier initialization to all KAN layers."""
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name or 'coeff' in name:
                    self.xavier_normal_initializer(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, x, y=None):
        """Forward pass of the neural network."""
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
        cache_states: bool = False,
        use_in_proj: bool = True,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        self.use_in_proj = use_in_proj

        if ny1 == 0:
            self.lstm_inv = None
            self.linear_out = None
            self.hiddeninv1 = None
        else:
            lstm_input_size = hiddeninv1 if use_in_proj else nx1
            if use_in_proj:
                self.in_proj = torch.nn.Linear(nx1, hiddeninv1)
            self.lstm_inv = torch.nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hiddeninv1,
                dropout=dr1,
            )
            self.linear_out = torch.nn.Linear(hiddeninv1, ny1)
            self.hiddeninv1 = hiddeninv1

        if USE_KAN:
            self.ann = KAN_2(nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)
        else:
            self.ann = torch.nn.Sequential(
                torch.nn.Linear(nx2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, ny2),
                torch.nn.Sigmoid(),
            )

        # Chunk prediction for distributed modeling
        self.sub_batch_size = sub_batch_size
        self.sub_batch_mode = False

    def set_mode(self, is_simulate: bool):
        """If True, ensables sub-batch mode for CPU forward."""
        self.sub_batch_mode = is_simulate

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
        list
            [lstm_out, h_out, c_out, ann_out]
        """
        lstm_out = torch.empty((z1.shape[0], z1.shape[1], 0), device=z1.device)
        h_out = torch.Tensor([])
        c_out = torch.Tensor([])

        if self.lstm_inv is not None and self.use_in_proj:
            z1 = self.in_proj(z1)

        if self.sub_batch_mode:  # output cpu tensor to save gpu memory
            device = next(self.parameters()).device
            total_size = z2.size(0)
            lstm_out_list = []
            h_out_list = []
            c_out_list = []
            ann_out_list = []
            for start in range(0, total_size, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size)
                if self.lstm_inv is not None:
                    batch_z1 = z1[:, start:end, :].to(device)
                    lstm_out_sub, (h_out_sub, c_out_sub) = self.lstm_inv(batch_z1)
                    lstm_out_sub = torch.sigmoid(self.linear_out(lstm_out_sub))
                    lstm_out_list.append(lstm_out_sub.detach().cpu())
                    h_out_list.append(h_out_sub.detach().cpu())
                    c_out_list.append(c_out_sub.detach().cpu())
                batch_z2 = z2[start:end, :].to(device)
                ann_out_sub = self.ann(batch_z2)
                ann_out_list.append(ann_out_sub.detach().cpu())
            if self.lstm_inv is not None:
                lstm_out = torch.cat(lstm_out_list, dim=1)
                h_out = torch.cat(h_out_list, dim=1)
                c_out = torch.cat(c_out_list, dim=1)
            ann_out = torch.cat(ann_out_list, dim=0)
        else:
            if self.lstm_inv is not None:
                lstm_out, (h_out, c_out) = self.lstm_inv(z1)
                lstm_out = torch.sigmoid(self.linear_out(lstm_out))
            ann_out = self.ann(z2)
        return [lstm_out, h_out, c_out, ann_out]


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
        cache_states: bool = False,
        use_in_proj: bool = True,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlp2Model'
        self.cache_states = cache_states
        self.use_in_proj = use_in_proj
        self.hn, self.cn = None, None

        if ny1 == 0:
            self.lstm_inv = None
            self.linear_out = None
            self.hiddeninv1 = None
        else:
            lstm_input_size = hiddeninv1 if use_in_proj else nx1
            if use_in_proj:
                self.in_proj = torch.nn.Linear(nx1, hiddeninv1)
            self.lstm_inv = torch.nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hiddeninv1,
                dropout=dr1,
            )
            self.linear_out = torch.nn.Linear(hiddeninv1, ny1)
            self.hiddeninv1 = hiddeninv1

        if USE_KAN:
            self.ann1 = KAN_2(nx=nx2, ny=ny2, hiddenSize=hiddeninv2, dropout_rate=dr2)
            self.ann2 = KAN_2(nx=nx3, ny=ny3, hiddenSize=hiddeninv3, dropout_rate=dr3)
        else:
            self.ann1 = torch.nn.Sequential(
                torch.nn.Linear(nx2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, hiddeninv2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr2),
                torch.nn.Linear(hiddeninv2, ny2),
                torch.nn.Sigmoid(),
            )
            self.ann2 = torch.nn.Sequential(
                torch.nn.Linear(nx3, hiddeninv3),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr3),
                torch.nn.Linear(hiddeninv3, hiddeninv3),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr3),
                torch.nn.Linear(hiddeninv3, hiddeninv3),
                torch.nn.ReLU(),
                torch.nn.Dropout(dr3),
                torch.nn.Linear(hiddeninv3, ny3),
                torch.nn.Sigmoid(),
            )

        # Chunk prediction for distributed modeling
        self.sub_batch_size = sub_batch_size
        self.sub_batch_mode = False

    def set_mode(self, is_simulate: bool):
        """If True, ensables sub-batch mode for CPU forward."""
        self.sub_batch_mode = is_simulate

    def reset_states(self):
        """Clear internal LSTM states."""
        self.hn = None
        self.cn = None

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
        reset_state: bool = True,
    ) -> list[torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z1
            The LSTM input tensor.
        z2
            The first MLP input tensor.
        z3
            The second MLP input tensor.
        h0
            Initial hidden state (from LF model via h_transfer). If None,
            falls back to internal cache (when cache_states=True and
            reset_state=False) or a fresh zero state.
        c0
            Initial cell state. Same fallback logic as h0.
        reset_state
            If False, use internally cached states when h0/c0 not provided.

        Returns
        -------
        list
            [lstm_out, ann_out1, ann_out2]
        """
        lstm_out = torch.empty((z1.shape[0], z1.shape[1], 0), device=z1.device)

        if self.lstm_inv is not None and self.use_in_proj:
            z1 = self.in_proj(z1)

        # Resolve initial state
        if h0 is not None and c0 is not None and h0.numel() > 0:
            hx = (h0, c0)
        elif self.cache_states and self.hn is not None and not reset_state:
            hx = (self.hn, self.cn)
        else:
            hx = None

        if self.sub_batch_mode:  # output cpu tensor to save gpu memory
            device = next(self.parameters()).device
            total_size_static_1 = z2.size(0)
            total_size_static_2 = z3.size(0)
            lstm_out_list = []
            ann_out1_list = []
            ann_out2_list = []
            hn_list = []
            cn_list = []
            for start in range(0, total_size_static_1, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size_static_1)
                if self.lstm_inv is not None:
                    batch_z1 = z1[:, start:end, :].to(device)
                    batch_hx = None
                    if hx is not None:
                        batch_hx = (
                            hx[0][:, start:end, :].to(device),
                            hx[1][:, start:end, :].to(device),
                        )
                    lstm_out_sub, (hn_sub, cn_sub) = self.lstm_inv(batch_z1, batch_hx)
                    lstm_out_sub = torch.sigmoid(self.linear_out(lstm_out_sub))
                    lstm_out_list.append(lstm_out_sub.detach().cpu())
                    hn_list.append(hn_sub.detach().cpu())
                    cn_list.append(cn_sub.detach().cpu())
                batch_z2 = z2[start:end, :].to(device)
                ann_out1_list.append(self.ann1(batch_z2).detach().cpu())
            for start in range(0, total_size_static_2, self.sub_batch_size):
                end = min(start + self.sub_batch_size, total_size_static_2)
                batch_z3 = z3[start:end, :].to(device)
                ann_out2_list.append(self.ann2(batch_z3).detach().cpu())
            if self.lstm_inv is not None:
                lstm_out = torch.cat(lstm_out_list, dim=1)
                if self.cache_states:
                    self.hn = torch.cat(hn_list, dim=1)
                    self.cn = torch.cat(cn_list, dim=1)
            ann_out1 = torch.cat(ann_out1_list, dim=0)
            ann_out2 = torch.cat(ann_out2_list, dim=0)
        else:
            if self.lstm_inv is not None:
                lstm_out, (hn_new, cn_new) = self.lstm_inv(z1, hx)
                lstm_out = torch.sigmoid(self.linear_out(lstm_out))
                if self.cache_states:
                    self.hn = hn_new.detach()
                    self.cn = cn_new.detach()
            ann_out1 = self.ann1(z2)
            ann_out2 = self.ann2(z3)
        return [lstm_out, ann_out1, ann_out2]


class StackLstmMlpModel(torch.nn.Module):
    """Stacked LSTM-MLP model for multi-scale learning."""

    def __init__(
        self,
        lstm_mlp: LstmMlpModel,
        lstm_mlp2: LstmMlp2Model,
        use_transfer: bool = True,
    ):
        super().__init__()
        self.lstm_mlp = lstm_mlp
        self.lstm_mlp2 = lstm_mlp2
        self.use_transfer = use_transfer
        self.lof_params_cache = None

        if (
            use_transfer
            and lstm_mlp.lstm_inv is not None
            and lstm_mlp2.lstm_inv is not None
        ):
            self.h_transfer = nn.Linear(lstm_mlp.hiddeninv1, lstm_mlp2.hiddeninv1)
            self.c_transfer = nn.Linear(lstm_mlp.hiddeninv1, lstm_mlp2.hiddeninv1)
            nn.init.zeros_(self.h_transfer.weight)
            nn.init.zeros_(self.h_transfer.bias)
            nn.init.zeros_(self.c_transfer.weight)
            nn.init.zeros_(self.c_transfer.bias)

    def set_mode(self, is_simulate: bool):
        """If True, ensables sub-batch mode for CPU forward."""
        self.lstm_mlp.sub_batch_mode = is_simulate
        self.lstm_mlp2.sub_batch_mode = is_simulate

    def forward(
        self,
        input1: Optional[tuple[torch.Tensor, torch.Tensor]],
        input2: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: bool = True,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        When batch=True: runs the LF model, transfers its final hidden states
        to initialize the HF LSTM, and caches LF params for step-wise use.
        When batch=False: skips the LF model and uses cached params + HF
        internal state (requires cache_states=True on lstm_mlp2).
        """
        if batch and input1 is not None:
            lstm_out_1, h_out_1, c_out_1, ann_out_1 = self.lstm_mlp(*input1)
            self.lof_params_cache = [lstm_out_1[-1:].detach(), ann_out_1.detach()]

            h_out_init = torch.tensor([])
            c_out_init = torch.tensor([])
            if (
                self.use_transfer
                and self.lstm_mlp.lstm_inv is not None
                and self.lstm_mlp2.lstm_inv is not None
            ):
                h_out_init = self.h_transfer(h_out_1)
                c_out_init = self.c_transfer(c_out_1)

            out2 = self.lstm_mlp2(
                *input2, h0=h_out_init, c0=c_out_init, reset_state=True
            )
            return [lstm_out_1, ann_out_1], out2
        else:
            out2 = self.lstm_mlp2(*input2, reset_state=False)
            return self.lof_params_cache, out2
