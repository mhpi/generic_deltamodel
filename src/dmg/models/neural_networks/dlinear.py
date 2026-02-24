import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class _MovingAvg(nn.Module):
    """Causal moving average over time with left replication padding."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        assert kernel_size % 2 == 1 and kernel_size >= 1
        self.kernel = kernel_size
        self.norm = 1.0 / kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, N, C]
        T, N, C = x.shape
        pad = self.kernel - 1  # causal: all pad on the left
        left = x[0:1].expand(pad, N, C)
        x_pad = torch.cat([left, x], dim=0)               # [T+pad, N, C]
        x_nct = x_pad.permute(1, 2, 0)                    # [N, C, T+pad]
        w = torch.ones(C, 1, self.kernel, device=x.device, dtype=x.dtype) * self.norm
        # causal average
        trend = F.conv1d(x_nct, w, groups=C)              # [N, C, T]
        return trend.permute(2, 0, 1)                     # [T, N, C]

class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.mavg = _MovingAvg(kernel_size)
    def forward(self, x: torch.Tensor):
        trend = self.mavg(x)              # [T, N, C]
        resid = x - trend                 # [T, N, C]
        return resid, trend

class Dlinear(nn.Module):
    """
    Paper-style DLinear (sliding/causal, no dual-forward).
    Constructor matches your factory: Dlinear(nx, ny, hidden_size, dr).
    Forward takes a tensor only: x: [T, N, F] -> y: [T, N, ny]
    """
    def __init__(self, nx: int, ny: int, hidden_size: int, dr: float):
        super().__init__()
        self.nx = nx
        self.ny = ny

        # ---- IMPORTANT: temporal kernel that plays the role of seq_len ----
        # For strict parity with the paper, set this to your rho (history length).
        # Without factory access to rho, we use a reasonable default and allow env override.
        k_time = int(os.getenv("DLINEAR_KTIME", "25"))
        k_time = max(1, k_time)
        self.k_time = k_time

        self.decomp = _SeriesDecomp(kernel_size=25)
        self.dropout = nn.Dropout(dr) if dr and dr > 0 else nn.Identity()

        # Temporal linear maps (seasonal & trend) as **depthwise causal convs** over time.
        # Each feature/channel has its own temporal kernel (paper’s per-channel linear over time).
        # in: [N, C, T] -> out: [N, C, T] (same length via left padding).
        self.temp_seasonal = nn.Conv1d(self.nx, self.nx, kernel_size=self.k_time, groups=self.nx, bias=True)
        self.temp_trend    = nn.Conv1d(self.nx, self.nx, kernel_size=self.k_time, groups=self.nx, bias=True)

        # Initialize small (paper uses averaging start; small weights work fine)
        nn.init.zeros_(self.temp_seasonal.weight); nn.init.zeros_(self.temp_seasonal.bias)
        nn.init.zeros_(self.temp_trend.weight);    nn.init.zeros_(self.temp_trend.bias)

        # After per-channel temporal mapping, project channels -> ny.
        self.head = nn.Linear(self.nx, self.ny)

    def _causal_pad(self, x_ncT: torch.Tensor) -> torch.Tensor:
        # x_ncT: [N, C, T]; pad left with (k_time-1)
        pad = self.k_time - 1
        if pad <= 0:
            return x_ncT
        left = x_ncT[:, :, :1].expand(-1, -1, pad)
        return torch.cat([left, x_ncT], dim=2)  # [N, C, pad+T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, N, F] where F == nx
        y: [T, N, ny]
        """
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[2] != self.nx:
            raise ValueError(f"Dlinear expected tensor [T,N,{self.nx}], got {tuple(getattr(x, 'shape', []))}")

        resid, trend = self.decomp(x)               # [T, N, F] each
        resid = self.dropout(resid); trend = self.dropout(trend)

        # To [N, C, T]
        r_ncT = resid.permute(1, 2, 0).contiguous()
        t_ncT = trend.permute(1, 2, 0).contiguous()

        # Causal pad and temporal depthwise conv (per-channel time mixing)
        r_ncT = self._causal_pad(r_ncT)
        t_ncT = self._causal_pad(t_ncT)
        y_seas_ncT = self.temp_seasonal(r_ncT)      # [N, C, T]
        y_trnd_ncT = self.temp_trend(t_ncT)         # [N, C, T]

        # Back to [T, N, C]
        y_TNC = (y_seas_ncT + y_trnd_ncT).permute(2, 0, 1).contiguous()

        # Pointwise projection channels -> targets
        y_flat = y_TNC.view(-1, self.nx)            # [T*N, C]
        y_out  = self.head(y_flat).view(x.shape[0], x.shape[1], self.ny)  # [T, N, ny]
        return y_out
