
import torch
import torch.nn.functional as F
from typing import Optional


class Linear(torch.nn.Module):
    """Linear regression model that matches the HydroDL LSTM model interface.

    Parameters
    ----------
    nx
        Number of input features.
    ny
        Number of output features.
    hidden_size
        Number of hidden units (kept for interface compatibility, but not used).
    dr
        Dropout rate (kept for interface compatibility, but not used).
    """
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: Optional[int] = None,  # Keep for compatibility
        dr: Optional[float] = None,  # Keep for compatibility
    ) -> None:
        super().__init__()
        self.name = 'LinearRegressionModel'
        self.nx = nx
        self.ny = ny
        self.hidden_size = hidden_size  # Store but don't use
        self.ct = 0
        self.n_layers = 1

        # Simple linear transformation from input to output
        self.linear = torch.nn.Linear(nx, ny)

    def forward(
        self,
        x: torch.Tensor,
        do_drop_mc: Optional[bool] = False,  # Keep for interface compatibility
        dr_false: Optional[bool] = False,    # Keep for interface compatibility
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x
            The input tensor of shape (seq_len, batch_size, nx).
        do_drop_mc
            Flag for applying dropout (ignored for linear regression).
        dr_false
            Flag for applying dropout (ignored for linear regression).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (seq_len, batch_size, ny).
        """
        # Apply linear transformation to each time step
        # x is expected to be (seq_len, batch_size, nx)
        seq_len, batch_size, nx = x.shape
        
        # Reshape to (seq_len * batch_size, nx) for linear layer
        x_reshaped = x.view(-1, nx)
        
        # Apply linear transformation
        output_reshaped = self.linear(x_reshaped)
        
        # Reshape back to (seq_len, batch_size, ny)
        output = output_reshaped.view(seq_len, batch_size, self.ny)
        
        return output
