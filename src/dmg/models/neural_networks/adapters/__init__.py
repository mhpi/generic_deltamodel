"""The adapters all follow the same interface:
def forward(self, hidden_states: torch.Tensor, time_features: torch.Tensor, static_features=None):
where hidden_states is of shape [batch_size, seq_len, d_model],
time_features is of shape [batch_size, seq_len, input_size], and
static_features is of shape [batch_size, n_static_features] (optional).
"""

from .attention_adapter import AttentionAdapter
from .bottleneck_adapter import BottleneckAdapter
from .conv_adapter import ConvAdapter
from .dual_residual_adapter import DualResidualAdapter
from .feedforward_adapter import FeedforwardAdapter
from .gated_adapter import GatedAdapter
from .moe_adapter import MoEAdapter

__all__ = [
    'DualResidualAdapter',
    'GatedAdapter',
    'FeedforwardAdapter',
    'ConvAdapter',
    'AttentionAdapter',
    'BottleneckAdapter',
    'MoEAdapter',
]
