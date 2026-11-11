"""Shared adapter construction/dispatch for fine-tuning models that sit on top
of a foundation-model hidden representation (DirectFinetuneing, EmbeddingFinetuneing).

Both models bolt an adapter onto a [B, T, d_model] hidden stream before
decoding; this module is the single place that knows how to build each
adapter type and how to call it, so the two models don't drift out of sync.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Adapter types whose forward signature is (hidden, time_features, static_features).
_STANDARD_ADAPTERS = (
    'gated',
    'feedforward',
    'conv',
    'attention',
    'bottleneck',
    'moe',
    'dual_residual',
)


def build_adapter(
    adapter_type: str,
    d_model: int,
    n_time: int,
    n_static: int,
    params: Dict[str, Any],
) -> nn.Module:
    """Construct an adapter module by name. `d_model` may be any width
    (e.g. 256, 1024, ...); every adapter takes it as a plain constructor arg.
    """
    if adapter_type == 'dual_residual':
        from models.neural_networks.adapters.dual_residual_adapter import (
            DualResidualAdapter,
        )

        return DualResidualAdapter(
            d_model,
            n_time,
            n_static,
            dropout=params.get('dropout', 0.1),
            combined_dropout=params.get('combined_dropout', 0.2),
            hidden_multiplier=params.get('hidden_multiplier', 2),
        )
    elif adapter_type == 'gated':
        from models.neural_networks.adapters.gated_adapter import GatedAdapter

        return GatedAdapter(d_model, n_time)
    elif adapter_type == 'feedforward':
        from models.neural_networks.adapters.feedforward_adapter import (
            FeedforwardAdapter,
        )

        return FeedforwardAdapter(d_model, n_time, params.get('hidden_multiplier', 2))
    elif adapter_type == 'conv':
        from models.neural_networks.adapters.conv_adapter import ConvAdapter

        return ConvAdapter(d_model, n_time, params.get('kernel_size', 3))
    elif adapter_type == 'attention':
        from models.neural_networks.adapters.attention_adapter import (
            AttentionAdapter,
        )

        return AttentionAdapter(d_model, n_time, params.get('num_heads', 4))
    elif adapter_type == 'bottleneck':
        from models.neural_networks.adapters.bottleneck_adapter import (
            BottleneckAdapter,
        )

        return BottleneckAdapter(d_model, n_time, params.get('bottleneck_size', 64))
    elif adapter_type == 'moe':
        from models.neural_networks.adapters.moe_adapter import MoEAdapter

        return MoEAdapter(
            d_model,
            n_time,
            params.get('num_experts', 4),
            params.get('expert_size', d_model),
        )
    elif adapter_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")


def apply_adapter(
    adapter: nn.Module,
    adapter_type: str,
    hidden: torch.Tensor,
    batch_x_ft: torch.Tensor,
    batch_c_ft: torch.Tensor,
    obs: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Call `adapter` with the argument set its type expects."""
    if adapter_type in _STANDARD_ADAPTERS:
        return adapter(hidden, batch_x_ft, batch_c_ft)
    elif adapter_type == 'none':
        return adapter(hidden)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
