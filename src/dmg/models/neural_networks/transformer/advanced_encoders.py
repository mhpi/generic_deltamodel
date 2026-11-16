"""
Advanced Temporal Encoders for Time Series Transformers

Citations:
- Informer: Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" AAAI 2021
- tAPE: Xu et al. "Time-Aware Positional Encoding for Time Series Forecasting" arXiv:2023
- TFT: Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" ICLR 2021

This file provides three distinct temporal encoding approaches optimized for daily time series data
that can be used as drop-in replacements for standard PositionalEncoding in StefaLand.
"""

import math

import numpy as np
import torch
import torch.nn as nn


class InformerPositionalEncoding(nn.Module):
    """
    Informer-style temporal encoding combining learnable and sinusoidal position encodings.

    Citation: Zhou et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" AAAI 2021

    Mathematical Foundation:
    The Informer uses enhanced input representation combining:
    1. Learnable position embedding: PE_l ∈ R^{L×d}
    2. Sinusoidal position embedding: PE_s[pos, 2i] = sin(pos/10000^{2i/d})
                                      PE_s[pos, 2i+1] = cos(pos/10000^{2i/d})
    3. Combined encoding: PE_combined = W_fusion([PE_l; PE_s])

    For temporal features, applies multi-scale embedding:
    Temporal_embed = Σ W_i * normalize(temporal_feature_i)

    Final encoding uses adaptive gating:
    output = gate_pos * (x + PE_combined) + gate_temp * (x + Temporal_embed) + gate_cyc * (x + Cyclical_embed)
    where gates are learned through sigmoid(W_gate * features)

    Optimized for daily time series data.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Dual position encoding approach (Informer paper Section 3.3)
        # PE_l: learnable position embedding
        self.learnable_pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        # PE_s: sinusoidal position embedding
        self.sinusoidal_pe = self._create_sinusoidal_encoding(max_len, d_model)
        # W_fusion: fusion weight matrix
        self.pe_fusion = nn.Linear(d_model * 2, d_model)

        # Multi-scale temporal feature processing for daily data
        # Optimized for day, week, month, quarter, weekday, year, special features
        self.temporal_projections = nn.ModuleDict(
            {
                'day': nn.Linear(1, d_model // 7),
                'week': nn.Linear(1, d_model // 7),
                'month': nn.Linear(1, d_model // 7),
                'quarter': nn.Linear(1, d_model // 7),
                'weekday': nn.Linear(1, d_model // 7),
                'year': nn.Linear(1, d_model // 7),
                'special': nn.Linear(1, d_model // 7),
            }
        )

        # Cyclical encoding for periodic patterns in daily data
        # Uses sin/cos transformations: f_cyc(t) = [sin(2πt/period), cos(2πt/period)]
        self.cyclical_projections = nn.ModuleDict(
            {
                'weekly': nn.Linear(2, d_model // 5),  # 7-day cycle
                'monthly': nn.Linear(2, d_model // 5),  # ~30-day cycle
                'quarterly': nn.Linear(2, d_model // 5),  # ~90-day cycle
                'yearly': nn.Linear(2, d_model // 5),  # ~365-day cycle
                'seasonal': nn.Linear(2, d_model // 5),  # seasonal patterns
            }
        )

        # Adaptive fusion with attention-based gating (Informer enhancement)
        # gate_i = σ(W_gate_i * features_i)
        self.position_gate = nn.Linear(d_model, 1)
        self.temporal_gate = nn.Linear(d_model, 1)
        self.cyclical_gate = nn.Linear(d_model, 1)
        self.final_projection = nn.Linear(d_model * 3, d_model)

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(d_model)

    def _create_sinusoidal_encoding(self, max_len, d_model):
        """
        Create fixed sinusoidal position encoding as in Attention Is All You Need.

        Mathematical formulation:
        PE[pos, 2i] = sin(pos / 10000^{2i/d_model})
        PE[pos, 2i+1] = cos(pos / 10000^{2i/d_model})
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, input_data, index=None, temporal_features=None):
        """
        Args:
            input_data: [batch_size, seq_len, d_model]
            index: position indices (optional)
            temporal_features: [batch_size, seq_len, 7] - [day, week, month, quarter, weekday, year, special]
        """
        batch_size, seq_len, _ = input_data.shape

        # Position encoding combination (Informer Section 3.3)
        if index is None:
            indices = torch.arange(seq_len, device=input_data.device)
        else:
            indices = (
                torch.tensor(index, device=input_data.device)
                if isinstance(index, np.ndarray)
                else index
            )
            if indices.dim() > 1:
                indices = indices.flatten()

        # Combine learnable and sinusoidal encodings: PE_combined = W_fusion([PE_l; PE_s])
        learnable_pos = self.learnable_pe[indices]
        sinusoidal_pos = self.sinusoidal_pe[indices]
        combined_pos = torch.cat([learnable_pos, sinusoidal_pos], dim=-1)
        position_encoding = self.pe_fusion(combined_pos)

        if indices.dim() == 0 or len(indices) == seq_len:
            position_encoding = position_encoding.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            position_encoding = position_encoding.view(batch_size, seq_len, -1)

        pos_encoded = input_data + position_encoding

        # Temporal feature encoding for daily data
        temp_encoded = input_data
        if temporal_features is not None and temporal_features.shape[-1] >= 7:
            temp_features = []

            # Normalize and embed each temporal feature
            day = temporal_features[:, :, 0:1] / 31.0  # day of month
            week = temporal_features[:, :, 1:2] / 53.0  # week of year
            month = temporal_features[:, :, 2:3] / 12.0  # month
            quarter = temporal_features[:, :, 3:4] / 4.0  # quarter
            weekday = temporal_features[:, :, 4:5] / 7.0  # day of week
            year = (temporal_features[:, :, 5:6] - 2000) / 50.0  # normalized year
            special = temporal_features[:, :, 6:7]  # special events/holidays

            temp_features.extend(
                [
                    self.temporal_projections['day'](day),
                    self.temporal_projections['week'](week),
                    self.temporal_projections['month'](month),
                    self.temporal_projections['quarter'](quarter),
                    self.temporal_projections['weekday'](weekday),
                    self.temporal_projections['year'](year),
                    self.temporal_projections['special'](special),
                ]
            )

            temporal_embedding = torch.cat(temp_features, dim=-1)
            temp_encoded = input_data + temporal_embedding

        # Cyclical encoding for daily time series
        # Mathematical foundation: f_cyc(t) = [sin(2πt/T), cos(2πt/T)] where T is the period
        cycle_encoded = input_data
        if temporal_features is not None and temporal_features.shape[-1] >= 6:
            day = temporal_features[:, :, 0]
            week = temporal_features[:, :, 1]
            month = temporal_features[:, :, 2]
            weekday = temporal_features[:, :, 4]

            cyclical_features = []

            # Weekly cycle: T = 7 days
            weekly_sin = torch.sin(2 * math.pi * weekday / 7).unsqueeze(-1)
            weekly_cos = torch.cos(2 * math.pi * weekday / 7).unsqueeze(-1)
            cyclical_features.append(
                self.cyclical_projections['weekly'](
                    torch.cat([weekly_sin, weekly_cos], dim=-1)
                )
            )

            # Monthly cycle: T = 30 days (approximate)
            monthly_sin = torch.sin(2 * math.pi * day / 30).unsqueeze(-1)
            monthly_cos = torch.cos(2 * math.pi * day / 30).unsqueeze(-1)
            cyclical_features.append(
                self.cyclical_projections['monthly'](
                    torch.cat([monthly_sin, monthly_cos], dim=-1)
                )
            )

            # Quarterly cycle: T = 90 days (approximate)
            quarterly_sin = torch.sin(2 * math.pi * day / 90).unsqueeze(-1)
            quarterly_cos = torch.cos(2 * math.pi * day / 90).unsqueeze(-1)
            cyclical_features.append(
                self.cyclical_projections['quarterly'](
                    torch.cat([quarterly_sin, quarterly_cos], dim=-1)
                )
            )

            # Yearly cycle: T = 365 days
            yearly_sin = torch.sin(2 * math.pi * day / 365).unsqueeze(-1)
            yearly_cos = torch.cos(2 * math.pi * day / 365).unsqueeze(-1)
            cyclical_features.append(
                self.cyclical_projections['yearly'](
                    torch.cat([yearly_sin, yearly_cos], dim=-1)
                )
            )

            # Seasonal cycle: T = 12 months
            seasonal_sin = torch.sin(2 * math.pi * month / 12).unsqueeze(-1)
            seasonal_cos = torch.cos(2 * math.pi * month / 12).unsqueeze(-1)
            cyclical_features.append(
                self.cyclical_projections['seasonal'](
                    torch.cat([seasonal_sin, seasonal_cos], dim=-1)
                )
            )

            cyclical_embedding = torch.cat(cyclical_features, dim=-1)
            cycle_encoded = input_data + cyclical_embedding

        # Adaptive fusion with gating (Informer enhancement)
        # Mathematical formulation: gate_i = σ(W_gate_i * features_i)
        pos_weight = torch.sigmoid(self.position_gate(pos_encoded))
        temp_weight = torch.sigmoid(self.temporal_gate(temp_encoded))
        cycle_weight = torch.sigmoid(self.cyclical_gate(cycle_encoded))

        # Normalize weights: w_i = w_i / Σ w_j
        total_weight = pos_weight + temp_weight + cycle_weight + 1e-8
        pos_weight = pos_weight / total_weight
        temp_weight = temp_weight / total_weight
        cycle_weight = cycle_weight / total_weight

        # Weighted combination
        combined = torch.cat(
            [
                pos_weight * pos_encoded,
                temp_weight * temp_encoded,
                cycle_weight * cycle_encoded,
            ],
            dim=-1,
        )

        output = self.final_projection(combined)
        output = self.layer_norm(output)

        return self.dropout(output)


class TAPEPositionalEncoding(nn.Module):
    """
    Time-Aware Positional Encoding (tAPE) implementation
    Enhanced with learnable time dependencies and adaptive scaling.

    Key innovations:
    - Time-aware position embeddings that consider temporal distances
    - Learnable temporal scaling factors
    - Adaptive positional bias based on time intervals
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Learnable temporal scaling factors
        self.temporal_scales = nn.Parameter(torch.ones(d_model // 2))

        # Time-aware position embedding layers
        self.time_projection = nn.Linear(1, d_model)
        self.position_projection = nn.Linear(1, d_model)
        self.time_position_fusion = nn.Linear(d_model * 2, d_model)

        # Adaptive positional bias network
        self.positional_bias_network = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Learnable frequency embeddings for different time scales
        self.frequency_embeddings = nn.Parameter(torch.randn(8, d_model // 8))

        # Time interval encoding
        self.interval_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, d_model)
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_data, index=None, temporal_features=None):
        """
        Args:
            input_data: [batch_size, seq_len, d_model]
            index: position indices or actual timestamps
            temporal_features: [batch_size, seq_len, n_features] - additional temporal context
        """
        batch_size, seq_len, _ = input_data.shape
        device = input_data.device

        # Handle position indices
        if index is None:
            positions = torch.arange(seq_len, dtype=torch.float, device=device)
        else:
            if isinstance(index, np.ndarray):
                positions = torch.tensor(index, dtype=torch.float, device=device)
            else:
                positions = index.float()
            if positions.dim() > 1:
                positions = positions.flatten()

        # Time-aware positional encoding
        positions = positions.unsqueeze(-1)  # [seq_len, 1]

        # Create sinusoidal encodings with learnable scaling
        position_angles = positions / (
            10000 ** (torch.arange(0, self.d_model, 2, device=device) / self.d_model)
        )
        position_angles = position_angles * self.temporal_scales.unsqueeze(0)

        # Apply sine and cosine
        sin_encodings = torch.sin(position_angles)
        cos_encodings = torch.cos(position_angles)

        # Interleave sine and cosine
        pos_encoding = torch.zeros(seq_len, self.d_model, device=device)
        pos_encoding[:, 0::2] = sin_encodings
        pos_encoding[:, 1::2] = cos_encodings[:, : pos_encoding[:, 1::2].shape[1]]

        # Add learnable position projection
        position_proj = self.position_projection(positions)
        pos_encoding = pos_encoding + position_proj

        # Time interval encoding (capture relative temporal distances)
        if seq_len > 1:
            time_diffs = positions[1:] - positions[:-1]
            time_diffs = torch.cat([torch.zeros(1, 1, device=device), time_diffs])
            interval_encoding = self.interval_encoder(time_diffs)
            pos_encoding = pos_encoding + interval_encoding

        # Adaptive positional bias based on position and time features
        if temporal_features is not None and temporal_features.shape[-1] > 0:
            # Use first temporal feature as time reference
            time_ref = temporal_features[:, :, 0:1].mean(dim=0)  # [seq_len, 1]

            # Create position-time pairs for bias calculation
            pos_time_pairs = torch.cat(
                [positions.expand(seq_len, -1), time_ref], dim=-1
            )  # [seq_len, 2]

            positional_bias = self.positional_bias_network(pos_time_pairs)
            pos_encoding = pos_encoding + positional_bias

        # Multi-frequency encoding
        freq_encodings = []
        for i, freq_emb in enumerate(self.frequency_embeddings):
            freq_scale = 2**i  # Different frequency scales
            freq_pos = torch.sin(positions * freq_scale / 1000)
            freq_encoding = freq_pos * freq_emb.unsqueeze(0)
            freq_encodings.append(freq_encoding)

        multi_freq_encoding = torch.cat(freq_encodings, dim=-1)
        pos_encoding = pos_encoding + multi_freq_encoding

        # Expand for batch dimension
        if pos_encoding.dim() == 2:
            pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply to input
        output = input_data + pos_encoding
        output = self.layer_norm(output)

        return self.dropout(output)


class TFTPositionalEncoding(nn.Module):
    """
    Temporal Fusion Transformer (TFT) style temporal encoding
    with variable selection network and gated feature fusion.

    Key innovations:
    - Variable selection network for temporal features
    - Gated linear unit (GLU) for feature processing
    - Multi-head temporal attention
    - Static and dynamic temporal context fusion
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Standard positional encoding
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # Variable selection network (VSN) for temporal features
        self.temporal_vsn = nn.Sequential(
            nn.Linear(8, d_model),  # Assuming 8 temporal features
            nn.ReLU(),
            nn.Linear(d_model, 8),
            nn.Softmax(dim=-1),
        )

        # Temporal feature embeddings with GLU
        self.temporal_embeddings = nn.ModuleDict(
            {
                'hour': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'day': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'week': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'month': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'quarter': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'weekday': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'year': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
                'special': nn.Sequential(nn.Linear(1, d_model * 2), nn.GLU()),
            }
        )

        # Temporal context encoder (multi-head attention for temporal patterns)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=dropout, batch_first=True
        )

        # Gated fusion networks
        self.static_gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.dynamic_gate = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        # Final fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.GLU(), nn.Linear(d_model, d_model)
        )

        # Residual and normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, input_data, index=None, temporal_features=None):
        """
        Args:
            input_data: [batch_size, seq_len, d_model]
            index: position indices
            temporal_features: [batch_size, seq_len, 8] temporal context
        """
        batch_size, seq_len, _ = input_data.shape
        max_len = self.position_embedding.size(0)

        # Static positional encoding
        if index is None:
            indices = torch.arange(seq_len, device=input_data.device)
        else:
            indices = (
                torch.tensor(index, device=input_data.device)
                if isinstance(index, np.ndarray)
                else index
            )
            if indices.dim() > 1:
                indices = indices.flatten()
        # If seq_len exceeds max_len, fold indices so we never go out of bounds.
        indices = indices % max_len

        static_pos = self.position_embedding[indices]
        if static_pos.dim() == 2:
            static_pos = static_pos.unsqueeze(0).expand(batch_size, -1, -1)

        static_encoded = input_data + static_pos

        # Dynamic temporal encoding
        dynamic_encoded = input_data
        if temporal_features is not None and temporal_features.shape[-1] >= 8:
            # Variable selection for temporal features
            temporal_importance = self.temporal_vsn(
                temporal_features
            )  # [batch_size, seq_len, 8]

            # Process each temporal feature
            temporal_embeds = []
            feature_names = [
                'hour',
                'day',
                'week',
                'month',
                'quarter',
                'weekday',
                'year',
                'special',
            ]

            for i, name in enumerate(feature_names):
                # Normalize features
                if i == 0:  # hour
                    norm_feature = temporal_features[:, :, i : i + 1] / 24.0
                elif i == 1:  # day
                    norm_feature = temporal_features[:, :, i : i + 1] / 31.0
                elif i == 2:  # week
                    norm_feature = temporal_features[:, :, i : i + 1] / 53.0
                elif i == 3:  # month
                    norm_feature = temporal_features[:, :, i : i + 1] / 12.0
                elif i == 4:  # quarter
                    norm_feature = temporal_features[:, :, i : i + 1] / 4.0
                elif i == 5:  # weekday
                    norm_feature = temporal_features[:, :, i : i + 1] / 7.0
                elif i == 6:  # year
                    norm_feature = (temporal_features[:, :, i : i + 1] - 2000) / 50.0
                else:  # special
                    norm_feature = temporal_features[:, :, i : i + 1]

                # Embed and weight by importance
                embed = self.temporal_embeddings[name](norm_feature)
                weighted_embed = embed * temporal_importance[:, :, i : i + 1]
                temporal_embeds.append(weighted_embed)

            # Combine temporal embeddings
            combined_temporal = torch.stack(temporal_embeds, dim=-1).sum(dim=-1)

            # Apply temporal attention for pattern capture
            temporal_attended, _ = self.temporal_attention(
                combined_temporal, combined_temporal, combined_temporal
            )

            dynamic_encoded = input_data + temporal_attended

        # Gated fusion of static and dynamic components
        static_gate = self.static_gate(static_encoded)
        dynamic_gate = self.dynamic_gate(dynamic_encoded)

        # Apply gates
        gated_static = static_gate * static_encoded
        gated_dynamic = dynamic_gate * dynamic_encoded

        # Combine with residual connection
        combined = torch.cat([gated_static, gated_dynamic], dim=-1)
        fused = self.feature_fusion(combined)

        # Residual connection and normalization
        output = self.layer_norm1(input_data + fused)
        output = self.layer_norm2(output)

        return self.dropout(output)


# Enhanced base PositionalEncoding with backward compatibility
class PositionalEncoding(nn.Module):
    """
    Enhanced positional encoding that maintains backward compatibility
    while providing access to advanced temporal encoders.

    This can be used as a drop-in replacement for the original PositionalEncoding.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1000, encoding_type='original'):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.encoding_type = encoding_type

        if encoding_type == 'original':
            # Original learnable position embedding for backward compatibility
            self.position_embedding = nn.Parameter(
                torch.empty(max_len, d_model), requires_grad=True
            )
            nn.init.uniform_(self.position_embedding, -0.02, 0.02)
        elif encoding_type == 'informer':
            self.encoder = InformerPositionalEncoding(d_model, dropout, max_len)
        elif encoding_type == 'tAPE':
            self.encoder = TAPEPositionalEncoding(d_model, dropout, max_len)
        elif encoding_type == 'tft':
            self.encoder = TFTPositionalEncoding(d_model, dropout, max_len)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

    def forward(self, input_data, index=None, temporal_features=None):
        """
        Args:
            input_data: [batch_size, seq_len, d_model]
            index: position indices (for backward compatibility)
            temporal_features: [batch_size, seq_len, n_features] - additional temporal context
                              For daily data: [day, week, month, quarter, weekday, year, special]
        """
        if self.encoding_type == 'original':
            # Original implementation for backward compatibility
            if index is None:
                index = np.arange(input_data.size(1))

            pe = self.position_embedding[index].unsqueeze(0)
            input_data = input_data + pe
            return self.dropout(input_data)
        else:
            # Use advanced temporal encoder
            return self.encoder(input_data, index, temporal_features)
