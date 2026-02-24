import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

log = logging.getLogger(__name__)


class LocalizedStationAdapter(nn.Module):
    """
    Localized adapter that uses station identification embeddings with positional encoding.
    Similar stations will be grouped together in the embedding space.
    
    Features:
    - Station ID embeddings with learnable positional encoding
    - Dropout-like behavior during training (randomly drops station info)
    - Always available during temporal testing
    - Disabled during spatial testing
    
    Parameters
    ----------
    d_model : int
        Model dimension
    n_stations : int
        Number of unique stations
    embedding_dim : int
        Dimension of station embeddings
    dropout_rate : float
        Rate for randomly dropping station information during training
    use_positional : bool
        Whether to use positional encoding for station embeddings
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_stations: int,
        embedding_dim: int = 64,
        dropout_rate: float = 0.2,
        use_positional: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.use_positional = use_positional
        
        # Station ID embedding layer
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        
        # Positional encoding for station embeddings
        if use_positional:
            self.positional_encoding = self._create_positional_encoding(n_stations, embedding_dim)
            self.register_buffer('pos_encoding', self.positional_encoding)
        
        # Projection layers
        self.station_proj = nn.Linear(embedding_dim, d_model)
        self.feature_proj = nn.Linear(d_model + d_model, d_model)  # input + station features
        
        # Gating mechanism to control station influence
        self.gate = nn.Sequential(
            nn.Linear(d_model + embedding_dim, d_model),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        log.info(f"Initialized LocalizedStationAdapter: {n_stations} stations, {embedding_dim}D embeddings")
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding for station IDs."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        station_ids: torch.Tensor = None,
        is_spatial_test: bool = False,
        is_training: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the localized station adapter.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Input features [batch_size, seq_len, d_model]
        station_ids : torch.Tensor
            Station IDs [batch_size] or None
        is_spatial_test : bool
            Whether this is spatial testing (disables station info)
        is_training : bool
            Whether model is in training mode
            
        Returns
        -------
        torch.Tensor
            Adapted features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Skip station adaptation during spatial testing
        if is_spatial_test:
            log.debug("Spatial testing mode: skipping station adaptation")
            return hidden_states
        
        # Skip if no station IDs provided
        if station_ids is None:
            log.debug("No station IDs provided: returning original features")
            return hidden_states
        
        # Randomly drop station information during training (like dropout)
        if is_training and self.dropout_rate > 0:
            drop_mask = torch.rand(batch_size, device=hidden_states.device) < self.dropout_rate
            if drop_mask.any():
                log.debug(f"Dropping station info for {drop_mask.sum().item()}/{batch_size} samples")
                # For dropped samples, return original features
                station_ids = station_ids.clone()
                station_ids[drop_mask] = 0  # Use a default/neutral station ID
        
        # Get station embeddings
        station_embeds = self.station_embedding(station_ids)  # [batch_size, embedding_dim]
        
        # Add positional encoding if enabled
        if self.use_positional and hasattr(self, 'pos_encoding'):
            # Clamp station IDs to valid range for positional encoding
            pos_ids = torch.clamp(station_ids, 0, self.pos_encoding.size(0) - 1)
            pos_embeds = self.pos_encoding[pos_ids]  # [batch_size, embedding_dim]
            station_embeds = station_embeds + pos_embeds
        
        # Project station embeddings to model dimension
        station_features = self.station_proj(station_embeds)  # [batch_size, d_model]
        
        # Expand station features to match sequence length
        station_features = station_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, d_model]
        
        # Compute gating weights
        gate_input = torch.cat([hidden_states, station_features], dim=-1)  # [batch_size, seq_len, d_model*2]
        gate_weights = self.gate(gate_input)  # [batch_size, seq_len, d_model]
        
        # Apply gated combination
        gated_station = gate_weights * station_features
        combined = hidden_states + gated_station
        
        # Final projection and normalization
        output = self.feature_proj(torch.cat([hidden_states, gated_station], dim=-1))
        output = self.layer_norm(output)
        
        return output
    
    def get_station_similarities(self, station_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Compute similarity matrix between station embeddings.
        Useful for analyzing which stations are grouped together.
        
        Parameters
        ----------
        station_ids : torch.Tensor, optional
            Specific station IDs to compute similarities for.
            If None, computes for all stations.
            
        Returns
        -------
        torch.Tensor
            Similarity matrix [n_stations, n_stations]
        """
        if station_ids is None:
            station_ids = torch.arange(self.n_stations, device=next(self.parameters()).device)
        
        # Get embeddings
        embeds = self.station_embedding(station_ids)
        
        # Add positional encoding if enabled
        if self.use_positional and hasattr(self, 'pos_encoding'):
            pos_ids = torch.clamp(station_ids, 0, self.pos_encoding.size(0) - 1)
            pos_embeds = self.pos_encoding[pos_ids]
            embeds = embeds + pos_embeds
        
        # Compute cosine similarity
        embeds_norm = F.normalize(embeds, p=2, dim=1)
        similarity = torch.mm(embeds_norm, embeds_norm.t())
        
        return similarity
    
    def set_dropout_rate(self, rate: float):
        """Update dropout rate for station information."""
        self.dropout_rate = rate
        log.info(f"Updated station dropout rate to {rate}")


class AdaptiveLocalizedAdapter(LocalizedStationAdapter):
    """
    Enhanced version that adapts station influence based on local context.
    
    Additional features:
    - Context-aware station weighting
    - Adaptive embedding dimension based on local variance
    - Multi-scale station grouping
    """
    
    def __init__(
        self,
        d_model: int,
        n_stations: int,
        embedding_dim: int = 64,
        dropout_rate: float = 0.2,
        use_positional: bool = True,
        use_adaptive_weighting: bool = True,
        context_window: int = 7
    ):
        super().__init__(d_model, n_stations, embedding_dim, dropout_rate, use_positional)
        
        self.use_adaptive_weighting = use_adaptive_weighting
        self.context_window = context_window
        
        if use_adaptive_weighting:
            # Context analysis for adaptive weighting
            self.context_analyzer = nn.Sequential(
                nn.Conv1d(d_model, d_model//2, kernel_size=context_window, padding=context_window//2),
                nn.ReLU(),
                nn.Conv1d(d_model//2, 1, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        station_ids: torch.Tensor = None,
        is_spatial_test: bool = False,
        is_training: bool = True
    ) -> torch.Tensor:
        """Enhanced forward pass with adaptive weighting."""
        
        # Use base implementation if adaptive weighting is disabled
        if not self.use_adaptive_weighting:
            return super().forward(hidden_states, station_ids, is_spatial_test, is_training)
        
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Skip station adaptation during spatial testing
        if is_spatial_test or station_ids is None:
            return hidden_states
        
        # Analyze local context to determine station influence
        # Reshape for conv1d: [batch_size, d_model, seq_len]
        context_input = hidden_states.transpose(1, 2)
        context_weights = self.context_analyzer(context_input)  # [batch_size, 1, seq_len]
        context_weights = context_weights.transpose(1, 2)  # [batch_size, seq_len, 1]
        
        # Get base station adaptation
        base_output = super().forward(hidden_states, station_ids, is_spatial_test, is_training)
        
        # Apply context-aware weighting
        station_contribution = base_output - hidden_states
        weighted_contribution = station_contribution * context_weights
        
        final_output = hidden_states + weighted_contribution
        
        return final_output