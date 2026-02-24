"""
Kriging-Inspired Locality Adapter with Deep Sets Aggregation.

Two operating modes controlled by `use_fm_embeddings`:

─────────────────────────────────────────────────────────────────
MODE 1  use_fm_embeddings=False  (default, attribute-only gate)
─────────────────────────────────────────────────────────────────
Bank stores static attributes c only.

    q  = φ(c₀)
    kᵢ = φ(cᵢ),  vᵢ = V(cᵢ)       for i in bank
    aᵢ = softmax(q·kᵢ / √d)        attention weights
    z  = Σᵢ aᵢ · vᵢ                Deep Sets attended sum
    α  = σ( ρ([q, z]) )

─────────────────────────────────────────────────────────────────
MODE 2  use_fm_embeddings=True  (richer gate using FM knowledge)
─────────────────────────────────────────────────────────────────
Bank stores [c, h̄] where h̄ = mean_t h_FM(s, t).

φ takes concat(c, h̄) as input — the familiarity score now also
reflects what the foundation model learned about each bank site.

    h̄ᵢ = mean_t h_FM(sᵢ, t)         temporal mean FM embedding
    q  = φ(c₀, h̄₀)
    kᵢ = φ(cᵢ, h̄ᵢ),  vᵢ = V(cᵢ, h̄ᵢ)
    aᵢ = softmax(q·kᵢ / √d)
    z  = Σᵢ aᵢ · vᵢ
    α  = σ( ρ([q, z]) )

─────────────────────────────────────────────────────────────────
In both modes the output is:

    δ(s₀, t) = LayerNorm( MLP([h_FM | project(x) | project(c)]) )
    ŷ(s₀, t) = h_FM(s₀, t) + α(s₀) · δ(s₀, t)

α → 0 : site is ungauged / unseen  → foundation model dominates
α → 1 : site is well-observed / seen → local correction applied
─────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class KrigingLocalityAdapter(nn.Module):
    """
    Learnable, self-calibrating locality adapter.

    The site bank is populated automatically during training from each
    forward pass. No external distance data or station IDs are required.

    Parameters
    ----------
    d_model : int
        Foundation model hidden dimension.
    n_static : int
        Number of static catchment attributes (site signature).
    n_time : int
        Number of time-series forcing features.
    key_dim : int
        Dimension for Q/K/V projections (kriging embedding space).
    max_bank_size : int
        Maximum sites retained in the bank (circular buffer).
    bank_subsample : int
        Sites sampled from bank per forward pass when bank is large.
    bank_dropout : float
        Fraction of bank sites randomly masked during training.
        Forces α to be calibrated (prevents gate always being 1).
    dropout : float
        Dropout in the local correction network.
    use_fm_embeddings : bool
        If True (Mode 2), the bank also stores the temporal-mean FM
        embedding per site and φ takes [c, h̄] as input.  This makes
        the familiarity score sensitive to what the FM already learned
        about each training site, not only its static attributes.
        If False (Mode 1, default), only static attributes are used.
    """

    def __init__(
        self,
        d_model: int,
        n_static: int,
        n_time: int,
        key_dim: int = 64,
        max_bank_size: int = 1024,
        bank_subsample: int = 256,
        bank_dropout: float = 0.3,
        dropout: float = 0.1,
        use_fm_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model           = d_model
        self.n_static          = n_static
        self.n_time            = n_time
        self.key_dim           = key_dim
        self.max_bank_size     = max_bank_size
        self.bank_subsample    = bank_subsample
        self.bank_dropout      = bank_dropout
        self.use_fm_embeddings = use_fm_embeddings

        # φ input dim: static only (Mode 1) or static + mean FM emb (Mode 2)
        _phi_in         = n_static + d_model if use_fm_embeddings else n_static
        _bank_entry_dim = n_static + d_model if use_fm_embeddings else n_static

        # ------------------------------------------------------------------
        # Site signature projections (Q/K/V for kriging attention)
        # ------------------------------------------------------------------
        self.phi = nn.Sequential(
            nn.Linear(_phi_in, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )
        self.value_proj = nn.Linear(_phi_in, key_dim)

        # ------------------------------------------------------------------
        # ρ network: [query_emb | attended_context] → α ∈ [0,1]
        # ------------------------------------------------------------------
        self.rho = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(key_dim, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Local temporal correction δ(s₀, t)
        # Lightweight residual: takes FM embeddings + local forcing data
        # ------------------------------------------------------------------
        self.local_time_proj   = nn.Linear(n_time,   d_model)
        self.local_static_proj = nn.Linear(n_static, d_model)
        self.local_correction  = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.correction_norm = nn.LayerNorm(d_model)

        # ------------------------------------------------------------------
        # Site bank — populated during training.
        # Stores [c] (Mode 1) or [c, h̄] (Mode 2) per site.
        # Plain buffer: moves with .to(device), no gradient.
        # ------------------------------------------------------------------
        self.register_buffer('_bank', torch.zeros(0, _bank_entry_dim))
        self._bank_ptr  = 0
        self._bank_full = False

        log.info(
            f"KrigingLocalityAdapter | d_model={d_model}, n_static={n_static}, "
            f"n_time={n_time}, key_dim={key_dim}, max_bank={max_bank_size}, "
            f"bank_dropout={bank_dropout}, use_fm_embeddings={use_fm_embeddings}"
        )

    # ------------------------------------------------------------------
    # Bank management
    # ------------------------------------------------------------------

    def _make_bank_entry(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the bank entry vector per site.

        Mode 1: just c                           → [B, n_static]
        Mode 2: concat(c, mean_t h_FM(s, t))    → [B, n_static + d_model]
        """
        if self.use_fm_embeddings and hidden_states is not None:
            h_mean = hidden_states.detach().mean(dim=1)   # [B, d_model]
            return torch.cat([static_features.detach(), h_mean], dim=-1)
        return static_features.detach()

    @torch.no_grad()
    def _update_bank(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> None:
        """Add the current batch's entries to the circular bank."""
        new_sites = self._make_bank_entry(static_features, hidden_states)
        B = new_sites.size(0)

        capacity = self._bank.size(0)
        if capacity < self.max_bank_size:
            space  = self.max_bank_size - capacity
            to_add = new_sites[:space]
            self._bank = torch.cat([self._bank, to_add], dim=0)
            B_remaining = B - space
            if B_remaining > 0:
                for i in range(B_remaining):
                    self._bank[self._bank_ptr] = new_sites[space + i]
                    self._bank_ptr = (self._bank_ptr + 1) % self.max_bank_size
                self._bank_full = True
        else:
            for i in range(B):
                self._bank[self._bank_ptr] = new_sites[i]
                self._bank_ptr = (self._bank_ptr + 1) % self.max_bank_size
            self._bank_full = True

    def set_site_bank(
        self,
        static_attrs: torch.Tensor,
        h_means:      Optional[torch.Tensor] = None,
    ) -> None:
        """
        Pre-populate the site bank from all training sites.
        Call this once before training starts for fastest convergence.

        Parameters
        ----------
        static_attrs : [N_sites, n_static]
        h_means      : [N_sites, d_model], optional
            Temporal-mean FM embeddings per site.
            Required (and used) only when use_fm_embeddings=True.
        """
        if self.use_fm_embeddings:
            if h_means is None:
                raise ValueError(
                    "h_means is required when use_fm_embeddings=True. "
                    "Pass the temporal-mean FM embeddings for each training site."
                )
            entries = torch.cat(
                [static_attrs, h_means.to(static_attrs.device)], dim=-1
            )
        else:
            entries = static_attrs

        N = min(entries.size(0), self.max_bank_size)
        self._bank      = entries[:N].to(self._bank.device).detach()
        self._bank_ptr  = N % self.max_bank_size
        self._bank_full = (N >= self.max_bank_size)
        log.info(f"Site bank pre-populated with {N} / {entries.size(0)} sites")

    def bank_size(self) -> int:
        """Number of sites currently in the bank."""
        return self._bank.size(0)

    # ------------------------------------------------------------------
    # Locality gate  α(s₀)
    # ------------------------------------------------------------------

    def _compute_gate(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute α(s₀) ∈ [0,1] for each site in the batch via Deep Sets.

        Mode 1  query_input = c₀
        Mode 2  query_input = concat(c₀, mean_t h_FM(s₀, t))

            q  = φ(query_input)
            kᵢ = φ(bankᵢ),  vᵢ = V(bankᵢ)
            aᵢ = softmax(q·kᵢ / √d)
            z  = Σᵢ aᵢ · vᵢ
            α  = σ( ρ([q, z]) )

        Returns
        -------
        alpha : [B, 1, 1]   broadcastable over [B, T, d_model]
        """
        bank = self._bank   # [N, bank_entry_dim]
        N    = bank.size(0)

        if N == 0:
            return torch.zeros(
                static_features.size(0), 1, 1, device=static_features.device
            )

        # Sub-sample bank for efficiency
        if N > self.bank_subsample:
            idx  = torch.randperm(N, device=bank.device)[: self.bank_subsample]
            bank = bank[idx]
            N    = self.bank_subsample

        # Leave-site-out bank dropout: prevents α collapsing to 1 for all
        # training sites by randomly masking a fraction of the bank.
        if self.training and self.bank_dropout > 0.0 and N > 1:
            keep = torch.rand(N, device=bank.device) > self.bank_dropout
            if keep.sum() >= 1:
                bank = bank[keep]
                N    = bank.size(0)

        # Build query: static only (Mode 1) or static + mean FM emb (Mode 2)
        query_input = self._make_bank_entry(static_features, hidden_states)  # [B, phi_in]

        # Query embedding
        q = self.phi(query_input)   # [B, key_dim]

        # Key / value from bank (no gradient)
        with torch.no_grad():
            k = self.phi(bank)           # [N, key_dim]
            v = self.value_proj(bank)    # [N, key_dim]

        # Attention weights: [B, N]
        scores  = torch.matmul(q, k.t()) / (self.key_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)

        # Deep Sets attended aggregation: Σᵢ aᵢ · vᵢ   [B, key_dim]
        context = torch.matmul(weights, v)

        # ρ([q, z]) → α
        gate_input = torch.cat([q, context], dim=-1)   # [B, 2*key_dim]
        alpha      = self.rho(gate_input)               # [B, 1]

        return alpha.unsqueeze(1)                        # [B, 1, 1]

    # ------------------------------------------------------------------
    # Local temporal correction  δ(s₀, t)
    # ------------------------------------------------------------------

    def _local_correction(
        self,
        hidden_states:   torch.Tensor,
        time_features:   torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        δ(s₀, t) = LayerNorm( MLP([h_FM | project(x) | project(c)]) )

        Parameters
        ----------
        hidden_states   : [B, T, d_model]
        time_features   : [B, T, n_time]
        static_features : [B, n_static]

        Returns
        -------
        delta : [B, T, d_model]
        """
        T = hidden_states.size(1)

        time_proj   = self.local_time_proj(time_features)                     # [B, T, d_model]
        static_proj = self.local_static_proj(static_features)                 # [B, d_model]
        static_proj = static_proj.unsqueeze(1).expand(-1, T, -1)             # [B, T, d_model]

        combined = torch.cat(
            [hidden_states, time_proj, static_proj], dim=-1
        )                                                                      # [B, T, 3*d_model]
        delta = self.local_correction(combined)                                # [B, T, d_model]
        return self.correction_norm(delta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states:   torch.Tensor,
        time_features:   torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states   : [B, T, d_model]  Foundation model embeddings.
        time_features   : [B, T, n_time]   Fine-tuning time-series forcings.
        static_features : [B, n_static]    Fine-tuning static attributes.

        Returns
        -------
        output : [B, T, d_model]
            h_FM + α · δ

        Behaviour
        ---------
        During training  : progressively builds site bank from each batch.
                           Mode 2 stores mean FM embedding alongside c.
        During inference : bank is fixed; α reflects training-distribution support.
        """
        # Populate bank during training
        if self.training:
            self._update_bank(
                static_features,
                hidden_states if self.use_fm_embeddings else None,
            )

        # Locality gate α(s₀) ∈ [0,1]
        alpha = self._compute_gate(
            static_features,
            hidden_states if self.use_fm_embeddings else None,
        )  # [B, 1, 1]

        # Local temporal correction δ(s₀, t)
        delta = self._local_correction(
            hidden_states, time_features, static_features
        )  # [B, T, d_model]

        # Gated output: h_FM + α · δ
        # α = 0 → pure foundation model  (ungauged / unseen sites)
        # α = 1 → full local correction  (well-observed / seen sites)
        return hidden_states + alpha * delta

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gate_values(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inspect gate values for a batch of sites without modifying the bank.
        Useful for analysing which sites are treated as seen vs unseen.

        Parameters
        ----------
        static_features : [B, n_static]
        hidden_states   : [B, T, d_model], optional
            Required when use_fm_embeddings=True.

        Returns
        -------
        alpha : [B]   gate values in [0,1]
        """
        was_training = self.training
        self.eval()
        alpha = self._compute_gate(static_features, hidden_states).squeeze(-1).squeeze(-1)
        if was_training:
            self.train()
        return alpha
