"""
Kriging-Inspired Locality Adapter with Relational Deep Sets.

Extension of KrigingLocalityAdapter where the key encoder φ_rel
is a relational (pair) function over (query_site, bank_site) rather
than a purely element-wise function over bank sites alone.

For spatial catchment attributes this is more principled: the relevance
of a bank site to the current query depends on *how different* the two
sites are, not just on the bank site's absolute attribute values.

─────────────────────────────────────────────────────────────────
GATE FORMULATION
─────────────────────────────────────────────────────────────────
φ is split into two networks:

  φ_q   (element-wise)  : c₀                          → key_dim
  φ_rel (relational)    : (cᵢ, c₀-cᵢ, ‖c₀-cᵢ‖₂)    → key_dim

    q   = φ_q(c₀)
    kᵢ  = φ_rel(cᵢ, c₀-cᵢ, ‖c₀-cᵢ‖₂)    relational key for bank site i
    aᵢ  = softmax( kᵢ · q / √d )          attention weights
    z   = Σᵢ aᵢ · V(cᵢ)                   attended context
    α   = σ( ρ([q, z]) )                   locality gate ∈ [0,1]

─────────────────────────────────────────────────────────────────
MODE 1  use_fm_embeddings=False  (default)
─────────────────────────────────────────────────────────────────
Bank stores c only.

    φ_q   input : c₀                          [n_static]
    φ_rel input : concat(cᵢ, c₀-cᵢ, ‖…‖₂)   [2·n_static + 1]

─────────────────────────────────────────────────────────────────
MODE 2  use_fm_embeddings=True
─────────────────────────────────────────────────────────────────
Bank stores [c, h̄] where h̄ = mean_t h_FM(s, t).

φ_q and φ_rel receive the FM mean embedding too.
Distance features are still over static attrs only.

    φ_q   input : concat(c₀, h̄₀)                      [n_static + d_model]
    φ_rel input : concat(cᵢ, h̄ᵢ, c₀-cᵢ, ‖c₀-cᵢ‖₂)   [2·n_static + d_model + 1]

─────────────────────────────────────────────────────────────────
OUTPUT (both modes)

    δ(s₀, t) = LayerNorm( MLP([h_FM | project(x) | project(c)]) )
    ŷ(s₀, t) = h_FM(s₀, t) + α(s₀) · δ(s₀, t)

α → 0 : ungauged / unseen  → foundation model dominates
α → 1 : well-observed      → local correction applied
─────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class RelationalKrigingAdapter(nn.Module):
    """
    Locality adapter with a relational Deep Sets familiarity gate.

    Differs from KrigingLocalityAdapter in that the key encoder φ_rel
    is a pair function over (bank_site, query_site), incorporating the
    element-wise difference vector and scalar L2 distance between sites.
    This allows the gate to distinguish between sites that are globally
    similar but directionally different in attribute space.

    Parameters
    ----------
    d_model : int
        Foundation model hidden dimension.
    n_static : int
        Number of static catchment attributes (site signature).
    n_time : int
        Number of time-series forcing features.
    key_dim : int
        Dimension for Q/K/V projections.
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
        If True (Mode 2), bank also stores temporal-mean FM embeddings
        and φ_q / φ_rel receive [c, h̄] rather than c alone.
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

        # Bank entry: [c] or [c, h̄]
        _bank_entry_dim = n_static + d_model if use_fm_embeddings else n_static
        # φ_q input: same as bank entry (element-wise query encoding)
        _phi_q_in       = _bank_entry_dim
        # φ_rel input: bank_entry + static_diff + scalar_dist
        _phi_rel_in     = _bank_entry_dim + n_static + 1

        # ------------------------------------------------------------------
        # φ_q : element-wise query encoder
        # ------------------------------------------------------------------
        self.phi_q = nn.Sequential(
            nn.Linear(_phi_q_in, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        # ------------------------------------------------------------------
        # φ_rel : relational key encoder  (bank_entry, diff, dist) → key_dim
        # ------------------------------------------------------------------
        self.phi_rel = nn.Sequential(
            nn.Linear(_phi_rel_in, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )

        # ------------------------------------------------------------------
        # Value projection: element-wise from bank entry
        # ------------------------------------------------------------------
        self.value_proj = nn.Linear(_bank_entry_dim, key_dim)

        # ------------------------------------------------------------------
        # ρ : [q | attended_context] → α ∈ [0,1]
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
        # Site bank — plain buffer, no gradient.
        # ------------------------------------------------------------------
        self.register_buffer('_bank', torch.zeros(0, _bank_entry_dim))
        self._bank_ptr  = 0
        self._bank_full = False

        log.info(
            f"RelationalKrigingAdapter | d_model={d_model}, n_static={n_static}, "
            f"n_time={n_time}, key_dim={key_dim}, max_bank={max_bank_size}, "
            f"bank_dropout={bank_dropout}, use_fm_embeddings={use_fm_embeddings}, "
            f"phi_rel_in={_phi_rel_in}"
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
        Mode 1: c                      → [B, n_static]
        Mode 2: concat(c, mean_t h_FM) → [B, n_static + d_model]
        """
        if self.use_fm_embeddings and hidden_states is not None:
            h_mean = hidden_states.detach().mean(dim=1)
            return torch.cat([static_features.detach(), h_mean], dim=-1)
        return static_features.detach()

    @torch.no_grad()
    def _update_bank(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> None:
        """Add current batch's entries to the circular bank."""
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

        Parameters
        ----------
        static_attrs : [N_sites, n_static]
        h_means      : [N_sites, d_model], optional
            Required when use_fm_embeddings=True.
        """
        if self.use_fm_embeddings:
            if h_means is None:
                raise ValueError(
                    "h_means is required when use_fm_embeddings=True."
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
        return self._bank.size(0)

    # ------------------------------------------------------------------
    # Pair features for φ_rel
    # ------------------------------------------------------------------

    def _pair_features(
        self,
        query_static: torch.Tensor,   # [B, n_static]
        bank:         torch.Tensor,   # [N, bank_entry_dim]
    ) -> torch.Tensor:
        """
        Build relational input for φ_rel for each (query, bank) pair.

            diff  = c₀ - cᵢ           element-wise static difference [n_static]
            dist  = ‖c₀ - cᵢ‖₂        scalar L2 distance             [1]
            input = concat(bank_entry, diff, dist)

        Distance is always computed over static attrs only — FM embeddings
        are not physically interpretable as a spatial distance.

        Returns
        -------
        pair_in : [B, N, bank_entry_dim + n_static + 1]
        """
        B = query_static.size(0)
        N = bank.size(0)

        # Static part of bank only (first n_static columns)
        bank_static = bank[:, :self.n_static]                           # [N, n_static]

        q_exp = query_static.unsqueeze(1).expand(-1, N, -1)             # [B, N, n_static]
        b_exp = bank_static.unsqueeze(0).expand(B, -1, -1)              # [B, N, n_static]

        diff = q_exp - b_exp                                             # [B, N, n_static]
        dist = diff.norm(dim=-1, keepdim=True)                           # [B, N, 1]

        bank_exp = bank.unsqueeze(0).expand(B, -1, -1)                  # [B, N, bank_entry_dim]

        return torch.cat([bank_exp, diff, dist], dim=-1)                 # [B, N, phi_rel_in]

    # ------------------------------------------------------------------
    # Locality gate
    # ------------------------------------------------------------------

    def _compute_gate(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute α ∈ [0,1] per site using relational Deep Sets.

            q   = φ_q(c₀)
            kᵢ  = φ_rel(cᵢ, c₀-cᵢ, ‖c₀-cᵢ‖)
            aᵢ  = softmax( kᵢ · q / √d )
            z   = Σᵢ aᵢ · V(cᵢ)
            α   = σ( ρ([q, z]) )

        Returns: [B, 1, 1]
        """
        bank = self._bank
        N    = bank.size(0)

        if N == 0:
            return torch.zeros(
                static_features.size(0), 1, 1, device=static_features.device
            )

        if N > self.bank_subsample:
            idx  = torch.randperm(N, device=bank.device)[: self.bank_subsample]
            bank = bank[idx]
            N    = self.bank_subsample

        if self.training and self.bank_dropout > 0.0 and N > 1:
            keep = torch.rand(N, device=bank.device) > self.bank_dropout
            if keep.sum() >= 1:
                bank = bank[keep]
                N    = bank.size(0)

        # --- Query (element-wise) ---
        query_input = self._make_bank_entry(static_features, hidden_states)
        q = self.phi_q(query_input)   # [B, key_dim]

        # --- Relational keys (pair function, grads flow through diff) ---
        pair_in = self._pair_features(static_features, bank)   # [B, N, phi_rel_in]
        k = self.phi_rel(pair_in)                               # [B, N, key_dim]

        # --- Values (element-wise from bank, no gradient needed) ---
        with torch.no_grad():
            bank_exp = bank.unsqueeze(0).expand(static_features.size(0), -1, -1)
            v = self.value_proj(bank_exp)   # [B, N, key_dim]

        # --- Attention: kᵢ · q / √d ---
        scores  = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / (self.key_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)   # [B, N]

        # --- Attended context ---
        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)   # [B, key_dim]

        # --- Gate ---
        alpha = self.rho(torch.cat([q, context], dim=-1))   # [B, 1]
        return alpha.unsqueeze(1)                             # [B, 1, 1]

    # ------------------------------------------------------------------
    # Local temporal correction
    # ------------------------------------------------------------------

    def _local_correction(
        self,
        hidden_states:   torch.Tensor,
        time_features:   torch.Tensor,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        T = hidden_states.size(1)

        time_proj   = self.local_time_proj(time_features)
        static_proj = self.local_static_proj(static_features).unsqueeze(1).expand(-1, T, -1)

        combined = torch.cat([hidden_states, time_proj, static_proj], dim=-1)
        return self.correction_norm(self.local_correction(combined))

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
        hidden_states   : [B, T, d_model]
        time_features   : [B, T, n_time]
        static_features : [B, n_static]

        Returns
        -------
        [B, T, d_model]   h_FM + α · δ
        """
        if self.training:
            self._update_bank(
                static_features,
                hidden_states if self.use_fm_embeddings else None,
            )

        alpha = self._compute_gate(
            static_features,
            hidden_states if self.use_fm_embeddings else None,
        )   # [B, 1, 1]

        delta = self._local_correction(hidden_states, time_features, static_features)

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
        Inspect gate values without modifying the bank.

        Parameters
        ----------
        static_features : [B, n_static]
        hidden_states   : [B, T, d_model], optional. Required if use_fm_embeddings=True.

        Returns
        -------
        alpha : [B]
        """
        was_training = self.training
        self.eval()
        alpha = self._compute_gate(static_features, hidden_states).squeeze(-1).squeeze(-1)
        if was_training:
            self.train()
        return alpha
