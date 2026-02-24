"""
Kriging Adapter with Observation-Aware DeepSets Correction.

Extends the standard locality adapter to incorporate nearby *observed*
target values from the site bank into a kernel-weighted DeepSets
embedding correction.

─────────────────────────────────────────────────────────────────
ARCHITECTURE
─────────────────────────────────────────────────────────────────
Foundation branch (same as KrigingLocalityAdapter):

    E  = h_FM(s₀, t)            [B, T, d_model]
    α  = gate(c₀, bank)         [B, 1, 1]       attribute locality gate
    δ  = MLP([E | x_t | c₀])   [B, T, d_model]  temporal correction

Observation branch (new — the key addition over the original adapter):

    Δᵢ = [c₀ − cᵢ, ‖c₀ − cᵢ‖₂]          relative attribute features
    mᵢ ∈ {0,1}                             valid-observation flag
    yᵢ_safe = yᵢ · mᵢ                     masked obs (0 when missing)

    wᵢ = softplus(κ(Δᵢ, mᵢ)) · mᵢ         learned kernel weight
                                            → 0 when obs missing or far
    hᵢ = ϕ([Δᵢ, yᵢ_safe, mᵢ])             neighbor token encoding
    H  = Σᵢ wᵢ hᵢ                          kernel-weighted aggregation
    dE = ρ(H) ∈ ℝ^d_model                 embedding correction

Density gate (makes correction fade in sparse regions):

    g = σ(γ([log(Σwᵢ + ε), Σwᵢ]))        g ∈ [0,1]

Final output:

    output(s₀, t) = h_FM + α · δ + g · dE
                    └────────────┘  └──────┘
                    attribute branch    observation branch (NEW)

─────────────────────────────────────────────────────────────────
FADE-OUT GUARANTEES
─────────────────────────────────────────────────────────────────
• No valid neighbors (mᵢ=0 ∀i):  Σwᵢ ≈ 0 → g ≈ 0 → dE contributes 0
• Far neighbors (‖Δᵢ‖ large):    κ → 0   → wᵢ → 0 → same fade-out
• Bank is empty:                   g = 0 exactly (forced)
• Both branches are additive:
    α≈0, g≈0  →  pure foundation model
    α≈1, g≈0  →  attribute-driven correction only
    α≈0, g≈1  →  observation-driven correction only
    α≈1, g≈1  →  both corrections active

─────────────────────────────────────────────────────────────────
BANK FORMAT
─────────────────────────────────────────────────────────────────
Mode 1 (use_fm_embeddings=False):
    bank row = [c (n_static) | y_safe (1) | obs_mask (1)]

Mode 2 (use_fm_embeddings=True):
    bank row = [c (n_static) | h̄ (d_model) | y_safe (1) | obs_mask (1)]

h̄ = mean_t h_FM(s, t)  — temporal-mean FM embedding per site.
The attribute gate φ takes [c, h̄] in Mode 2 (same as KrigingLocalityAdapter).
The obs DeepSets branch always uses static-attribute differences as relative
features (not h̄), since static attrs define the spatial similarity structure.

─────────────────────────────────────────────────────────────────
RESIDUAL TIP (from formulation notes)
─────────────────────────────────────────────────────────────────
Instead of passing raw obs, you can pass residuals:

    obs = y_observed − y_hat_global(x, t; E)

This makes the DeepSets branch learn *local corrections* rather than
the full target signal, which tends to generalize better and prevents
the obs branch from fighting the foundation model.
─────────────────────────────────────────────────────────────────
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Feedforward MLP with GELU activations."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: tuple,
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
            if dropout > 0.0:
                layers += [nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class KrigingObsAdapter(nn.Module):
    """
    Kriging-inspired adapter that corrects foundation model embeddings
    using kernel-weighted DeepSets aggregation of nearby *observations*.

    Parameters
    ----------
    d_model : int
        Foundation model hidden dimension.
    n_static : int
        Number of static catchment attributes (site signature).
    n_time : int
        Number of time-series forcing features.
    key_dim : int
        Dimension for Q/K/V projections in the attribute gate.
    phi_hidden : tuple of int
        Hidden layer sizes for the DeepSets encoder φ (obs branch).
    rho_hidden : tuple of int
        Hidden layer sizes for the DeepSets output network ρ (obs branch).
    weight_hidden : tuple of int
        Hidden layer sizes for the kernel weight network κ.
    max_bank_size : int
        Maximum sites retained in the bank (circular buffer).
    bank_subsample : int
        Sites sampled from the bank per forward pass.
    bank_dropout : float
        Fraction of bank sites randomly masked during training.
        Prevents gate α from collapsing to 1 for all training sites.
    dropout : float
        Dropout applied in the local correction and obs-branch networks.
    use_gate : bool
        If True, add a learned density gate g that scales the obs correction.
        When False, dE is used directly without gating.
    use_fm_embeddings : bool
        If True (Mode 2), bank stores temporal-mean FM embedding h̄ alongside
        static attrs, and the attribute gate uses [c, h̄] as input.
        If False (Mode 1, default), only static attrs are used.
    eps : float
        Numerical stability constant in the density gate log term.
    """

    # Number of extra scalars appended to each bank entry: [y_safe, obs_mask]
    _OBS_EXTRA: int = 2

    def __init__(
        self,
        d_model: int,
        n_static: int,
        n_time: int,
        key_dim: int = 64,
        phi_hidden: tuple = (128, 128),
        rho_hidden: tuple = (128,),
        weight_hidden: tuple = (64, 64),
        max_bank_size: int = 1024,
        bank_subsample: int = 256,
        bank_dropout: float = 0.3,
        dropout: float = 0.1,
        use_gate: bool = True,
        use_fm_embeddings: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model           = d_model
        self.n_static          = n_static
        self.n_time            = n_time
        self.key_dim           = key_dim
        self.max_bank_size     = max_bank_size
        self.bank_subsample    = bank_subsample
        self.bank_dropout      = bank_dropout
        self.use_gate          = use_gate
        self.use_fm_embeddings = use_fm_embeddings
        self.eps               = eps

        # ------------------------------------------------------------------
        # Attribute gate (same design as KrigingLocalityAdapter)
        # phi_in = n_static (Mode 1)  or  n_static + d_model (Mode 2)
        # ------------------------------------------------------------------
        _phi_in          = n_static + d_model if use_fm_embeddings else n_static
        _bank_attr_dim   = _phi_in  # portion of bank row used by the gate

        self.phi_gate = nn.Sequential(
            nn.Linear(_phi_in, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
        )
        self.value_proj = nn.Linear(_phi_in, key_dim)
        self.rho_gate   = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(key_dim, 1),
            nn.Sigmoid(),
        )

        # ------------------------------------------------------------------
        # Local temporal correction  δ(s₀, t)
        # MLP([h_FM | project(x_t) | project(c₀)]) → d_model
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
        # Observation DeepSets branch
        #
        # Relative features  Δᵢ = [c₀ − cᵢ (n_static), ‖c₀ − cᵢ‖₂ (1)]
        #   rel_dim   = n_static + 1
        #
        # Kernel weight network  κ: [Δᵢ, mᵢ] → scalar
        #   w_in_dim  = rel_dim + 1   (obs_mask; y deliberately excluded)
        #
        # Neighbor token  ϕ: [Δᵢ, yᵢ_safe, mᵢ] → phi_hidden[-1]
        #   token_dim = rel_dim + 1 + 1
        # ------------------------------------------------------------------
        _rel_dim   = n_static + 1
        _w_in_dim  = _rel_dim + 1          # + obs_mask
        _token_dim = _rel_dim + 1 + 1      # + y_safe + obs_mask
        _phi_out   = phi_hidden[-1]

        self.kappa   = _MLP(_w_in_dim,  weight_hidden, 1,        dropout=0.0)
        self.phi_obs = _MLP(_token_dim, phi_hidden,    _phi_out, dropout=dropout)
        self.rho_obs = _MLP(_phi_out,   rho_hidden,    d_model,  dropout=dropout)

        if use_gate:
            # Input: [log(Σwᵢ + ε), Σwᵢ]  →  g ∈ [0, 1]
            self.gate_net = _MLP(2, (32, 32), 1, dropout=0.0)

        # ------------------------------------------------------------------
        # Site bank — populated automatically during training.
        #
        # Mode 1 row: [c (n_static)           | y_safe (1) | obs_mask (1)]
        # Mode 2 row: [c (n_static) | h̄ (d_m) | y_safe (1) | obs_mask (1)]
        # ------------------------------------------------------------------
        _bank_full_dim = _bank_attr_dim + self._OBS_EXTRA
        self.register_buffer('_bank', torch.zeros(0, _bank_full_dim))
        self._bank_attr_dim = _bank_attr_dim  # dims before the two obs scalars
        self._bank_ptr      = 0
        self._bank_full     = False

        log.info(
            "KrigingObsAdapter | d_model=%d, n_static=%d, n_time=%d, "
            "key_dim=%d, max_bank=%d, bank_dropout=%.2f, "
            "use_gate=%s, use_fm_embeddings=%s",
            d_model, n_static, n_time, key_dim, max_bank_size,
            bank_dropout, use_gate, use_fm_embeddings,
        )

    # ------------------------------------------------------------------
    # Bank management
    # ------------------------------------------------------------------

    def _make_bank_entry(
        self,
        static_features: torch.Tensor,
        obs:             torch.Tensor,
        obs_mask:        torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build one bank row per site (detached from the computation graph).

        Mode 1: [c,    y_safe, obs_mask]        [B, n_static + 2]
        Mode 2: [c, h̄, y_safe, obs_mask]        [B, n_static + d_model + 2]

        Parameters
        ----------
        static_features : [B, n_static]
        obs             : [B, 1]  observed target (or residual); 0-filled if missing
        obs_mask        : [B, 1]  1 = valid, 0 = missing
        hidden_states   : [B, T, d_model], optional — required for Mode 2
        """
        y_safe   = (obs * obs_mask).detach()        # zero out invalid entries
        obs_mask = obs_mask.detach()
        c        = static_features.detach()

        if self.use_fm_embeddings and hidden_states is not None:
            h_bar = hidden_states.detach().mean(dim=1)   # [B, d_model]
            return torch.cat([c, h_bar, y_safe, obs_mask], dim=-1)

        return torch.cat([c, y_safe, obs_mask], dim=-1)

    @torch.no_grad()
    def _update_bank(
        self,
        static_features: torch.Tensor,
        obs:             torch.Tensor,
        obs_mask:        torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> None:
        """Insert the current batch's entries into the circular bank."""
        new_entries = self._make_bank_entry(
            static_features, obs, obs_mask, hidden_states
        )
        B        = new_entries.size(0)
        capacity = self._bank.size(0)

        if capacity < self.max_bank_size:
            space  = self.max_bank_size - capacity
            to_add = new_entries[:space]
            self._bank = torch.cat([self._bank, to_add], dim=0)
            remaining  = B - space
            if remaining > 0:
                for i in range(remaining):
                    self._bank[self._bank_ptr] = new_entries[space + i]
                    self._bank_ptr = (self._bank_ptr + 1) % self.max_bank_size
                self._bank_full = True
        else:
            for i in range(B):
                self._bank[self._bank_ptr] = new_entries[i]
                self._bank_ptr = (self._bank_ptr + 1) % self.max_bank_size
            self._bank_full = True

    def set_site_bank(
        self,
        static_attrs: torch.Tensor,
        obs:          torch.Tensor,
        obs_mask:     torch.Tensor,
        h_means:      Optional[torch.Tensor] = None,
    ) -> None:
        """
        Pre-populate the site bank from all training sites.
        Call once before training for fastest convergence.

        Parameters
        ----------
        static_attrs : [N_sites, n_static]
        obs          : [N_sites] or [N_sites, 1]
            Observed target value per site (temporal mean, or pre-computed
            residual  r = y − ŷ_global).
        obs_mask     : [N_sites] or [N_sites, 1]
            1 = valid observation, 0 = missing / ungauged.
        h_means      : [N_sites, d_model], optional
            Temporal-mean FM embeddings.  Required when use_fm_embeddings=True.
        """
        if self.use_fm_embeddings and h_means is None:
            raise ValueError(
                "h_means is required when use_fm_embeddings=True. "
                "Pass temporal-mean FM embeddings for each training site."
            )

        dev = self._bank.device
        obs      = obs.view(-1, 1).to(dev)
        obs_mask = obs_mask.view(-1, 1).to(dev)

        entries = self._make_bank_entry(
            static_attrs.to(dev),
            obs,
            obs_mask,
            h_means.to(dev) if h_means is not None else None,
        )
        N = min(entries.size(0), self.max_bank_size)
        self._bank      = entries[:N].detach()
        self._bank_ptr  = N % self.max_bank_size
        self._bank_full = N >= self.max_bank_size
        log.info(
            "Site bank pre-populated: %d / %d sites (obs coverage: %.1f%%)",
            N, entries.size(0),
            100.0 * obs_mask[:N].mean().item(),
        )

    def bank_size(self) -> int:
        """Number of sites currently in the bank."""
        return self._bank.size(0)

    # ------------------------------------------------------------------
    # Attribute gate  α(s₀) — identical to KrigingLocalityAdapter
    # ------------------------------------------------------------------

    def _compute_gate(
        self,
        static_features: torch.Tensor,
        hidden_states:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attribute-based locality gate α ∈ [0,1] via attended Deep Sets.

        Returns
        -------
        alpha : [B, 1, 1]   broadcastable over [B, T, d_model]
        """
        # Strip the two obs scalars at the end to get the attr portion
        bank_attrs = self._bank[:, : self._bank_attr_dim]   # [N, attr_dim]
        N          = bank_attrs.size(0)

        if N == 0:
            return torch.zeros(
                static_features.size(0), 1, 1, device=static_features.device
            )

        # Sub-sample for efficiency
        if N > self.bank_subsample:
            idx        = torch.randperm(N, device=bank_attrs.device)[: self.bank_subsample]
            bank_attrs = bank_attrs[idx]
            N          = self.bank_subsample

        # Bank dropout (training only) — prevents α → 1 for all seen sites
        if self.training and self.bank_dropout > 0.0 and N > 1:
            keep = torch.rand(N, device=bank_attrs.device) > self.bank_dropout
            if keep.sum() >= 1:
                bank_attrs = bank_attrs[keep]

        # Query: c₀ (Mode 1) or [c₀, h̄₀] (Mode 2)
        if self.use_fm_embeddings and hidden_states is not None:
            h_mean      = hidden_states.detach().mean(dim=1)    # [B, d_model]
            query_input = torch.cat([static_features, h_mean], dim=-1)
        else:
            query_input = static_features                        # [B, n_static]

        q = self.phi_gate(query_input)   # [B, key_dim]

        with torch.no_grad():
            k = self.phi_gate(bank_attrs)       # [N, key_dim]
            v = self.value_proj(bank_attrs)     # [N, key_dim]

        scores  = torch.matmul(q, k.t()) / (self.key_dim ** 0.5)   # [B, N]
        weights = F.softmax(scores, dim=-1)                          # [B, N]
        context = torch.matmul(weights, v)                           # [B, key_dim]

        gate_in = torch.cat([q, context], dim=-1)   # [B, 2*key_dim]
        alpha   = self.rho_gate(gate_in)             # [B, 1]
        return alpha.unsqueeze(1)                    # [B, 1, 1]

    # ------------------------------------------------------------------
    # Observation DeepSets branch — the new addition
    # ------------------------------------------------------------------

    def _obs_correction(
        self,
        static_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute observation-based embedding correction dE and density gate g.

        Algorithm
        ---------
        For each bank site i:
            Δᵢ    = [c₀ − cᵢ,  ‖c₀ − cᵢ‖₂]          relative features [B,N,n_static+1]
            wᵢ    = softplus(κ([Δᵢ, mᵢ])) · mᵢ        kernel weight     [B,N,1]
            hᵢ    = ϕ([Δᵢ, yᵢ_safe, mᵢ])              neighbor token    [B,N,phi_out]

        Aggregate:
            H     = Σᵢ wᵢ hᵢ                                              [B,phi_out]
            dE    = ρ(H)                                                   [B,d_model]

        Density gate:
            sum_w = Σᵢ wᵢ                                                  [B,1]
            g     = σ(γ([log(sum_w + ε), sum_w]))                         [B,1]

        Returns
        -------
        dE : [B, d_model]   embedding correction (0 when all w ≈ 0)
        g  : [B, 1]         density gate in [0,1]
        """
        bank = self._bank   # [N, bank_full_dim]
        N    = bank.size(0)
        B    = static_features.size(0)
        dev  = static_features.device

        if N == 0:
            return (
                torch.zeros(B, self.d_model, device=dev),
                torch.zeros(B, 1,            device=dev),
            )

        # Sub-sample
        if N > self.bank_subsample:
            idx  = torch.randperm(N, device=bank.device)[: self.bank_subsample]
            bank = bank[idx]
            N    = self.bank_subsample

        # Bank dropout (training only)
        if self.training and self.bank_dropout > 0.0 and N > 1:
            keep = torch.rand(N, device=bank.device) > self.bank_dropout
            if keep.sum() >= 1:
                bank = bank[keep]
                N    = bank.size(0)

        # Unpack bank: static attrs are always the first n_static columns;
        # y_safe and obs_mask are always the last two.
        c_bank = bank[:, : self.n_static]          # [N, n_static]
        y_bank = bank[:, -2].unsqueeze(-1)          # [N, 1]  y_safe
        m_bank = bank[:, -1].unsqueeze(-1)          # [N, 1]  obs_mask

        # Relative features: Δᵢ = c₀ − cᵢ  and  ‖Δᵢ‖₂
        c0   = static_features.unsqueeze(1)         # [B, 1, n_static]
        cb   = c_bank.unsqueeze(0)                  # [1, N, n_static]
        diff = c0 - cb                              # [B, N, n_static]
        norm = diff.norm(dim=-1, keepdim=True)      # [B, N, 1]
        rel  = torch.cat([diff, norm], dim=-1)      # [B, N, n_static+1]

        # Expand bank obs/mask to [B, N, 1]
        y_exp = y_bank.unsqueeze(0).expand(B, -1, -1)  # [B, N, 1]
        m_exp = m_bank.unsqueeze(0).expand(B, -1, -1)  # [B, N, 1]

        # Kernel weights: κ takes [Δᵢ, mᵢ] — y is intentionally excluded
        # to avoid the weight network itself leaking observation values.
        w_in = torch.cat([rel, m_exp], dim=-1)          # [B, N, n_static+2]
        w    = F.softplus(self.kappa(w_in)) * m_exp     # [B, N, 1]
        # w_i = 0 whenever m_i = 0 (no valid obs at that bank site)

        # Neighbor token: ϕ takes [Δᵢ, yᵢ_safe, mᵢ]
        token = torch.cat([rel, y_exp, m_exp], dim=-1)  # [B, N, n_static+3]
        h     = self.phi_obs(token)                      # [B, N, phi_out]

        # Kernel-weighted permutation-invariant aggregation
        H  = (w * h).sum(dim=1)                          # [B, phi_out]
        dE = self.rho_obs(H)                             # [B, d_model]

        if not self.use_gate:
            return dE, torch.ones(B, 1, device=dev)

        # Density gate — fades correction to 0 in sparse / obs-free regions
        sum_w     = w.sum(dim=1)                          # [B, 1]
        gate_feat = torch.cat(
            [torch.log(sum_w + self.eps), sum_w], dim=-1
        )                                                 # [B, 2]
        g = torch.sigmoid(self.gate_net(gate_feat))       # [B, 1]

        return dE, g

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
        δ(s₀, t) = LayerNorm( MLP([h_FM | project(x_t) | project(c₀)]) )

        Parameters
        ----------
        hidden_states   : [B, T, d_model]
        time_features   : [B, T, n_time]
        static_features : [B, n_static]

        Returns
        -------
        delta : [B, T, d_model]
        """
        T           = hidden_states.size(1)
        time_proj   = self.local_time_proj(time_features)            # [B, T, d_model]
        static_proj = self.local_static_proj(static_features)        # [B, d_model]
        static_proj = static_proj.unsqueeze(1).expand(-1, T, -1)    # [B, T, d_model]

        combined = torch.cat(
            [hidden_states, time_proj, static_proj], dim=-1
        )                                                             # [B, T, 3*d_model]
        delta = self.local_correction(combined)                       # [B, T, d_model]
        return self.correction_norm(delta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states:   torch.Tensor,
        time_features:   torch.Tensor,
        static_features: torch.Tensor,
        obs:             Optional[torch.Tensor] = None,
        obs_mask:        Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states   : [B, T, d_model]
            Foundation model embeddings.
        time_features   : [B, T, n_time]
            Fine-tuning time-series forcings.
        static_features : [B, n_static]
            Fine-tuning static attributes.
        obs             : [B] or [B, 1], optional
            Observed target value at this site (temporal mean, or a pre-computed
            residual  r = y_observed − ŷ_global).
            Pass None to treat the site as completely ungauged — the obs branch
            will contribute zero correction but the attribute branch stays active.
        obs_mask        : [B] or [B, 1], optional
            1 = obs is valid, 0 = missing.
            Inferred as all-ones when obs is provided and obs_mask is None.

        Returns
        -------
        output : [B, T, d_model]

            h_FM  +  α · δ  +  g · dE
            └──────────────┘  └───────┘
            attribute branch  observation branch

        Notes
        -----
        During training the bank is updated from each batch.
        During inference the bank is fixed; g reflects the density of training
        observations near the query site, α reflects attribute similarity.
        """
        B   = hidden_states.size(0)
        dev = hidden_states.device

        # Default obs / obs_mask when caller does not supply them
        if obs is None:
            obs      = torch.zeros(B, 1, device=dev)
            obs_mask = torch.zeros(B, 1, device=dev)
        else:
            obs = obs.view(B, 1).to(dev)
            if obs_mask is None:
                obs_mask = torch.ones(B, 1, device=dev)
            else:
                obs_mask = obs_mask.view(B, 1).to(dev)

        # Update bank during training
        if self.training:
            self._update_bank(
                static_features,
                obs,
                obs_mask,
                hidden_states if self.use_fm_embeddings else None,
            )

        # ── Attribute gate  α ∈ [0, 1] ──────────────────────────────────
        alpha = self._compute_gate(
            static_features,
            hidden_states if self.use_fm_embeddings else None,
        )  # [B, 1, 1]

        # ── Temporal local correction  δ(s₀, t) ─────────────────────────
        delta = self._local_correction(
            hidden_states, time_features, static_features
        )  # [B, T, d_model]

        # ── Observation DeepSets correction  dE, g ──────────────────────
        dE, g = self._obs_correction(static_features)
        # dE: [B, d_model],  g: [B, 1]

        # Expand obs correction over the time axis
        dE_expanded = (g * dE).unsqueeze(1)   # [B, 1, d_model]

        # ── Combine ──────────────────────────────────────────────────────
        # α → 0 : site unseen / attribute-OOD  → attribute branch inactive
        # g → 0 : no valid obs neighbors       → obs branch inactive
        return hidden_states + alpha * delta + dE_expanded

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
        Return attribute gate values α ∈ [0,1] for a batch without
        modifying the bank.  Useful for inspecting seen vs unseen sites.

        Returns
        -------
        alpha : [B]
        """
        was_training = self.training
        self.eval()
        alpha = (
            self._compute_gate(static_features, hidden_states)
            .squeeze(-1).squeeze(-1)
        )
        if was_training:
            self.train()
        return alpha

    @torch.no_grad()
    def density_gate_values(
        self,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return observation density gate values g ∈ [0,1] for a batch
        without modifying the bank.
        g ≈ 1 means many valid nearby observations exist in the bank.
        g ≈ 0 means the site is obs-sparse / ungauged.

        Returns
        -------
        g : [B]
        """
        was_training = self.training
        self.eval()
        _, g = self._obs_correction(static_features)
        if was_training:
            self.train()
        return g.squeeze(-1)

    @torch.no_grad()
    def obs_correction_magnitude(
        self,
        static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return ‖g · dE‖₂ per site — a scalar measuring how strongly the
        observation branch is correcting the foundation model embedding.

        Returns
        -------
        magnitude : [B]
        """
        was_training = self.training
        self.eval()
        dE, g = self._obs_correction(static_features)
        mag = (g * dE).norm(dim=-1)
        if was_training:
            self.train()
        return mag
