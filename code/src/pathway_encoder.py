"""Learnable structured pathway-bottleneck encoder for Gene2Image.

Replaces GeneFlow's RNA encoder with a biologically structured one while keeping
the rectified-flow + UNet backbone untouched. The encoder maps raw gene expression
to a [B, 512] patch-conditioning vector through three modules:

    A  PathwayMaskEmbedding : genes -> pathway tokens, gated by a fixed binary
       pathway-gene mask, with a learnable D_token vector per non-zero (pathway,
       gene) pair (or frozen ssGSEA weights for the PathPrior ablation).
    B  PathwayTransformer   : self-attention across pathway tokens (pathway
       co-regulation), no positional encoding (pathways are unordered).
    C  CLS aggregation      : a CLS token pools the pathways into a cell embedding;
       its attention over pathways is the interpretability signal (RQ4).

Two entry classes mirror GeneFlow's encoders so they are drop-in replaceable:
    PathwaySingleEncoder(x[B, G])           -> [B, 512]      (single, primary)
    PathwayMultiEncoder(x[B, C_max, G], ...) -> [B, 512]     (multi, secondary;
        reuses GeneFlow's multi-head cell attention for cell aggregation)

Three orthogonal ablation switches map here:
    mask        : real / random-same-density / all-ones  (-> chosen via the mask arg)
    learnable   : True (Gene2Image) / False + init_weight (PathPrior)
    transformer : True / False (noTrans)

Design references: TOSICA (pathway token + CLS), MUPAD (contrast: fixed scoring),
GeneFlow (backbone + cell aggregation reused verbatim). See docs/implementation.md
section 3.1.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module A: mask embedding
# ---------------------------------------------------------------------------
class PathwayMaskEmbedding(nn.Module):
    """Map gene expression to pathway tokens through a fixed sparse mask.

    Implements  t_{p}[r] = sum_{g: A[p,g]=1} W[(p,g)][r] * x[g] + bias[p][r].
    Uses an edge list + index_add_ so the dense [P, G, D] tensor is never built.
    """

    def __init__(self, mask, d_token=48, learnable=True, init_weight=None):
        """
        Args:
            mask: Tensor [P, G] (0/1). Registered as a buffer (saved, not trained).
            d_token: per-pathway token dimension.
            learnable: if False, weights are frozen (PathPrior).
            init_weight: optional Tensor [P, G] of fixed (pathway, gene) weights
                (e.g. ssGSEA). Each edge's scalar weight scales a fixed, seeded,
                zero-mean unit-norm per-edge d_token profile (NOT a plain scalar
                broadcast over d_token: that yields c*ones tokens which the
                transformer's pre-norm LayerNorm annihilates). When given together
                with learnable=False this realises PathPrior.
        """
        super().__init__()
        mask = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
        mask = (mask != 0)
        P, G = mask.shape
        self.P, self.G, self.d_token = int(P), int(G), int(d_token)

        # Edge list of non-zero (pathway, gene) pairs.
        edges = mask.nonzero(as_tuple=False)  # [E, 2] -> (p_idx, g_idx)
        edge_p = edges[:, 0].contiguous()
        edge_g = edges[:, 1].contiguous()
        self.register_buffer("edge_p", edge_p)
        self.register_buffer("edge_g", edge_g)
        self.register_buffer("mask", mask.to(torch.int8))
        self.E = int(edge_p.numel())

        # Learnable (or frozen) weight per edge, and per-pathway bias.
        self.W = nn.Parameter(torch.empty(self.E, self.d_token))
        self.bias = nn.Parameter(torch.zeros(self.P, self.d_token))

        if init_weight is not None:
            init_weight = init_weight if torch.is_tensor(init_weight) else torch.as_tensor(init_weight)
            init_w = init_weight.to(torch.float32)[edge_p, edge_g]  # [E] scalar ssGSEA weight/edge
            # Give each edge a FIXED, zero-mean, unit-norm d_token profile and scale it by
            # the (frozen) ssGSEA scalar, instead of broadcasting the scalar into an
            # all-ones vector. Broadcasting makes every pathway token equal to c*ones,
            # i.e. the whole signal lies in the ones-direction — which the transformer's
            # pre-norm LayerNorm removes (LayerNorm centres each token), collapsing every
            # pathway token to a constant and rendering the PathPrior conditioning
            # input-independent (breaks the RQ3 learnable-vs-fixed ablation). A zero-mean
            # profile survives LayerNorm, so the ssGSEA-weighted expression still reaches
            # the transformer. The profile uses a fixed seed (not the global RNG) so
            # PathPrior is a genuinely fixed, reproducible encoder across run seeds.
            _g = torch.Generator().manual_seed(1234567)
            profile = torch.randn(self.E, self.d_token, generator=_g)
            profile = profile - profile.mean(dim=1, keepdim=True)
            profile = profile / (profile.norm(dim=1, keepdim=True) + 1e-8)
            with torch.no_grad():
                self.W.copy_(init_w.unsqueeze(1) * profile)
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if not learnable:
            self.W.requires_grad_(False)
            self.bias.requires_grad_(False)
        self.learnable = learnable

    def forward(self, x):
        """x: [N, G] -> pathway tokens [N, P, d_token]."""
        N = x.shape[0]
        # Gather expression on each edge's gene, weight it, scatter-add by pathway.
        x_edge = x[:, self.edge_g]                      # [N, E]
        contrib = x_edge.unsqueeze(-1) * self.W.unsqueeze(0)  # [N, E, d_token]
        T = x.new_zeros(N, self.P, self.d_token)        # [N, P, d_token]
        idx = self.edge_p.view(1, self.E, 1).expand(N, self.E, self.d_token)
        T.scatter_add_(1, idx, contrib)
        T = T + self.bias.unsqueeze(0)
        return T

    def l1_penalty(self):
        """L1 over the (pathway, gene) weights -> implicit feature selection."""
        return self.W.abs().sum()


# ---------------------------------------------------------------------------
# Module B + C: pathway transformer with CLS aggregation
# ---------------------------------------------------------------------------
class _CLSAttentionLayer(nn.Module):
    """One pre-norm transformer encoder layer that can return its attention map.

    Equivalent to nn.TransformerEncoderLayer(norm_first=True) but exposes the
    averaged attention weights so the CLS->pathway attention can be read out for
    interpretability (RQ4). batch_first throughout.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # [N, L, L] averaged over heads, filled when need_weights

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=need_weights,
                                     average_attn_weights=True)
        if need_weights:
            self.last_attn = attn_w.detach()
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class PathwayTransformer(nn.Module):
    """Pathway-token self-attention (module B) + CLS pooling (module C)."""

    def __init__(self, d_token=48, n_layers=2, n_heads=8, dropout=0.1,
                 use_transformer=True, d_cell=256):
        super().__init__()
        self.use_transformer = use_transformer
        self.d_token = d_token
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls, std=0.02)

        if use_transformer:
            self.layers = nn.ModuleList([
                _CLSAttentionLayer(d_token, n_heads, 4 * d_token, dropout)
                for _ in range(n_layers)
            ])
            self.final_norm = nn.LayerNorm(d_token)
        else:
            self.layers = None
            self.final_norm = None

        self.proj = nn.Linear(d_token, d_cell)

    def forward(self, T, return_attention=False):
        """T: [N, P, d_token] -> cell embedding [N, d_cell].

        If return_attention, also returns CLS->pathway attention [N, P] from the
        last layer (averaged over heads). With use_transformer=False the pooling
        is a plain mean over pathways and attention is uniform.
        """
        N, P, d = T.shape
        if self.use_transformer:
            cls = self.cls.expand(N, 1, d)
            seq = torch.cat([cls, T], dim=1)        # [N, P+1, d]
            for i, layer in enumerate(self.layers):
                want = return_attention and (i == len(self.layers) - 1)
                seq = layer(seq, need_weights=want)
            seq = self.final_norm(seq)
            h_cls = seq[:, 0]                        # [N, d]
            attn = None
            if return_attention:
                # CLS row over the P pathway columns (drop the CLS self column).
                attn = self.layers[-1].last_attn[:, 0, 1:]  # [N, P]
        else:
            h_cls = T.mean(dim=1)                    # noTrans: mean pooling
            attn = T.new_full((N, P), 1.0 / P) if return_attention else None

        h_cell = self.proj(h_cls)                    # [N, d_cell]
        if return_attention:
            return h_cell, attn
        return h_cell


# ---------------------------------------------------------------------------
# Entry class: single-cell (primary)
# ---------------------------------------------------------------------------
class PathwaySingleEncoder(nn.Module):
    """Single-cell pathway encoder: [B, G] -> [B, output_dim] (=512).

    Drop-in replacement for src.single_model.RNAEncoder; forward signature
    (x, mask=None) is preserved (the mask arg is ignored - the pathway mask is
    injected at construction).
    """

    def __init__(self, mask, output_dim=512, d_token=48, n_layers=2, n_heads=8,
                 d_cell=256, dropout=0.1, learnable=True, use_transformer=True,
                 init_weight=None):
        super().__init__()
        self.embed = PathwayMaskEmbedding(mask, d_token=d_token, learnable=learnable,
                                          init_weight=init_weight)
        self.transformer = PathwayTransformer(
            d_token=d_token, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            use_transformer=use_transformer, d_cell=d_cell)
        self.head = nn.Sequential(
            nn.LayerNorm(d_cell),
            nn.Linear(d_cell, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x, mask=None):
        T = self.embed(x)                 # [B, P, d_token]
        h_cell = self.transformer(T)      # [B, d_cell]
        return self.head(h_cell)          # [B, 512]

    def l1_penalty(self):
        return self.embed.l1_penalty()

    @torch.no_grad()
    def get_pathway_attention(self, x):
        """Return CLS->pathway attention [B, P] for interpretability (RQ4)."""
        T = self.embed(x)
        _, attn = self.transformer(T, return_attention=True)
        return attn


# ---------------------------------------------------------------------------
# Cell aggregator (reused GeneFlow multi-head cell attention) for the multi entry
# ---------------------------------------------------------------------------
class _CellAggregator(nn.Module):
    """GeneFlow's multi-head cell attention, copied so multi stays comparable.

    Mirrors MultiCellRNAEncoder's aggregation block (multi_model.py): per-head
    attention over valid cells (padding masked via num_cells), head-specific
    projections, mean over heads, final projection + feature gate to output_dim.
    """

    def __init__(self, d_cell, output_dim, num_heads=4, dropout=0.1, use_layer_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Sequential(
            nn.LayerNorm(d_cell) if use_layer_norm else nn.Identity(),
            nn.Linear(d_cell, d_cell),
            nn.SiLU(),
            nn.Linear(d_cell, num_heads),
        )
        self.head_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_cell) if use_layer_norm else nn.Identity(),
                nn.Linear(d_cell, d_cell),
                nn.SiLU(),
            ) for _ in range(num_heads)
        ])
        final_layers = []
        if use_layer_norm:
            final_layers.append(nn.LayerNorm(d_cell))
        final_layers += [nn.Linear(d_cell, output_dim), nn.Dropout(dropout)]
        if use_layer_norm:
            final_layers.append(nn.LayerNorm(output_dim))
        self.final = nn.Sequential(*final_layers)
        self.gate = nn.Sequential(nn.Linear(output_dim, output_dim), nn.Sigmoid())

    def forward(self, cell_emb, num_cells=None):
        """cell_emb: [B, C_max, d_cell] -> [B, output_dim]."""
        B, C_max, _ = cell_emb.shape
        cell_mask = cell_emb.new_zeros(B, C_max, 1)
        if num_cells is not None:
            for i, n_c in enumerate(num_cells):
                cell_mask[i, :int(n_c), :] = 1.0
        else:
            cell_mask[:] = 1.0

        logits = self.attn(cell_emb)                          # [B, C_max, H]
        logits = logits.masked_fill(cell_mask == 0, float('-inf'))
        weights = F.softmax(logits.permute(0, 2, 1), dim=2)   # [B, H, C_max]

        head_outs = []
        for h in range(self.num_heads):
            proj = self.head_proj[h](cell_emb)                # [B, C_max, d_cell]
            w = weights[:, h, :].unsqueeze(1)                 # [B, 1, C_max]
            head_outs.append(torch.bmm(w, proj).squeeze(1))   # [B, d_cell]
        agg = torch.stack(head_outs, dim=1).mean(dim=1)       # [B, d_cell]

        out = self.final(agg)
        return out * self.gate(out)


# ---------------------------------------------------------------------------
# Entry class: multi-cell (secondary)
# ---------------------------------------------------------------------------
class PathwayMultiEncoder(nn.Module):
    """Multi-cell pathway encoder: [B, C_max, G] -> [B, output_dim] (=512).

    Drop-in replacement for src.multi_model.MultiCellRNAEncoder; forward signature
    (x, mask=None, num_cells=None) preserved. Per-cell pathway encoding then
    GeneFlow-style multi-head cell aggregation.
    """

    def __init__(self, mask, output_dim=512, d_token=48, n_layers=2, n_heads=8,
                 d_cell=256, dropout=0.1, learnable=True, use_transformer=True,
                 init_weight=None, num_aggregation_heads=4, use_layer_norm=True):
        super().__init__()
        self.embed = PathwayMaskEmbedding(mask, d_token=d_token, learnable=learnable,
                                          init_weight=init_weight)
        self.transformer = PathwayTransformer(
            d_token=d_token, n_layers=n_layers, n_heads=n_heads, dropout=dropout,
            use_transformer=use_transformer, d_cell=d_cell)
        self.aggregator = _CellAggregator(
            d_cell, output_dim, num_heads=num_aggregation_heads,
            dropout=dropout, use_layer_norm=use_layer_norm)
        self.output_dim = output_dim

    def forward(self, x, mask=None, num_cells=None):
        B, C_max, G = x.shape
        x_flat = x.reshape(B * C_max, G)
        T = self.embed(x_flat)                    # [B*C_max, P, d_token]
        h = self.transformer(T)                   # [B*C_max, d_cell]
        cell_emb = h.reshape(B, C_max, -1)        # [B, C_max, d_cell]
        return self.aggregator(cell_emb, num_cells=num_cells)  # [B, 512]

    def l1_penalty(self):
        return self.embed.l1_penalty()

    @torch.no_grad()
    def get_pathway_attention(self, x):
        """Return per-cell CLS->pathway attention [B, C_max, P]."""
        B, C_max, G = x.shape
        T = self.embed(x.reshape(B * C_max, G))
        _, attn = self.transformer(T, return_attention=True)
        return attn.reshape(B, C_max, -1)
