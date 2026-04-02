#!/usr/bin/env python3
"""Robotron AI v3 — Set Transformer with ISAB/PMA for entity-set processing.

Architecture:
  1. Entity set → Linear projection → ISAB stack → PMA → global entity repr
  2. Global context (core + ELIST) → MLP → context repr
  3. Temporal stack: concat z_{t-3}..z_t
  4. Fusion MLP → Actor (factored move/fire) + Critic (value)
  5. Auxiliary heads: next-state entity position prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Multihead Attention Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class MultiheadAttention(nn.Module):
    """Standard scaled dot-product multihead attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = query.shape
        _, S, _ = key.shape

        # Project Q from query, K/V from key/value
        q = self._proj(query, 0)          # (B, H, N, d)
        k = self._proj_kv(key, 0)         # (B, H, S, d)
        v = self._proj_kv(value, 1)       # (B, H, S, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, S)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        return self.out_proj(out)

    def _proj(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        return qkv[:, :, idx].permute(0, 2, 1, 3)  # (B, H, N, d)

    def _proj_kv(self, x: torch.Tensor, kv_idx: int) -> torch.Tensor:
        """Project key (kv_idx=0) or value (kv_idx=1) from separate input."""
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        return qkv[:, :, 1 + kv_idx].permute(0, 2, 1, 3)  # (B, H, S, d)


class CrossAttention(nn.Module):
    """Cross-attention: Q from one set, K/V from another."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,     # (B, N, D)
        key_value: torch.Tensor,  # (B, S, D)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = query.shape
        _, S, _ = key_value.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(query).reshape(B, N, H, d).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, S, H, d).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, S, H, d).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # Use a large finite negative value so all-masked rows stay stable.
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════════════════════
# ISAB — Induced Set Attention Block
# ═══════════════════════════════════════════════════════════════════════════════

class ISAB(nn.Module):
    """Induced Set Attention Block.

    Uses M learnable inducing points to reduce self-attention from
    O(N²) to O(NM). Two cross-attention steps:
      1. Inducing points attend to input set  → H = CrossAttn(I, X)
      2. Input set attends to inducing points → Y = CrossAttn(X, H)
    """

    def __init__(self, dim: int, num_heads: int, num_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)

        self.attn1 = CrossAttention(dim, num_heads, dropout)
        self.attn2 = CrossAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        I = self.inducing_points.expand(B, -1, -1)

        # Step 1: inducing points attend to input
        H = self.norm1(I + self.attn1(I, x, mask=mask))
        # Step 2: input attends to inducing points (no mask needed on inducing)
        y = self.norm2(x + self.attn2(x, H))
        # Feed-forward
        y = self.norm3(y + self.ff(y))
        return y


# ═══════════════════════════════════════════════════════════════════════════════
# PMA — Pooling by Multihead Attention
# ═══════════════════════════════════════════════════════════════════════════════

class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Uses K learnable seed vectors to aggregate a set of embeddings
    into K fixed-length output vectors via cross-attention.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
        self.attn = CrossAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        S = self.seeds.expand(B, -1, -1)
        out = self.norm(S + self.attn(S, x, mask=mask))
        out = self.norm2(out + self.ff(out))
        return out  # (B, num_seeds, dim)


# ═══════════════════════════════════════════════════════════════════════════════
# Set Transformer Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class SetTransformerEncoder(nn.Module):
    """Encodes a variable-size entity set into a fixed-length representation.

    Input:  (B, N, feature_dim) entity features + (B, N) padding mask
    Output: (B, embed_dim) global set representation
    """

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 256,
        num_isab_layers: int = 3,
        num_heads: int = 8,
        num_inducing: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.isab_layers = nn.ModuleList([
            ISAB(embed_dim, num_heads, num_inducing, dropout)
            for _ in range(num_isab_layers)
        ])

        self.pool = PMA(embed_dim, num_heads, num_seeds=1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, feature_dim) entity feature vectors
            mask: (B, N) bool tensor, True = padding (ignore)

        Returns:
            (B, embed_dim) global set representation
        """
        h = self.input_proj(x)  # (B, N, embed_dim)
        all_masked = None
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            h = h.masked_fill(mask_exp, 0.0)
            all_masked = mask.all(dim=1)

        for isab in self.isab_layers:
            h = isab(h, mask=mask)
            if mask is not None:
                h = h.masked_fill(mask_exp, 0.0)

        pooled = self.pool(h, mask=mask)  # (B, 1, embed_dim)
        pooled = pooled.squeeze(1)        # (B, embed_dim)
        if all_masked is not None and bool(all_masked.any()):
            pooled = pooled.clone()
            pooled[all_masked] = 0.0
        return pooled


# ═══════════════════════════════════════════════════════════════════════════════
# Full Policy/Value Network
# ═══════════════════════════════════════════════════════════════════════════════

class RobotronPPONet(nn.Module):
    """Full PPO network for Robotron with Set Transformer encoder.

    Combines:
      - Set Transformer over entity pools
      - Global context encoder (core + ELIST features)
      - Temporal stacking (4 frames)
      - Factored actor heads (move + fire) with shared base
      - Value head (critic)
      - Optional auxiliary next-state prediction heads
    """

    def __init__(
        self,
        entity_feature_dim: int = 18,
        max_entities: int = 128,
        embed_dim: int = 256,
        num_isab_layers: int = 1,
        num_heads: int = 8,
        num_inducing: int = 32,
        global_context_dim: int = 40,
        frame_stack: int = 2,
        fusion_hidden: int = 512,
        fusion_layers: int = 2,
        num_move_actions: int = 9,
        num_fire_actions: int = 9,
        use_auxiliary_head: bool = True,
        auxiliary_predict_steps: Optional[list[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_entities = max_entities
        self.entity_feature_dim = entity_feature_dim
        self.embed_dim = embed_dim
        self.frame_stack = frame_stack
        self.global_context_dim = global_context_dim
        self.num_move_actions = num_move_actions
        self.num_fire_actions = num_fire_actions
        self.use_auxiliary_head = use_auxiliary_head

        # Entity set encoder
        self.entity_encoder = SetTransformerEncoder(
            feature_dim=entity_feature_dim,
            embed_dim=embed_dim,
            num_isab_layers=num_isab_layers,
            num_heads=num_heads,
            num_inducing=num_inducing,
            dropout=dropout,
        )

        # Global context encoder (core player features + ELIST)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_context_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # Temporal fusion: frame_stack * (entity_repr + global_repr) → fusion
        temporal_input_dim = frame_stack * (embed_dim + embed_dim)

        # Fusion MLP
        fusion = []
        in_dim = temporal_input_dim
        for _ in range(fusion_layers):
            fusion.extend([
                nn.Linear(in_dim, fusion_hidden),
                nn.LayerNorm(fusion_hidden),
                nn.GELU(),
            ])
            in_dim = fusion_hidden
        self.fusion = nn.Sequential(*fusion)

        # Actor: factored move + fire heads
        self.move_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.GELU(),
            nn.Linear(128, num_move_actions),
        )
        self.fire_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.GELU(),
            nn.Linear(128, num_fire_actions),
        )

        # Critic: value head
        self.value_head = nn.Sequential(
            nn.Linear(fusion_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Auxiliary heads for next-state prediction
        if use_auxiliary_head:
            self.aux_steps = auxiliary_predict_steps or [1, 5]
            self.aux_heads = nn.ModuleDict({
                f"aux_{s}": nn.Sequential(
                    nn.Linear(fusion_hidden, 256),
                    nn.GELU(),
                    nn.Linear(256, max_entities * 2),  # predict (x,y) for each entity slot
                )
                for s in self.aux_steps
            })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy heads: smaller init for stable early training
        for head in [self.move_head, self.fire_head]:
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
            nn.init.zeros_(head[-1].bias)
        # Value head: unit gain
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        entity_features: torch.Tensor,    # (B, N, entity_feature_dim)
        entity_mask: torch.Tensor,         # (B, N) bool: True = padding
        global_context: torch.Tensor,      # (B, frame_stack, global_context_dim)
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass producing policy logits and value estimate.

        The caller is responsible for extracting entities from the raw state
        and stacking frames. This network operates on the processed tensors.

        Returns dict with keys:
          - move_logits: (B, num_move_actions)
          - fire_logits: (B, num_fire_actions)
          - value: (B,)
          - aux_{k}: (B, max_entities, 2) for each auxiliary predict step
        """
        B = entity_features.shape[0]
        T = self.frame_stack

        # entity_features: (B, T, N, feat) → process each frame independently
        if entity_features.dim() == 4:
            BT = B * T
            ef_flat = entity_features.reshape(BT, self.max_entities, self.entity_feature_dim)
            em_flat = entity_mask.reshape(BT, self.max_entities) if entity_mask.dim() == 3 else entity_mask.unsqueeze(1).expand(B, T, self.max_entities).reshape(BT, self.max_entities)
            entity_repr = self.entity_encoder(ef_flat, em_flat)  # (BT, embed_dim)
            entity_repr = entity_repr.reshape(B, T, self.embed_dim)
        else:
            # Single frame: caller handles stacking externally
            entity_repr = self.entity_encoder(entity_features, entity_mask).unsqueeze(1)  # (B, 1, embed_dim)
            entity_repr = entity_repr.expand(B, T, self.embed_dim)

        # Global context per frame
        if global_context.dim() == 2:
            global_context = global_context.unsqueeze(1).expand(B, T, self.global_context_dim)
        global_repr = self.global_encoder(global_context.reshape(B * T, self.global_context_dim))
        global_repr = global_repr.reshape(B, T, self.embed_dim)

        # Temporal concatenation
        temporal = torch.cat([entity_repr, global_repr], dim=-1)  # (B, T, 2*embed_dim)
        temporal = temporal.reshape(B, T * 2 * self.embed_dim)

        # Fusion
        fused = self.fusion(temporal)  # (B, fusion_hidden)

        # Actor heads
        move_logits = self.move_head(fused)
        fire_logits = self.fire_head(fused)

        # Value
        value = self.value_head(fused).squeeze(-1)

        result = {
            "move_logits": move_logits,
            "fire_logits": fire_logits,
            "value": value,
        }

        # Auxiliary predictions
        if self.use_auxiliary_head:
            for s in self.aux_steps:
                aux_out = self.aux_heads[f"aux_{s}"](fused)
                result[f"aux_{s}"] = aux_out.reshape(B, self.max_entities, 2)

        return result

    def get_action_and_value(
        self,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        global_context: torch.Tensor,
        move_action: Optional[torch.Tensor] = None,
        fire_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method for PPO: sample or evaluate actions.

        If move_action/fire_action are None, samples new actions.
        Returns: (move_action, fire_action, log_prob, entropy, value)
        """
        out = self.forward(entity_features, entity_mask, global_context)

        # Clamp logits and replace NaN/Inf with uniform fallback to prevent
        # Categorical from crashing (weights can produce NaN early in training).
        move_logits = out["move_logits"].clamp(-50.0, 50.0)
        fire_logits = out["fire_logits"].clamp(-50.0, 50.0)
        if torch.isnan(move_logits).any() or torch.isinf(move_logits).any():
            move_logits = torch.zeros_like(move_logits)
        if torch.isnan(fire_logits).any() or torch.isinf(fire_logits).any():
            fire_logits = torch.zeros_like(fire_logits)

        move_dist = torch.distributions.Categorical(logits=move_logits)
        fire_dist = torch.distributions.Categorical(logits=fire_logits)

        if move_action is None:
            move_action = move_dist.sample()
        if fire_action is None:
            fire_action = fire_dist.sample()

        log_prob = move_dist.log_prob(move_action) + fire_dist.log_prob(fire_action)
        entropy = move_dist.entropy() + fire_dist.entropy()

        return move_action, fire_action, log_prob, entropy, out["value"]

    def get_value(
        self,
        entity_features: torch.Tensor,
        entity_mask: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """Just the value estimate, for GAE bootstrapping."""
        out = self.forward(entity_features, entity_mask, global_context)
        v = out["value"]
        if torch.isnan(v).any():
            v = torch.zeros_like(v)
        return v
