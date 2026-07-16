"""TinySTR — minimal transformer recognizer for fixed-length captchas.

Patch embed → ViT encoder → 5 learnable position queries via cross-attention
→ shared per-position 62-class head. Outputs [B, 5, 62] logits (no EOS, no
autoregression). See ../CONTEXT.md for design rationale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUM_CLASSES = len(CHARSET)  # 62
NUM_POSITIONS = 5


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TinySTR(nn.Module):
    def __init__(
        self,
        img_h: int = 60,
        img_w: int = 200,
        patch_h: int = 4,
        patch_w: int = 8,
        in_channels: int = 3,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_positions: int = NUM_POSITIONS,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.0,
        query_self_attn_layers: int = 0,
    ):
        super().__init__()
        assert img_h % patch_h == 0, f"img_h={img_h} not divisible by patch_h={patch_h}"
        assert img_w % patch_w == 0, f"img_w={img_w} not divisible by patch_w={patch_w}"

        self.img_h, self.img_w = img_h, img_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.grid_h = img_h // patch_h
        self.grid_w = img_w // patch_w
        self.num_patches = self.grid_h * self.grid_w  # 60/4 * 200/8 = 15 * 25 = 375
        self.embed_dim = embed_dim
        self.num_positions = num_positions
        self.num_classes = num_classes

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, encoder_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.position_queries = nn.Parameter(torch.zeros(num_positions, embed_dim))

        self.query_self_attn = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, encoder_heads, mlp_ratio, dropout)
            for _ in range(query_self_attn_layers)
        ])

        self.decoder_norm_q = nn.LayerNorm(embed_dim)
        self.decoder_norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, encoder_heads, dropout=dropout, batch_first=True)
        self.decoder_norm_mlp = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.position_queries, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 60, 200] image batch in [0, 1] (normalize upstream).

        Returns logits [B, 5, 62] — one softmax per position over the charset.
        """
        B = x.size(0)

        feats = self.patch_embed(x)               # [B, D, 15, 25]
        feats = feats.flatten(2).transpose(1, 2)  # [B, 375, D]
        feats = feats + self.pos_embed

        for block in self.encoder:
            feats = block(feats)
        feats = self.encoder_norm(feats)

        queries = self.position_queries.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, 5, D]
        for layer in self.query_self_attn:
            queries = layer(queries)
        q = self.decoder_norm_q(queries)
        kv = self.decoder_norm_kv(feats)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x_dec = queries + attn_out
        x_dec = x_dec + self.decoder_mlp(self.decoder_norm_mlp(x_dec))

        logits = self.head(self.head_norm(x_dec))  # [B, 5, 62]
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Convenience: returns (decoded strings, per-position max-prob confidences)."""
        logits = self.forward(x)
        probs = logits.softmax(dim=-1)
        confs, idx = probs.max(dim=-1)  # [B, 5], [B, 5]
        strs = ["".join(CHARSET[i] for i in row.tolist()) for row in idx]
        return strs, confs


def decode(logits: torch.Tensor) -> list[str]:
    """Greedy-decode logits [B, 5, 62] -> list of 5-char strings."""
    idx = logits.argmax(dim=-1)
    return ["".join(CHARSET[i] for i in row.tolist()) for row in idx]
