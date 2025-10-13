import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Mlp
from typing import Optional, Tuple
from einops import einsum, rearrange, repeat
from coda.network.base_arch.transformer.layer import modulate


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # attn_mask: (L, L)
        # key_padding_mask: (B, L)
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1)
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1)
            attn_score = attn_score.masked_fill(key_padding_mask, float("-inf"))

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        output = einsum(attn_score, xv, "b n i j, b n j c -> b n i c")  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output


class EncoderBlockDIT(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6, bias=True),
        )

    def forward(self, x, c, attn_mask=None, tgt_key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self._sa_block(
            modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x
