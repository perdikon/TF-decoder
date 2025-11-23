import math
import torch
from torch import nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, ctx_dim: int, d_model: int, max_scale: float = 1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, 2 * d_model),
        )
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, gate: float = 1.0):
        if ctx.dim() == 2:
            ctx = ctx[:, None, :]
        gamma, beta = self.mlp(ctx).chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * self.max_scale
        if gamma.size(1) == 1:
            gamma = gamma.expand_as(x)
            beta = beta.expand_as(x)
        return x * (1.0 + gate * gamma) + gate * beta

def make_causal_mask(T, device, dtype=torch.float32):
    mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.masked_fill(mask == 0, 0.0)
    return mask

def make_alibi_bias(n_head, T, device, dtype=torch.float32):
    slopes = 1.0 / (2.0 ** torch.arange(n_head, device=device, dtype=dtype))
    pos = torch.arange(T, device=device)
    dist = (pos[None, :] - pos[:, None]).abs().to(dtype)
    return -dist[None, :, :] * slopes[:, None, None]

class CausalEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout,
                 use_film=False, ctx_dim=0, use_alibi=False):
        super().__init__()
        self.use_film = use_film
        self.use_alibi = use_alibi
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        if use_film:
            self.film1 = FiLM(ctx_dim, d_model)
            self.film2 = FiLM(ctx_dim, d_model)
        self.register_buffer("_alibi_bias", None, persistent=False)

    def forward(self, x, attn_mask, key_padding_mask=None, ctx=None):
        B, T, _ = x.shape
        combined_mask = attn_mask
        if self.use_alibi:
            if (self._alibi_bias is None) or (self._alibi_bias.size(1) != T):
                H = self.self_attn.num_heads
                self._alibi_bias = make_alibi_bias(H, T, x.device, x.dtype)
            base = combined_mask.unsqueeze(0)
            combined_mask = base + self._alibi_bias
        h = self.norm1(x)
        h, _ = self.self_attn(
            h, h, h,
            attn_mask=combined_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        if self.use_film and ctx is not None:
            h = self.film1(h, ctx)
        x = x + self.drop(h)
        h = self.norm2(x)
        h = self.ff(h)
        if self.use_film and ctx is not None:
            h = self.film2(h, ctx)
        x = x + self.drop(h)
        return x

class CausalTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_ff: int,
                 dropout: float, use_film: bool = False, ctx_dim: int = 0, use_alibi: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            CausalEncoderLayer(d_model, nhead, dim_ff, dropout, use_film, ctx_dim, use_alibi)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor,
                key_padding_mask: torch.Tensor = None, ctx: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask, ctx)
        return self.final_norm(x)