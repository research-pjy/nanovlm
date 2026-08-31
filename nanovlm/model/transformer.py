"""Shared transformer block (paper Figure 5), used by both the visual
encoder and the language decoder. Pre-LN: x = x + Attn(LN(x)); x = x +
MLP(LN(x)). See DESIGN_DECISIONS.md §4 for the MLP expansion ratio and
head-sizing decisions (the paper doesn't fully specify either).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention. `causal=True` gives the decoder's
    masked self-attention (paper §2.2.3); `causal=False` gives the encoder's
    full self-attention (paper §2.2.1).
    """

    def __init__(self, embd_dim: int, n_head: int, head_dim: int, dropout: float, causal: bool):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.causal = causal
        inner_dim = n_head * head_dim

        self.qkv = nn.Linear(embd_dim, 3 * inner_dim, bias=False)
        self.proj = nn.Linear(inner_dim, embd_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_head, head_dim)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, embd_dim: int, mlp_ratio: int, dropout: float):
        super().__init__()
        hidden = embd_dim * mlp_ratio
        self.fc1 = nn.Linear(embd_dim, hidden)
        self.fc2 = nn.Linear(hidden, embd_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        n_head: int,
        head_dim: int,
        mlp_ratio: int,
        dropout: float,
        causal: bool,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embd_dim)
        self.attn = MultiHeadAttention(embd_dim, n_head, head_dim, dropout, causal)
        self.ln2 = nn.LayerNorm(embd_dim)
        self.mlp = MLP(embd_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
