"""Language decoder (paper §2.2.3): causal transformer over the
[visual_token, text_token_0, ..., text_token_{T-1}] sequence, producing
next-token logits over the vocabulary.
"""

import torch
import torch.nn as nn

from nanovlm.config import NanoVLMConfig
from nanovlm.model.transformer import TransformerBlock


class Decoder(nn.Module):
    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Fixed cap on sequence length (1 visual token + text tokens); the
        # paper's partial-text prompts are short (a few dozen words at
        # most, DESIGN_DECISIONS.md §1/§6), so this is a generous ceiling,
        # not a tuned value. forward() asserts against it explicitly.
        max_positions = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, max_positions, cfg.n_embd))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embd_dim=cfg.n_embd,
                    n_head=cfg.n_head,
                    head_dim=cfg.head_size,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    causal=True,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.max_positions = max_positions

    def forward(self, visual_token: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """visual_token: (B, n_embd). text_ids: (B, T) token ids.
        Returns logits (B, 1 + T, vocab_size); position 0 is the visual
        token's output (excluded from loss — see NanoVLM.forward).
        """
        B, T = text_ids.shape
        seq_len = 1 + T
        assert seq_len <= self.max_positions, (
            f"sequence length {seq_len} exceeds decoder's max_positions={self.max_positions}"
        )

        text_embd = self.token_embed(text_ids)  # (B, T, n_embd)
        x = torch.cat([visual_token.unsqueeze(1), text_embd], dim=1)  # (B, 1+T, n_embd)
        x = x + self.pos_embed[:, :seq_len]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x)
