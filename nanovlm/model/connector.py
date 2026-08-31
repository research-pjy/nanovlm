"""Visual-textual connector (paper §2.2.2): a single learnable layer
followed by GELU that projects the encoder's [CLS] representation into the
decoder's text embedding space, producing one visual token that is
prepended to the text token sequence. See DESIGN_DECISIONS.md §4.
"""

import torch
import torch.nn as nn

from nanovlm.config import NanoVLMConfig


class MultimodalProjector(nn.Module):
    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.img_embd_dim, cfg.n_embd)
        self.act = nn.GELU()

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        """cls_repr: (B, img_embd_dim) -> (B, n_embd)."""
        return self.act(self.proj(cls_repr))
