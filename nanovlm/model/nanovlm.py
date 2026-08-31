"""Full NanoVLM: visual encoder -> multimodal projector -> causal decoder.

image, partial_text --> logits over next tokens, and (via generate())
autoregressive text completion conditioned on the image, matching the
paper's evaluation task (§3): "the model then completes the partial text
while attending to the image."
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovlm.config import NanoVLMConfig
from nanovlm.model.connector import MultimodalProjector
from nanovlm.model.decoder import Decoder
from nanovlm.model.encoder import VisualEncoder


class NanoVLM(nn.Module):
    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = VisualEncoder(cfg)
        self.projector = MultimodalProjector(cfg)
        self.decoder = Decoder(cfg)

    def forward(
        self,
        images: torch.Tensor,
        text_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """images: (B, 3, H, W). text_ids: (B, T) input tokens.
        targets: (B, T) next-token targets for text_ids, or None.

        Returns (logits, loss). logits: (B, 1+T, vocab_size). loss is None
        if targets is None, else scalar cross-entropy over the text
        positions only (the visual-token position at index 0 has no
        target and is excluded, per DESIGN_DECISIONS.md §4).
        """
        cls_repr = self.encoder(images)  # (B, img_embd_dim)
        visual_token = self.projector(cls_repr)  # (B, n_embd)
        logits = self.decoder(visual_token, text_ids)  # (B, 1+T, vocab_size)

        loss = None
        if targets is not None:
            # logits[:, :-1] predicts targets[:, :] shifted by the +1 visual
            # offset: logits at position i (0-indexed, 0=visual) predicts
            # text_ids[i] i.e. targets[i-1]. Concretely: text-position
            # logits are logits[:, :-1] (dropping the last prediction, which
            # has no target beyond the sequence) aligned to targets.
            text_logits = logits[:, :-1, :]  # (B, T, vocab_size) predicting targets
            loss = F.cross_entropy(
                text_logits.reshape(-1, text_logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy/temperature/top-k autoregressive completion of
        prompt_ids, conditioned on images. Returns (B, T_prompt + N) token
        ids, N <= max_new_tokens (stops early per-sequence at eos but keeps
        batch shape by continuing to append eos after it's hit).
        """
        self.eval()
        cls_repr = self.encoder(images)
        visual_token = self.projector(cls_repr)

        ids = prompt_ids
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

        for _ in range(max_new_tokens):
            seq_len = 1 + ids.size(1)
            if seq_len > self.decoder.max_positions:
                break
            logits = self.decoder(visual_token, ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) if temperature > 0 else probs.argmax(
                dim=-1, keepdim=True
            )

            if eos_token_id is not None:
                next_id = torch.where(
                    finished.unsqueeze(1), torch.full_like(next_id, eos_token_id), next_id
                )
                finished = finished | (next_id.squeeze(1) == eos_token_id)

            ids = torch.cat([ids, next_id], dim=1)
            if eos_token_id is not None and finished.all():
                break

        return ids

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def param_breakdown(self) -> dict:
        """Parameter count per module, matching the paper's Table 2 shape
        (visual encoder / multimodal projector / decoder), reported
        per-checkpoint since conv_on_image and conv_on_patches are NOT
        guaranteed to match in encoder param count (DESIGN_DECISIONS.md §3).
        """
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        encoder = count(self.encoder)
        projector = count(self.projector)
        decoder = count(self.decoder)
        total = encoder + projector + decoder
        return {
            "visual_encoder": encoder,
            "multimodal_projector": projector,
            "decoder": decoder,
            "total": total,
        }
