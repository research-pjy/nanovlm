"""Model hyperparameters.

`MODEL_SIZES` is Table 1 of the paper, verbatim. `FIXED_HPARAMS` is
everything the paper (or DESIGN_DECISIONS.md, for gaps the paper leaves
open) fixes across all three sizes and both encoder strategies. See
DESIGN_DECISIONS.md at the repo root for the reasoning behind every value
that isn't directly copied from the paper.
"""

from dataclasses import dataclass, field
from typing import Literal

PatchEmbedStrategy = Literal["conv_on_patches", "conv_on_image"]

# Table 1 — Variable hyperparameters of NanoVLMs (verbatim from the paper).
MODEL_SIZES = {
    "mini": dict(n_blks=1, n_layer=4, n_head=8, head_size=12, n_embd=96, img_embd_dim=400),
    "base": dict(n_blks=3, n_layer=8, n_head=8, head_size=16, n_embd=128, img_embd_dim=512),
    "large": dict(n_blks=5, n_layer=10, n_head=16, head_size=12, n_embd=192, img_embd_dim=512),
}

for _name, _hp in MODEL_SIZES.items():
    assert _hp["n_head"] * _hp["head_size"] == _hp["n_embd"], (
        f"{_name}: n_head * head_size must equal n_embd (Table 1 invariant)"
    )
    assert _hp["img_embd_dim"] % _hp["n_head"] == 0, (
        f"{_name}: img_embd_dim must divide evenly by n_head "
        "(DESIGN_DECISIONS.md §4 — encoder head sizing)"
    )

# Fixed across all sizes and both encoder strategies (paper §2.3 + this
# repo's own filled-in gaps, see DESIGN_DECISIONS.md).
FIXED_HPARAMS = dict(
    dropout=0.1,
    image_size=224,
    patch_size=16,
    learning_rate=1e-3,
    mlp_ratio=4,  # DESIGN_DECISIONS.md §4 — not specified by the paper
    vocab_size=8000,  # DESIGN_DECISIONS.md §5
    epochs=20,  # DESIGN_DECISIONS.md §8
    batch_size=64,  # DESIGN_DECISIONS.md §8
    seed=42,  # DESIGN_DECISIONS.md §7
    num_held_out=100,  # DESIGN_DECISIONS.md §6 (paper used 25)
    caption_word_range=(60, 70),  # LongDesc-style, DESIGN_DECISIONS.md §1
    eval_prompt_word_range=(18, 20),  # partial-text prompt length, §1 / §6
)

# conv_on_patches: per-16x16-patch conv stack (DESIGN_DECISIONS.md §2)
CONV_ON_PATCHES_HPARAMS = dict(
    conv1=dict(out_channels=32, kernel_size=3, stride=1, padding=1),
    conv2=dict(out_channels=64, kernel_size=3, stride=2, padding=1),
)

# conv_on_image: whole-image CNN stem, depth computed from image/patch size
# (DESIGN_DECISIONS.md §3)
CONV_ON_IMAGE_HPARAMS = dict(
    stem=dict(out_channels=32, kernel_size=3, stride=1, padding=1),
    stage_channels=[64, 128, 256, 256],  # one entry per stride-2 layer
    kernel_size=3,
    padding=1,
)


@dataclass
class NanoVLMConfig:
    """Full config for one (size, patch_embed_strategy) checkpoint."""

    size: str
    patch_embed_strategy: PatchEmbedStrategy

    n_blks: int = field(init=False)
    n_layer: int = field(init=False)
    n_head: int = field(init=False)
    head_size: int = field(init=False)
    n_embd: int = field(init=False)
    img_embd_dim: int = field(init=False)

    dropout: float = FIXED_HPARAMS["dropout"]
    image_size: int = FIXED_HPARAMS["image_size"]
    patch_size: int = FIXED_HPARAMS["patch_size"]
    mlp_ratio: int = FIXED_HPARAMS["mlp_ratio"]
    vocab_size: int = FIXED_HPARAMS["vocab_size"]

    def __post_init__(self):
        if self.size not in MODEL_SIZES:
            raise ValueError(f"Unknown size {self.size!r}; expected one of {list(MODEL_SIZES)}")
        if self.patch_embed_strategy not in ("conv_on_patches", "conv_on_image"):
            raise ValueError(
                f"Unknown patch_embed_strategy {self.patch_embed_strategy!r}; "
                "expected 'conv_on_patches' or 'conv_on_image'"
            )
        for key, value in MODEL_SIZES[self.size].items():
            object.__setattr__(self, key, value)

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def encoder_head_dim(self) -> int:
        return self.img_embd_dim // self.n_head

    def checkpoint_dir_name(self) -> str:
        return f"nanovlm_{self.size}_{self.patch_embed_strategy}"

    def results_file_name(self) -> str:
        return f"nanovlm_{self.size}_{self.patch_embed_strategy}_eval.json"
