"""Visual encoder (paper §2.2.1) — the component under test.

Both `ConvOnPatchesEmbed` and `ConvOnImageEmbed` map a `(B, 3, H, W)` image
to `(B, 196, img_embd_dim)` patch tokens. Everything downstream of that
point ([CLS] prepend, positional embedding, transformer blocks, CLS
aggregation) is identical between the two strategies — only the patch
embedding differs, which is the whole point of this experiment (see
EXPERIMENT_GUIDE_encoder_ambiguity.md §3).
"""

import torch
import torch.nn as nn

from nanovlm.config import CONV_ON_IMAGE_HPARAMS, CONV_ON_PATCHES_HPARAMS, NanoVLMConfig
from nanovlm.model.transformer import TransformerBlock


class ConvOnPatchesEmbed(nn.Module):
    """Section 2.2.1's literal text: each 16x16 patch, unfolded and
    processed independently. Conv2D -> ReLU -> Conv2D(stride=2) -> flatten
    -> LayerNorm -> ReLU -> Linear -> img_embd_dim.
    See DESIGN_DECISIONS.md §2 for the conv hyperparameters.
    """

    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        hp = CONV_ON_PATCHES_HPARAMS

        self.conv1 = nn.Conv2d(3, hp["conv1"]["out_channels"], **_conv_kwargs(hp["conv1"]))
        self.conv2 = nn.Conv2d(
            hp["conv1"]["out_channels"], hp["conv2"]["out_channels"], **_conv_kwargs(hp["conv2"])
        )
        self.relu = nn.ReLU()

        # spatial size after conv1 (stride 1, same padding) then conv2 (stride 2)
        post_conv2_size = cfg.patch_size // hp["conv2"]["stride"]
        flat_dim = hp["conv2"]["out_channels"] * post_conv2_size * post_conv2_size

        self.ln = nn.LayerNorm(flat_dim)
        self.fc = nn.Linear(flat_dim, cfg.img_embd_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        p = self.patch_size
        # unfold into non-overlapping p x p patches: (B, num_patches, C, p, p)
        patches = (
            images.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
            .permute(0, 2, 3, 1, 4, 5)
            .contiguous()
        )
        n_h, n_w = patches.shape[1], patches.shape[2]
        num_patches = n_h * n_w
        patches = patches.view(B * num_patches, C, p, p)

        x = self.relu(self.conv1(patches))
        x = self.conv2(x)
        x = x.flatten(1)  # (B*num_patches, flat_dim)
        x = self.relu(self.ln(x))
        x = self.fc(x)  # (B*num_patches, img_embd_dim)

        return x.view(B, num_patches, -1)


class ConvOnImageEmbed(nn.Module):
    """Figure 4's reading: one CNN stem over the whole 224x224 image,
    followed by a stack of stride-2 conv+ReLU layers whose depth is
    computed from config. See DESIGN_DECISIONS.md §3.
    """

    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        hp = CONV_ON_IMAGE_HPARAMS
        num_downsamples = _num_downsamples(cfg.image_size, cfg.patch_size, hp)

        layers = []
        in_ch = 3
        stem = hp["stem"]
        layers.append(nn.Conv2d(in_ch, stem["out_channels"], **_conv_kwargs(stem)))
        layers.append(nn.ReLU())
        in_ch = stem["out_channels"]

        for i in range(num_downsamples):
            out_ch = hp["stage_channels"][min(i, len(hp["stage_channels"]) - 1)]
            layers.append(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=hp["kernel_size"], stride=2, padding=hp["padding"]
                )
            )
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.stem_and_stages = nn.Sequential(*layers)
        self.final_channels = in_ch
        self.token_grid = cfg.image_size // cfg.patch_size  # 224/16 = 14

        # unconditional safety-net pool: no-op when the conv arithmetic
        # already lands on token_grid x token_grid, guarantees the right
        # shape otherwise (DESIGN_DECISIONS.md §3)
        self.pool = nn.AdaptiveAvgPool2d(self.token_grid)

        self.ln = nn.LayerNorm(self.final_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.final_channels, cfg.img_embd_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem_and_stages(images)  # (B, C, H', W')
        x = self.pool(x)  # (B, C, token_grid, token_grid)
        B, C, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)  # (B, num_patches, C)
        x = self.relu(self.ln(x))
        x = self.fc(x)  # (B, num_patches, img_embd_dim)
        return x


def _num_downsamples(image_size: int, patch_size: int, hp: dict) -> int:
    """How many stride-2 layers are needed after the stride-1 stem to go
    from image_size down to image_size/patch_size. For the paper's 224/16
    defaults this is 4 (224->112->56->28->14), matching
    DGX_GUIDE_nanovlm.md's stated arithmetic.
    """
    target = image_size // patch_size
    size = image_size
    n = 0
    while size > target:
        size = size // 2
        n += 1
    return n


def _conv_kwargs(spec: dict) -> dict:
    return dict(kernel_size=spec["kernel_size"], stride=spec["stride"], padding=spec["padding"])


PATCH_EMBED_REGISTRY = {
    "conv_on_patches": ConvOnPatchesEmbed,
    "conv_on_image": ConvOnImageEmbed,
}


class VisualEncoder(nn.Module):
    """Wraps a patch-embed strategy with [CLS] token, positional embedding,
    and n_blks transformer blocks (paper §2.2.1). Everything here is
    identical regardless of which patch-embed strategy is used.
    """

    def __init__(self, cfg: NanoVLMConfig):
        super().__init__()
        embed_cls = PATCH_EMBED_REGISTRY[cfg.patch_embed_strategy]
        self.patch_embed = embed_cls(cfg)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.img_embd_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches + 1, cfg.img_embd_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pre_ln = nn.LayerNorm(cfg.img_embd_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embd_dim=cfg.img_embd_dim,
                    n_head=cfg.n_head,
                    head_dim=cfg.encoder_head_dim,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    causal=False,
                )
                for _ in range(cfg.n_blks)
            ]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns the aggregated [CLS] representation: (B, img_embd_dim)."""
        x = self.patch_embed(images)  # (B, 196, img_embd_dim)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 197, img_embd_dim)
        x = x + self.pos_embed
        x = self.dropout(self.pre_ln(x))

        for block in self.blocks:
            x = block(x)

        return x[:, 0]  # aggregated [CLS] token
