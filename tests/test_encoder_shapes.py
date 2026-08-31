"""Both patch-embed strategies must produce identically-shaped output at
every size — that's what makes the downstream comparison (same connector,
same decoder) valid. See DESIGN_DECISIONS.md §2-3.
"""

import pytest
import torch

from nanovlm.config import MODEL_SIZES, NanoVLMConfig
from nanovlm.model.encoder import ConvOnImageEmbed, ConvOnPatchesEmbed, VisualEncoder

SIZES = list(MODEL_SIZES)
STRATEGIES = ["conv_on_patches", "conv_on_image"]


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_patch_embed_output_shape(size, strategy):
    cfg = NanoVLMConfig(size=size, patch_embed_strategy=strategy)
    embed_cls = ConvOnPatchesEmbed if strategy == "conv_on_patches" else ConvOnImageEmbed
    embed = embed_cls(cfg)

    images = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = embed(images)

    assert out.shape == (2, cfg.num_patches, cfg.img_embd_dim), (
        f"{strategy}/{size}: expected (2, {cfg.num_patches}, {cfg.img_embd_dim}), got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_visual_encoder_cls_output_shape(size, strategy):
    cfg = NanoVLMConfig(size=size, patch_embed_strategy=strategy)
    encoder = VisualEncoder(cfg)

    images = torch.randn(3, 3, cfg.image_size, cfg.image_size)
    cls_repr = encoder(images)

    assert cls_repr.shape == (3, cfg.img_embd_dim)


def test_conv_on_image_downsamples_to_correct_grid():
    cfg = NanoVLMConfig(size="base", patch_embed_strategy="conv_on_image")
    embed = ConvOnImageEmbed(cfg)
    assert embed.token_grid == cfg.image_size // cfg.patch_size == 14
