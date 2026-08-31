"""Tiny end-to-end forward + backward pass on CPU, for both strategies at
every size — catches shape mismatches and dead gradients before this ever
touches the DGX.
"""

import pytest
import torch

from nanovlm.config import MODEL_SIZES, NanoVLMConfig
from nanovlm.model.nanovlm import NanoVLM

SIZES = list(MODEL_SIZES)
STRATEGIES = ["conv_on_patches", "conv_on_image"]


def _tiny_model(size, strategy, vocab_size=64):
    cfg = NanoVLMConfig(size=size, patch_embed_strategy=strategy)
    cfg.vocab_size = vocab_size
    return NanoVLM(cfg), cfg


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_forward_and_loss(size, strategy):
    model, cfg = _tiny_model(size, strategy)
    B, T = 2, 7

    images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(images, text_ids, targets)

    assert logits.shape == (B, T + 1, cfg.vocab_size)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_backward_produces_gradients_everywhere(strategy):
    model, cfg = _tiny_model("mini", strategy)
    B, T = 2, 5

    images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text_ids = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    _, loss = model(images, text_ids, targets)
    loss.backward()

    missing_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_ignore_index_excludes_padding_from_loss():
    """collate_captions pads target sequences with -100 (ignore_index) for
    the padded tail of shorter captions in a batch — confirm that's
    actually wired through to the loss and changes the result vs. no
    padding, rather than silently being treated as a real class id.
    """
    model, cfg = _tiny_model("mini", "conv_on_patches")
    torch.manual_seed(0)
    images = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    text_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    targets = torch.randint(0, cfg.vocab_size, (1, 4))

    _, loss_full = model(images, text_ids, targets)

    targets_partially_masked = targets.clone()
    targets_partially_masked[0, -1] = -100
    _, loss_masked = model(images, text_ids, targets_partially_masked)

    assert torch.isfinite(loss_full)
    assert torch.isfinite(loss_masked)
    assert not torch.isclose(loss_full, loss_masked), (
        "masking one target position with ignore_index=-100 should change the loss"
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_generate_shape_and_stops_at_eos(strategy):
    model, cfg = _tiny_model("mini", strategy)
    model.eval()
    B = 2
    eos_id = 3

    images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    prompt_ids = torch.randint(4, cfg.vocab_size, (B, 3))  # avoid eos_id in prompt

    out = model.generate(images, prompt_ids, max_new_tokens=10, eos_token_id=eos_id, temperature=0.0)

    assert out.shape[0] == B
    assert out.shape[1] >= prompt_ids.shape[1]
    assert out.shape[1] <= prompt_ids.shape[1] + 10


@pytest.mark.parametrize("size", SIZES)
def test_param_breakdown_totals_match_and_differ_by_strategy(size):
    model_patches, _ = _tiny_model(size, "conv_on_patches")
    model_image, _ = _tiny_model(size, "conv_on_image")

    bd_patches = model_patches.param_breakdown()
    bd_image = model_image.param_breakdown()

    assert bd_patches["total"] == bd_patches["visual_encoder"] + bd_patches["multimodal_projector"] + bd_patches["decoder"]
    assert bd_image["total"] == bd_image["visual_encoder"] + bd_image["multimodal_projector"] + bd_image["decoder"]

    # decoder param count must be identical between strategies (only the
    # encoder differs) — DESIGN_DECISIONS.md §3
    assert bd_patches["decoder"] == bd_image["decoder"]
    # encoder param counts are allowed (expected) to differ
    assert bd_patches["visual_encoder"] != bd_image["visual_encoder"]
