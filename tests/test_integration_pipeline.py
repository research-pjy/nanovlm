"""End-to-end smoke test of the actual CLI scripts (not just the library
functions they call) on a tiny synthetic dataset, entirely on CPU, no
network/Ollama required: train_tokenizer.py -> train.py -> checkpoint is
loadable by evaluate.py's own loader. This is the closest thing to
"does the real pipeline work" this build session can run without a DGX.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_CAPTIONS = [
    "a small dog runs across the green grass near a red ball on a sunny day",
    "two kids sit on a blue bench and share a snack while birds fly above them",
    "a yellow car drives slowly down a quiet street lined with tall green trees",
    "a cat sleeps on a soft pillow while sunlight comes in through the window",
]


def _make_fake_dataset(root: Path, n_images: int = 4):
    images_dir = root / "images" / "train2017"
    images_dir.mkdir(parents=True)

    records = []
    for i in range(n_images):
        fname = f"{i:012d}.jpg"
        img = Image.new("RGB", (64, 64), color=(i * 30 % 255, 100, 150))
        img.save(images_dir / fname)
        records.append(
            {
                "image_id": i,
                "split": "train2017",
                "image_path": f"train2017/{fname}",
                "coco_captions": [FAKE_CAPTIONS[i % len(FAKE_CAPTIONS)]] * 5,
                "caption": FAKE_CAPTIONS[i % len(FAKE_CAPTIONS)],
            }
        )

    data_dir = root / "data" / "processed" / "nanovlm_28k"
    data_dir.mkdir(parents=True)
    # split: first 2 train, next 1 val, last 1 held-out
    with open(data_dir / "train.jsonl", "w") as f:
        for r in records[:2]:
            f.write(json.dumps(r) + "\n")
    with open(data_dir / "val.jsonl", "w") as f:
        for r in records[2:3]:
            f.write(json.dumps(r) + "\n")
    with open(data_dir / "eval_holdout.jsonl", "w") as f:
        for r in records[3:4]:
            f.write(json.dumps(r) + "\n")

    return root / "images", data_dir


@pytest.mark.parametrize("strategy", ["conv_on_patches", "conv_on_image"])
def test_train_tokenizer_then_train_cli_end_to_end(tmp_path, strategy):
    images_dir, data_dir = _make_fake_dataset(tmp_path)
    tokenizer_dir = data_dir / "tokenizer"
    checkpoints_dir = tmp_path / "checkpoints"

    env = {**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)}

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_tokenizer.py"),
            "--train-jsonl", str(data_dir / "train.jsonl"),
            "--out-dir", str(tokenizer_dir),
            "--vocab-size", "300",
        ],
        check=True, cwd=REPO_ROOT, env=env,
    )
    assert (tokenizer_dir / "vocab.json").exists()

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--size", "mini",
            "--patch-embed-strategy", strategy,
            "--data-dir", str(data_dir),
            "--images-dir", str(images_dir),
            "--tokenizer-dir", str(tokenizer_dir),
            "--checkpoints-dir", str(checkpoints_dir),
            "--epochs", "1",
            "--batch-size", "2",
            "--num-workers", "0",
            "--device", "cpu",
        ],
        check=True, cwd=REPO_ROOT, env=env,
    )

    ckpt_path = checkpoints_dir / f"nanovlm_mini_{strategy}" / "final.pt"
    assert ckpt_path.exists()

    import torch

    from nanovlm.config import NanoVLMConfig
    from nanovlm.model.nanovlm import NanoVLM

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = NanoVLMConfig(size=ckpt["config"]["size"], patch_embed_strategy=ckpt["config"]["patch_embed_strategy"])
    cfg.vocab_size = ckpt["config"]["vocab_size"]
    model = NanoVLM(cfg)
    model.load_state_dict(ckpt["model_state_dict"])  # must not raise

    assert ckpt["param_breakdown"]["total"] == model.num_parameters()

    loss_history = json.loads((checkpoints_dir / f"nanovlm_mini_{strategy}" / "loss_history.json").read_text())
    assert len(loss_history) == 1
    assert "train_loss" in loss_history[0] and "val_loss" in loss_history[0]
