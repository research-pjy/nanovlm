#!/usr/bin/env python3
"""Train one (size, patch_embed_strategy) checkpoint.

Run 6 times total (3 sizes x 2 strategies), all conv_on_patches before any
conv_on_image, all 3 sizes before moving to eval — DGX_GUIDE_nanovlm.md §4.

Same seed, same data, same tokenizer, same epoch/batch budget for every
run regardless of strategy (DESIGN_DECISIONS.md §7-8) — the only thing
that should differ between two runs at the same size is
--patch-embed-strategy itself.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nanovlm.config import FIXED_HPARAMS, NanoVLMConfig
from nanovlm.data.dataset import CaptionJsonlDataset, collate_captions
from nanovlm.data.tokenizer import NanoVLMTokenizer
from nanovlm.model.nanovlm import NanoVLM


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss, n_batches = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, input_ids, targets in loader:
            images, input_ids, targets = images.to(device), input_ids.to(device), targets.to(device)
            _, loss = model(images, input_ids, targets)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=["mini", "base", "large"])
    parser.add_argument("--patch-embed-strategy", required=True, choices=["conv_on_patches", "conv_on_image"])
    parser.add_argument("--data-dir", default="data/processed/nanovlm_28k")
    parser.add_argument("--images-dir", required=True, help="root images dir, e.g. /scratch/.../datasets/coco/images")
    parser.add_argument("--tokenizer-dir", default=None, help="default: <data-dir>/tokenizer")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=FIXED_HPARAMS["epochs"])
    parser.add_argument("--batch-size", type=int, default=FIXED_HPARAMS["batch_size"])
    parser.add_argument("--lr", type=float, default=FIXED_HPARAMS["learning_rate"])
    parser.add_argument("--seed", type=int, default=FIXED_HPARAMS["seed"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir or str(Path(args.data_dir) / "tokenizer")
    set_seed(args.seed)

    tokenizer = NanoVLMTokenizer.from_dir(tokenizer_dir)
    cfg = NanoVLMConfig(size=args.size, patch_embed_strategy=args.patch_embed_strategy)
    cfg.vocab_size = tokenizer.vocab_size

    set_seed(args.seed)  # re-seed immediately before model init (DESIGN_DECISIONS.md §7)
    model = NanoVLM(cfg).to(args.device)
    print(f"[train] {args.size}/{args.patch_embed_strategy} param breakdown: {model.param_breakdown()}")

    train_ds = CaptionJsonlDataset(
        str(Path(args.data_dir) / "train.jsonl"), args.images_dir, tokenizer, cfg.image_size
    )
    val_ds = CaptionJsonlDataset(
        str(Path(args.data_dir) / "val.jsonl"), args.images_dir, tokenizer, cfg.image_size
    )

    def collate(batch):
        return collate_captions(batch, pad_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoints_dir) / cfg.checkpoint_dir_name()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        train_loss = run_epoch(model, train_loader, optimizer, args.device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, args.device, train=False)
        dt = time.monotonic() - t0
        print(f"[train] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": dt})

        with open(ckpt_dir / "loss_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "size": cfg.size,
                "patch_embed_strategy": cfg.patch_embed_strategy,
                "vocab_size": cfg.vocab_size,
            },
            "param_breakdown": model.param_breakdown(),
            "tokenizer_dir": tokenizer_dir,
            "seed": args.seed,
        },
        ckpt_dir / "final.pt",
    )
    print(f"[train] saved {ckpt_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
