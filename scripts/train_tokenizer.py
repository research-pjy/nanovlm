#!/usr/bin/env python3
"""Train the shared tokenizer ONCE on the training split's captions
(DESIGN_DECISIONS.md §5). Run this after scripts/generate_captions.py has
produced train.jsonl and before scripts/train.py for either architecture —
both `conv_on_patches` and `conv_on_image` load the same tokenizer from
--out-dir, never retrain their own.
"""

import argparse
import json
from pathlib import Path

from nanovlm.config import FIXED_HPARAMS
from nanovlm.data.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, help="data/processed/nanovlm_28k/train.jsonl")
    parser.add_argument("--out-dir", required=True, help="e.g. data/processed/nanovlm_28k/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=FIXED_HPARAMS["vocab_size"])
    args = parser.parse_args()

    texts = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["caption"])

    if not texts:
        raise SystemExit(f"No captions found in {args.train_jsonl} — run scripts/generate_captions.py first")

    print(f"[tokenizer] training byte-level BPE, vocab_size={args.vocab_size}, on {len(texts)} training captions")
    train_tokenizer(texts, vocab_size=args.vocab_size, out_dir=args.out_dir)
    print(f"[tokenizer] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
