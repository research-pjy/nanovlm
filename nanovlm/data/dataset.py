"""JSONL caption dataset + collate function.

Expected JSONL record shape (one per line), written by
scripts/generate_captions.py:

    {
        "image_id": 179765,
        "split": "train2017" | "val2017",
        "image_path": "images/train2017/000000179765.jpg",   # relative to --images-dir
        "coco_captions": ["...", "...", "...", "...", "..."],
        "caption": "<generated LongDesc-style description>"
    }

Training uses the standard "prefix LM captioner" setup: the whole caption
is fed through the decoder autoregressively, conditioned on the prepended
visual token (§2.2.3 — "generate coherent text ... when provided with both
an image and partial text as input" is exactly next-token prediction given
image + text-so-far, of which the full caption is the T=len(caption) case
and eval's partial-text prompt is a shorter prefix of the same objective).
No separate "partial captioning" training signal is needed; see
scripts/evaluate.py for how held-out completion is scored.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from nanovlm.data.tokenizer import NanoVLMTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _image_to_tensor(img: Image.Image, image_size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).clone()
    arr = arr.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (arr - mean) / std


class CaptionJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        tokenizer: NanoVLMTokenizer,
        image_size: int,
        max_caption_tokens: int = 128,
    ):
        self.records = _read_jsonl(jsonl_path)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_caption_tokens = max_caption_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(self.images_dir / rec["image_path"])
        image_tensor = _image_to_tensor(img, self.image_size)

        ids = self.tokenizer.encode(rec["caption"], add_bos=True, add_eos=True)
        ids = ids[: self.max_caption_tokens]

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return image_tensor, input_ids, target_ids


def collate_captions(batch, pad_id: int):
    images, input_ids, target_ids = zip(*batch)
    images = torch.stack(images)

    max_len = max(x.size(0) for x in input_ids)
    B = len(batch)

    padded_input = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_target = torch.full((B, max_len), -100, dtype=torch.long)  # -100 = ignore_index
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        padded_input[i, : inp.size(0)] = inp
        padded_target[i, : tgt.size(0)] = tgt

    return images, padded_input, padded_target


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_image_for_eval(images_dir: str, image_path: str, image_size: int) -> torch.Tensor:
    img = Image.open(Path(images_dir) / image_path)
    return _image_to_tensor(img, image_size)
