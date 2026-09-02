#!/usr/bin/env python3
"""Selective COCO downloader — run on the DGX LOGIN node (has internet),
once, per DGX_GUIDE_nanovlm.md §1.

Downloads only the specific train2017/val2017 images this experiment
needs (roughly 2-3GB for 28,100 images) instead of the full ~18GB
train2017 zip. Downloads concurrently (--workers, default 16 — network
round-trip latency, not bandwidth, is the bottleneck for many small
files, so this matters a lot: a single-threaded version of this took
~1.5 days to get ~9,500/28,100 images in practice). Skip-existing, so
re-running after an interruption or a lower/higher --workers value picks
up exactly where it left off.

Selection (DESIGN_DECISIONS.md §6/§7, EXPERIMENT_GUIDE_encoder_ambiguity.md
§1):
  1. Build the candidate pool: every image (across the requested splits)
     that has all 5 COCO captions.
  2. Deterministically shuffle the pool with the fixed seed.
  3. Held-out set (--num-held-out, default 100) is taken FIRST, before the
     --num-images selection — so held-out images can never leak into
     train/val regardless of what --num-images is.
  4. The next --num-images images become the train/val pool, split 90/10.

Writes a selection manifest to
  <output-dir>/image_selection.json
so scripts/generate_captions.py (and anything else) reuses the exact same
split rather than re-deriving it — single source of truth for "which
image is in which split," reused identically by both architecture runs.

Example (matches DGX_GUIDE_nanovlm.md §1):
    python scripts/download_coco.py \\
        --splits train2017 val2017 \\
        --num-images 28000 \\
        --seed 42 \\
        --workers 16 \\
        --annotations-dir /scratch/cs26d002/datasets/coco/annotations \\
        --images-dir /scratch/cs26d002/datasets/coco/images
"""

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COCO_ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/{split}/{file_name}"

DEFAULT_NUM_HELD_OUT = 100  # DESIGN_DECISIONS.md §6
DEFAULT_TRAIN_FRACTION = 0.9  # paper's 90/10 split, §2.1.1


def ensure_annotations(annotations_dir: Path, splits: list[str]) -> None:
    annotations_dir.mkdir(parents=True, exist_ok=True)
    needed = [annotations_dir / f"captions_{s}.json" for s in splits]
    if all(p.exists() for p in needed):
        print(f"[annotations] already present in {annotations_dir}")
        return

    zip_path = annotations_dir / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print(f"[annotations] downloading {COCO_ANNOTATIONS_ZIP_URL} ...")
        urllib.request.urlretrieve(COCO_ANNOTATIONS_ZIP_URL, zip_path)

    print("[annotations] extracting captions_*.json ...")
    with zipfile.ZipFile(zip_path) as zf:
        for split in splits:
            member = f"annotations/captions_{split}.json"
            with zf.open(member) as src, open(annotations_dir / f"captions_{split}.json", "wb") as dst:
                dst.write(src.read())
    zip_path.unlink()


def load_candidate_pool(annotations_dir: Path, splits: list[str]) -> dict[int, dict]:
    """Returns {image_id: {file_name, split, coco_captions: [str, ...]}}
    for every image that has at least 5 captions, across all requested
    splits.
    """
    images_by_id: dict[int, dict] = {}
    captions_by_id: dict[int, list[str]] = {}

    for split in splits:
        path = annotations_dir / f"captions_{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        for img in coco["images"]:
            images_by_id[img["id"]] = {
                "file_name": img["file_name"],
                "split": split,
            }
        for ann in coco["annotations"]:
            captions_by_id.setdefault(ann["image_id"], []).append(ann["caption"])

    pool = {}
    for image_id, meta in images_by_id.items():
        caps = captions_by_id.get(image_id, [])
        if len(caps) >= 5:
            pool[image_id] = {**meta, "coco_captions": caps[:5]}
    return pool


def make_selection(
    pool: dict[int, dict], num_images: int, num_held_out: int, seed: int, train_fraction: float
) -> dict[str, list[int]]:
    ids = sorted(pool.keys())  # sort first so shuffle is reproducible regardless of dict order
    rng = random.Random(seed)
    rng.shuffle(ids)

    if num_held_out + num_images > len(ids):
        raise ValueError(
            f"candidate pool has {len(ids)} images with 5 captions, but "
            f"num_held_out({num_held_out}) + num_images({num_images}) = "
            f"{num_held_out + num_images} were requested"
        )

    held_out = ids[:num_held_out]
    selected = ids[num_held_out : num_held_out + num_images]

    n_train = round(len(selected) * train_fraction)
    train_ids = selected[:n_train]
    val_ids = selected[n_train:]

    return {"held_out": held_out, "train": train_ids, "val": val_ids}


def _fetch_one(image_id: int, meta: dict, images_dir: Path) -> str:
    """Returns 'downloaded', 'skipped', or 'failed'. Runs in a worker
    thread — pure I/O (urlretrieve), no shared mutable state touched here.
    """
    split_dir = images_dir / meta["split"]
    split_dir.mkdir(parents=True, exist_ok=True)
    dest = split_dir / meta["file_name"]

    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    url = COCO_IMAGE_URL_TEMPLATE.format(split=meta["split"], file_name=meta["file_name"])
    # unique temp suffix per image_id so concurrent workers never collide
    # on the same .part path
    tmp_dest = dest.with_suffix(dest.suffix + f".{image_id}.part")
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp_dest)
            tmp_dest.rename(dest)
            return "downloaded"
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                print(f"[download] FAILED {url}: {e}", file=sys.stderr)
                return "failed"
            time.sleep(1.5 * (attempt + 1))
    return "failed"  # unreachable, keeps type-checkers happy


def download_selected(
    pool: dict[int, dict], selection: dict[str, list[int]], images_dir: Path, workers: int = 16
) -> tuple[int, int, int]:
    """Concurrent, skip-existing, resumable download. A single-threaded
    sequential version of this (one urlretrieve at a time) is what made
    the first real run of this script take ~1.5 days for ~9,500/28,100
    images — network round-trip latency dominates, not bandwidth, so
    downloading many images in parallel (default 16 worker threads, matching
    the concurrency level that got a full run down to ~30 minutes) fixes
    the actual bottleneck. Safe to interrupt and resume: already-downloaded
    files are skipped on the next run either way.
    """
    all_ids = selection["held_out"] + selection["train"] + selection["val"]
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        futures = {
            pool_executor.submit(_fetch_one, image_id, pool[image_id], images_dir): image_id
            for image_id in all_ids
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                counts[result] += 1
                completed += 1
                if completed % 500 == 0 or completed == len(all_ids):
                    print(
                        f"[download] {completed}/{len(all_ids)} "
                        f"(downloaded={counts['downloaded']} skipped={counts['skipped']} failed={counts['failed']})"
                    )

    return counts["downloaded"], counts["skipped"], counts["failed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    parser.add_argument("--num-images", type=int, default=28000, help="train+val pool size (excludes held-out)")
    parser.add_argument("--num-held-out", type=int, default=DEFAULT_NUM_HELD_OUT)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument(
        "--workers", type=int, default=16, help="concurrent download threads (default 16)"
    )
    parser.add_argument(
        "--selection-out",
        default=None,
        help="where to write image_selection.json (default: <images-dir>/../image_selection.json)",
    )
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ensure_annotations(annotations_dir, args.splits)
    pool = load_candidate_pool(annotations_dir, args.splits)
    print(f"[pool] {len(pool)} candidate images with >=5 captions across {args.splits}")

    selection = make_selection(pool, args.num_images, args.num_held_out, args.seed, args.train_fraction)
    print(
        f"[selection] held_out={len(selection['held_out'])} "
        f"train={len(selection['train'])} val={len(selection['val'])} "
        f"(seed={args.seed})"
    )

    selection_out = Path(args.selection_out) if args.selection_out else images_dir.parent / "image_selection.json"
    selection_out.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": args.seed,
        "num_images": args.num_images,
        "num_held_out": args.num_held_out,
        "train_fraction": args.train_fraction,
        "splits": args.splits,
        "held_out": [{"image_id": i, **{k: pool[i][k] for k in ("file_name", "split", "coco_captions")}} for i in selection["held_out"]],
        "train": [{"image_id": i, **{k: pool[i][k] for k in ("file_name", "split", "coco_captions")}} for i in selection["train"]],
        "val": [{"image_id": i, **{k: pool[i][k] for k in ("file_name", "split", "coco_captions")}} for i in selection["val"]],
    }
    with open(selection_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"[selection] wrote manifest to {selection_out}")

    downloaded, skipped, failed = download_selected(pool, selection, images_dir, workers=args.workers)
    total = downloaded + skipped
    expected = args.num_held_out + args.num_images
    print(f"[done] {total}/{expected} images on disk (downloaded={downloaded} skipped_existing={skipped} failed={failed})")
    if failed:
        print(f"[done] {failed} images failed to download after retries — re-run this script to resume/retry.", file=sys.stderr)
        sys.exit(1)
    if total < expected:
        print(f"[done] WARNING: only {total} of {expected} expected images present.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
