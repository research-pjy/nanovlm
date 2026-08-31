#!/usr/bin/env python3
"""Stage 1 — generate the LongDesc-style caption dataset via the llama3:8b
teacher (DGX_GUIDE_nanovlm.md §3-4, DESIGN_DECISIONS.md §1).

Text-only: mirrors the paper's own dataset prep (Figure 3) of combining an
image's 5 COCO captions into the teacher prompt (nanovlm/data/prompts.py)
rather than feeding the raw image to the teacher — llama3:8b is a text
model, and this is also what the paper itself describes doing with GPT-4o.

Talks to the job-private Ollama server on port 11435 (DGX_GUIDE_nanovlm.md
§0 — never the ambient CPU-bound daemon on 11434).

Writes incrementally to
    <output-dir>/{train,val,eval_holdout}.jsonl
one line at a time, so a killed/resumed run (or the next self-chained
sbatch segment, DGX_GUIDE_nanovlm.md §4) picks up exactly where it left
off — already-written image_ids are skipped on start, matching the
manifest's train/val/held_out split from scripts/download_coco.py.

If --time-budget-seconds is given, the script stops cleanly (finishing the
in-flight call, not starting a new one) once the deadline is close, rather
than mid-call — the sbatch wrapper is what decides whether to resubmit.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from nanovlm.data.prompts import teacher_caption_prompt
from nanovlm.ollama_client import call_ollama_with_retries

SPLIT_TO_FILE = {"train": "train.jsonl", "val": "val.jsonl", "held_out": "eval_holdout.jsonl"}


def preflight_check_images(manifest: dict, images_dir: Path) -> None:
    """Per DGX_GUIDE_nanovlm.md §6: abort loudly if the manifest's images
    aren't actually on disk, rather than silently generating captions for
    fewer images than requested.
    """
    missing = []
    for split in ("held_out", "train", "val"):
        for rec in manifest[split]:
            path = images_dir / rec["split"] / rec["file_name"]
            if not path.exists():
                missing.append(str(path))
    if missing:
        print(
            f"[preflight] ABORT: {len(missing)} images from the manifest are missing on disk "
            f"(first few: {missing[:5]}). Re-run scripts/download_coco.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    total = sum(len(manifest[s]) for s in ("held_out", "train", "val"))
    print(f"[preflight] all {total} manifest images present on disk")


def already_done(output_dir: Path) -> dict[str, set[int]]:
    done = {}
    for split, fname in SPLIT_TO_FILE.items():
        path = output_dir / fname
        ids = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids.add(json.loads(line)["image_id"])
        done[split] = ids
    return done


def generate_for_record(host: str, model: str, rec: dict, retries: int = 3) -> str:
    prompt = teacher_caption_prompt(rec["coco_captions"])
    try:
        return call_ollama_with_retries(host, model, prompt, retries=retries)
    except RuntimeError as e:
        raise RuntimeError(f"Ollama generation failed for image_id={rec['image_id']}: {e}") from e


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="image_selection.json from download_coco.py")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="e.g. data/processed/nanovlm_28k")
    parser.add_argument("--ollama-host", default="http://localhost:11435")
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--resume", action="store_true", default=True, help="skip already-written records (default: always on)")
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    start = time.monotonic()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight_check_images(manifest, images_dir)
    done = already_done(output_dir)

    split_key_map = {"held_out": "held_out", "train": "train", "val": "val"}
    total_remaining = sum(
        len(manifest[m]) - len(done[m]) for m in split_key_map
    )
    print(f"[generate] {total_remaining} captions remaining across train/val/held_out")

    generated_this_run = 0
    for manifest_split, out_split in split_key_map.items():
        fpath = output_dir / SPLIT_TO_FILE[out_split]
        with open(fpath, "a", encoding="utf-8") as f:
            for rec in manifest[manifest_split]:
                if rec["image_id"] in done[out_split]:
                    continue

                if args.time_budget_seconds is not None:
                    elapsed = time.monotonic() - start
                    if elapsed > args.time_budget_seconds:
                        print(
                            f"[generate] time budget ({args.time_budget_seconds}s) reached, "
                            f"stopping cleanly after {generated_this_run} captions this run"
                        )
                        return

                caption = generate_for_record(args.ollama_host, args.model, rec)
                out_rec = {
                    "image_id": rec["image_id"],
                    "split": rec["split"],
                    "image_path": f"{rec['split']}/{rec['file_name']}",
                    "coco_captions": rec["coco_captions"],
                    "caption": caption,
                }
                f.write(json.dumps(out_rec) + "\n")
                f.flush()
                generated_this_run += 1

                if generated_this_run % args.log_every == 0:
                    elapsed = time.monotonic() - start
                    rate = generated_this_run / elapsed if elapsed > 0 else 0
                    print(
                        f"[generate] {generated_this_run} done this run "
                        f"({elapsed:.0f}s elapsed, {rate:.3f}/s)"
                    )

    print(f"[generate] finished — {generated_this_run} captions generated this run")

    remaining_after = sum(
        len(manifest[m]) - len(already_done(output_dir)[m]) for m in split_key_map
    )
    if remaining_after == 0:
        print("[generate] ALL splits complete — no further chaining needed")
    else:
        print(f"[generate] {remaining_after} captions still remaining — sbatch should resubmit")


if __name__ == "__main__":
    main()
