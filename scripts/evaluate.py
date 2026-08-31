#!/usr/bin/env python3
"""Evaluate one checkpoint against the shared 100-image held-out set
(DESIGN_DECISIONS.md §6), grading with the llama3:8b judge on the paper's
5-dimension rubric (paper §3, nanovlm/data/prompts.py), plus ROUGE-1 as a
diversity check (nanovlm/metrics.py).

Run only after all 3 sizes of a strategy are trained AND evaluated before
moving to the other strategy (DGX_GUIDE_nanovlm.md §4) — this script
itself just evaluates one checkpoint; the ordering discipline lives in how
you invoke it.

Same held-out 100 images, same partial-text prompt length, same grading
prompt for every one of the 6 checkpoints — nothing here is tuned per
checkpoint.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

from nanovlm.config import NanoVLMConfig
from nanovlm.data.dataset import load_image_for_eval
from nanovlm.data.prompts import RUBRIC_DIMENSIONS, eval_partial_text_prompt, grading_prompt
from nanovlm.data.tokenizer import NanoVLMTokenizer
from nanovlm.model.nanovlm import NanoVLM
from nanovlm.metrics import rouge1_f1
from nanovlm.ollama_client import call_ollama_with_retries


def load_checkpoint(checkpoint_path: str, device: str) -> tuple[NanoVLM, NanoVLMConfig, NanoVLMTokenizer]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = NanoVLMConfig(size=cfg_dict["size"], patch_embed_strategy=cfg_dict["patch_embed_strategy"])
    cfg.vocab_size = cfg_dict["vocab_size"]

    model = NanoVLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = NanoVLMTokenizer.from_dir(ckpt["tokenizer_dir"])
    return model, cfg, tokenizer


def parse_judge_scores(raw_response: str) -> dict[str, float] | None:
    try:
        start = raw_response.index("{")
        end = raw_response.rindex("}") + 1
        scores = json.loads(raw_response[start:end])
        return {dim: float(scores[dim]) for dim in RUBRIC_DIMENSIONS}
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to final.pt")
    parser.add_argument("--eval-holdout-jsonl", default="data/processed/nanovlm_28k/eval_holdout.jsonl")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--ollama-host", default="http://localhost:11435")
    parser.add_argument("--judge-model", default="llama3:8b")
    parser.add_argument("--output", default=None, help="default: results/<checkpoint_dir_name>_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, cfg, tokenizer = load_checkpoint(args.checkpoint, args.device)

    holdout_records = []
    with open(args.eval_holdout_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                holdout_records.append(json.loads(line))

    output_path = Path(args.output) if args.output else Path("results") / cfg.results_file_name()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_image = []
    for i, rec in enumerate(holdout_records, 1):
        image = load_image_for_eval(args.images_dir, rec["image_path"], cfg.image_size).unsqueeze(0).to(args.device)
        partial_text = eval_partial_text_prompt(rec["caption"])
        prompt_ids = torch.tensor(
            [tokenizer.encode(partial_text, add_bos=True, add_eos=False)], device=args.device
        )

        generated = model.generate(
            image, prompt_ids, max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_id
        )
        completion = tokenizer.decode(generated[0].tolist(), skip_special=True)

        rouge = rouge1_f1(completion, rec["caption"])

        judge_prompt = grading_prompt(
            image_context="; ".join(rec["coco_captions"]), partial_prompt=partial_text, completion=completion
        )
        raw_response = call_ollama_with_retries(args.ollama_host, args.judge_model, judge_prompt)
        scores = parse_judge_scores(raw_response)

        per_image.append(
            {
                "image_id": rec["image_id"],
                "partial_text": partial_text,
                "completion": completion,
                "reference_caption": rec["caption"],
                "rouge1_f1": rouge,
                "judge_scores": scores,
                "judge_raw_response": None if scores is not None else raw_response,
            }
        )

        if i % 10 == 0 or i == len(holdout_records):
            print(f"[evaluate] {i}/{len(holdout_records)} held-out images scored")

    valid_scored = [p for p in per_image if p["judge_scores"] is not None]
    n_failed_parse = len(per_image) - len(valid_scored)
    if n_failed_parse:
        print(f"[evaluate] WARNING: {n_failed_parse}/{len(per_image)} judge responses failed to parse as JSON")

    avg_scores = {
        dim: statistics.fmean(p["judge_scores"][dim] for p in valid_scored) if valid_scored else None
        for dim in RUBRIC_DIMENSIONS
    }
    average_total = sum(avg_scores.values()) if all(v is not None for v in avg_scores.values()) else None
    avg_rouge1 = statistics.fmean(p["rouge1_f1"] for p in per_image)

    result = {
        "size": cfg.size,
        "patch_embed_strategy": cfg.patch_embed_strategy,
        "checkpoint": str(args.checkpoint),
        "num_held_out": len(holdout_records),
        "num_judge_parse_failures": n_failed_parse,
        "param_breakdown": model.param_breakdown(),
        "avg_judge_scores": avg_scores,
        "average_total": average_total,
        "avg_rouge1_f1": avg_rouge1,
        "per_image": per_image,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[evaluate] wrote {output_path}")
    print(f"[evaluate] avg_judge_scores={avg_scores} average_total={average_total} avg_rouge1={avg_rouge1:.4f}")
    print(f"[evaluate] param_breakdown={model.param_breakdown()}")


if __name__ == "__main__":
    main()
