#!/usr/bin/env python3
"""Generic N-way comparison of scripts/evaluate.py output JSONs.

Run once per size (mini, base, large) per
EXPERIMENT_GUIDE_encoder_ambiguity.md §5 — three comparisons total, plus
worth reading whether any effect's *direction* is consistent across sizes.

    python experiments/encoder_ambiguity/compare_results.py \\
        results/nanovlm_mini_conv_on_patches_eval.json \\
        results/nanovlm_mini_conv_on_image_eval.json \\
        --labels conv_on_patches conv_on_image

Reports, per §6 of the same guide: judge sub-scores side by side, param
counts (never let a score win be mistaken for "just more parameters"),
and ROUGE-1 (diversity check, not a quality signal — flagged as such,
never used to pick a winner here).
"""

import argparse
import json
from pathlib import Path

RUBRIC_DIMENSIONS = ["grammar", "creativity", "consistency", "meaningfulness", "plot"]


def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_table(rows: list[list[str]]) -> None:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="two or more evaluate.py output JSON files")
    parser.add_argument("--labels", nargs="+", default=None, help="labels for each result file, same order")
    args = parser.parse_args()

    if len(args.results) < 2:
        raise SystemExit("need at least 2 result files to compare")

    labels = args.labels or [Path(p).stem for p in args.results]
    if len(labels) != len(args.results):
        raise SystemExit("--labels must have exactly one label per result file")

    results = [load_result(p) for p in args.results]
    sizes = {r["size"] for r in results}
    if len(sizes) > 1:
        print(
            f"WARNING: comparing results across different sizes {sizes} — "
            "EXPERIMENT_GUIDE_encoder_ambiguity.md §5 expects one comparison per size."
        )

    print(f"\n=== Judge scores (0-10, higher is better) — size(s) {sizes} ===")
    header = ["dimension"] + labels
    rows = [header]
    for dim in RUBRIC_DIMENSIONS:
        rows.append([dim] + [f"{r['avg_judge_scores'].get(dim):.2f}" if r["avg_judge_scores"].get(dim) is not None else "N/A" for r in results])
    rows.append(["TOTAL"] + [f"{r['average_total']:.2f}" if r["average_total"] is not None else "N/A" for r in results])
    print_table(rows)

    print("\n=== Parameter count (report alongside scores — a win via more params is a weaker claim) ===")
    rows = [["module"] + labels]
    modules = ["visual_encoder", "multimodal_projector", "decoder", "total"]
    for m in modules:
        rows.append([m] + [f"{r['param_breakdown'][m]:,}" for r in results])
    print_table(rows)

    print("\n=== ROUGE-1 F1 (diversity/memorization check ONLY — not a quality signal, see §6) ===")
    rows = [["metric"] + labels]
    rows.append(["avg_rouge1_f1"] + [f"{r['avg_rouge1_f1']:.4f}" for r in results])
    print_table(rows)

    print("\n=== Judge parse failures (out of held-out set size) ===")
    rows = [["metric"] + labels]
    rows.append(["num_judge_parse_failures"] + [f"{r['num_judge_parse_failures']}/{r['num_held_out']}" for r in results])
    print_table(rows)

    scored = [(l, r["average_total"]) for l, r in zip(labels, results) if r["average_total"] is not None]
    if scored:
        best_label, best_total = max(scored, key=lambda x: x[1])
        totals = [t for _, t in scored]
        spread = max(totals) - min(totals)
        print(f"\n=== Summary ===")
        print(f"Highest average_total: {best_label} ({best_total:.2f}); spread across arms: {spread:.2f}")
        param_note = ", ".join(f"{l}={r['param_breakdown']['total']:,} params" for l, r in zip(labels, results))
        print(f"Param counts: {param_note}")
        print(
            "Reminder: single seed, single teacher=grader model (llama3:8b) — "
            "don't over-read a small gap (EXPERIMENT_GUIDE_encoder_ambiguity.md §6)."
        )


if __name__ == "__main__":
    main()
