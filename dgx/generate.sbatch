#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --time=47:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --output=logs/nanovlm-generate-%j.out
#SBATCH --error=logs/nanovlm-generate-%j.err
#
# --qos=longq is required explicitly alongside --partition=longq — see
# train.sbatch's comment and DESIGN_DECISIONS.md §9: this account's
# allowed QOS set only contains {longq, shortq} with no usable default,
# so --partition alone was never sufficient and every earlier submission
# attempt (mediumq, shortq, longq-with-no-qos) failed the same way.
#
# --mem=16G, down from an earlier 32G placeholder: submitting at 32G left
# this job PD with reason AssocGrpMemLimit (the "student" account's group
# memory quota, shared across every user on that account, was already
# spoken for). This job's workload is the same private-Ollama-server +
# one-call-at-a-time pattern as test_ollama_gpu.sbatch, which already
# succeeded at --mem=16G, so 16G is not a blind guess. If it's still PD
# on AssocGrpMemLimit after this, the block is other users' jobs eating
# the shared quota, not this job's own request — see DESIGN_DECISIONS.md
# §9 for the diagnostic command to confirm the account's actual GrpTRES
# cap rather than guessing further.
# Stage 1 — generate. Self-chains internally: at ~34s/call steady-state
# (DGX_GUIDE_nanovlm.md §3) 28,100 sequential Ollama calls is ~265h,
# which cannot fit in one 48h longq job, so this script runs until close
# to its own time limit, stops cleanly, and resubmits itself if the
# dataset isn't complete yet. That resubmission IS this same stage
# continuing, not a new stage (DGX_GUIDE_nanovlm.md §4) — dgx1's
# one-job-at-a-time limit is respected because the resubmit only happens
# after this job has already exited.
#
# --mem/--time above are UNVALIDATED placeholders — DESIGN_DECISIONS.md §9.
# Run dgx/test_ollama_gpu.sbatch first and size --time from a real
# steady-state timing on today's cluster before trusting this for the
# real 28K run (DGX_GUIDE_nanovlm.md §6 preflight checklist).
#
# Preflight (images actually on disk, matching the manifest) happens
# INSIDE scripts/generate_captions.py itself (preflight_check_images),
# every run — not duplicated here, so it can never be skipped by calling
# the script directly instead of through this sbatch wrapper.

set -euo pipefail
# Repo location isn't hardcoded — submit this with `sbatch dgx/generate.sbatch`
# from the repo root and Slurm sets SLURM_SUBMIT_DIR to that directory.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit with sbatch from the repo root}"
mkdir -p logs
source dgx/common.sh

activate_env
trap stop_private_ollama_server EXIT
start_private_ollama_server

# image_selection.json is written by scripts/download_coco.py next to
# --images-dir's parent (see dgx/generate.sbatch's --selection-out default).
MANIFEST="/scratch/cs26d002/datasets/coco/image_selection.json"
IMAGES_DIR="/scratch/cs26d002/datasets/coco/images"
OUTPUT_DIR="data/processed/nanovlm_28k"

# Leave ~30min buffer under the 47:30:00 SBATCH --time above so the script
# can stop cleanly (finish the in-flight call, not start a new one) rather
# than being SIGKILLed mid-call.
TIME_BUDGET_SECONDS=169200  # 47h

python scripts/generate_captions.py \
    --manifest "$MANIFEST" \
    --images-dir "$IMAGES_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --ollama-host "http://127.0.0.1:${OLLAMA_PRIVATE_PORT}" \
    --model llama3:8b \
    --time-budget-seconds "$TIME_BUDGET_SECONDS"

# Self-chain check: are all three splits complete?
REMAINING=$(python -c "
import json
manifest = json.load(open('$MANIFEST'))
done = {}
for split, fname in [('held_out', 'eval_holdout.jsonl'), ('train', 'train.jsonl'), ('val', 'val.jsonl')]:
    path = '$OUTPUT_DIR/' + fname
    try:
        with open(path) as f:
            done[split] = sum(1 for line in f if line.strip())
    except FileNotFoundError:
        done[split] = 0
remaining = sum(len(manifest[s]) - done[s] for s in ('held_out', 'train', 'val'))
print(remaining)
")

if [ "$REMAINING" -gt 0 ]; then
    echo "[generate.sbatch] $REMAINING captions still remaining — resubmitting (self-chain)"
    sbatch dgx/generate.sbatch
else
    echo "[generate.sbatch] all captions complete — no further chaining"
fi
