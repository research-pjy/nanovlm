#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=mediumq
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --output=logs/nanovlm-evaluate-%j.out
#SBATCH --error=logs/nanovlm-evaluate-%j.err
#
# Evaluate one (SIZE, STRATEGY) checkpoint against the shared 100-image
# held-out set. Only run after all 3 sizes of a strategy are TRAINED
# (DGX_GUIDE_nanovlm.md §4). Example:
#   sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/evaluate.sbatch
#
# Needs the private Ollama server (judge model, llama3:8b) AND the GPU for
# the NanoVLM checkpoint's own generation — full GPU, not a MIG slice, same
# reasoning as generate.sbatch (DGX_GUIDE_nanovlm.md §0).
#
# --mem/--time above are UNVALIDATED placeholders — DESIGN_DECISIONS.md §9.

set -euo pipefail
cd /scratch/cs26d002/repos/nanovlm
mkdir -p logs
source dgx/common.sh

activate_env
trap stop_private_ollama_server EXIT
start_private_ollama_server

: "${SIZE:?SIZE must be set, e.g. --export=ALL,SIZE=base,STRATEGY=conv_on_patches}"
: "${STRATEGY:?STRATEGY must be set, e.g. --export=ALL,SIZE=base,STRATEGY=conv_on_patches}"

CHECKPOINT="checkpoints/nanovlm_${SIZE}_${STRATEGY}/final.pt"

python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --eval-holdout-jsonl data/processed/nanovlm_28k/eval_holdout.jsonl \
    --images-dir /scratch/cs26d002/datasets/coco/images \
    --ollama-host "http://127.0.0.1:${OLLAMA_PRIVATE_PORT}" \
    --judge-model llama3:8b \
    --output "results/nanovlm_${SIZE}_${STRATEGY}_eval.json" \
    --device cuda
