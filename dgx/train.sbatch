#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=mediumq
#SBATCH --time=11:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --output=logs/nanovlm-train-%j.out
#SBATCH --error=logs/nanovlm-train-%j.err
#
# One (SIZE, STRATEGY) checkpoint. Submit 6 times total — all 3 sizes of
# conv_on_patches (train+eval) before starting conv_on_image
# (DGX_GUIDE_nanovlm.md §4). Example:
#   sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/train.sbatch
#
# --mem/--time above are UNVALIDATED placeholders — DESIGN_DECISIONS.md §9,
# size them from a real shortq timing on this repo's actual training code
# before trusting a full 20-epoch run to fit. Pure-PyTorch train jobs were
# untested against MIG slices as of DGX_GUIDE_nanovlm.md §0 — full GPU is
# used here as the safe default, same as generate/evaluate.

set -euo pipefail
cd /scratch/cs26d002/repos/nanovlm
mkdir -p logs
source dgx/common.sh
activate_env

: "${SIZE:?SIZE must be set, e.g. --export=ALL,SIZE=base,STRATEGY=conv_on_patches}"
: "${STRATEGY:?STRATEGY must be set, e.g. --export=ALL,SIZE=base,STRATEGY=conv_on_patches}"

python scripts/train.py \
    --size "$SIZE" \
    --patch-embed-strategy "$STRATEGY" \
    --images-dir /scratch/cs26d002/datasets/coco/images \
    --data-dir data/processed/nanovlm_28k \
    --checkpoints-dir checkpoints \
    --device cuda
