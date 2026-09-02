#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=longq
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
# Partition is longq, not mediumq: this account (cs26d002) hit "Invalid
# qos specification" / "Job's QOS not permitted to use this partition"
# on mediumq on this exact cluster in the prior project on this account,
# and only longq was ever confirmed working — even though
# DGX_GUIDE_nanovlm.md §0 lists mediumq as available. --time=11:30:00 is
# well inside longq's 48h cap either way, so this loses nothing if
# mediumq does turn out to work now — flag it if you'd rather try mediumq
# again and confirm it's actually granted.
#
# --mem/--time above are UNVALIDATED placeholders — DESIGN_DECISIONS.md §9,
# size them from a real shortq timing on this repo's actual training code
# before trusting a full 20-epoch run to fit. Pure-PyTorch train jobs were
# untested against MIG slices as of DGX_GUIDE_nanovlm.md §0 — full GPU is
# used here as the safe default, same as generate/evaluate.

set -euo pipefail
# Repo location isn't hardcoded — submit this with `sbatch dgx/train.sbatch`
# from the repo root and Slurm sets SLURM_SUBMIT_DIR to that directory.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit with sbatch from the repo root}"
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
