#!/bin/bash
#SBATCH --job-name=test-gpu-llm
#SBATCH --partition=longq
#SBATCH --qos=longq
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --output=logs/nanovlm-test_ollama_gpu-%j.out
#SBATCH --error=logs/nanovlm-test_ollama_gpu-%j.err
#
# Quick diagnostic (--time=00:10:00 keeps it short regardless of which
# partition it runs on) — run this fresh once and confirm llama3:8b loads
# via the private GPU-capable server BEFORE trusting the real 28K generate
# job to use it. Full GPU, not a MIG slice (MIG slices are confirmed to
# hang Ollama-backed jobs — DGX_GUIDE_nanovlm.md §0).
#
# Root cause of every "Invalid qos specification" failure so far (mediumq,
# then shortq, then longq-with-no-qos) turned out to be the same thing:
# `sacctmgr show assoc where user=cs26d002 format=account,partition,qos`
# shows this account (Slurm account "student") has no default QOS in its
# allowed set {longq, shortq} — --partition alone doesn't set --qos, so
# without an explicit `--qos=` Slurm falls back to a default that isn't
# permitted, regardless of which partition is named. Fix is `--qos=longq`
# above, matching --partition. See DESIGN_DECISIONS.md §9.

set -euo pipefail
# Repo location isn't hardcoded — submit this with `sbatch dgx/test_ollama_gpu.sbatch`
# from the repo root and Slurm sets SLURM_SUBMIT_DIR to that directory.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set — submit with sbatch from the repo root}"
mkdir -p logs
source dgx/common.sh

activate_env
trap stop_private_ollama_server EXIT
start_private_ollama_server

echo "[test_ollama_gpu] timing one llama3:8b call..."
START=$(date +%s.%N)
curl -s http://127.0.0.1:${OLLAMA_PRIVATE_PORT}/api/generate -d '{
  "model": "llama3:8b",
  "prompt": "Say the word OK and nothing else.",
  "stream": false
}' | tee "logs/nanovlm-test_ollama_gpu-${SLURM_JOB_ID}.response.json"
END=$(date +%s.%N)

echo ""
echo "[test_ollama_gpu] elapsed: $(echo "$END - $START" | bc) seconds"
echo "[test_ollama_gpu] compare this against the ~34s/call steady-state figure in DGX_GUIDE_nanovlm.md §3 —"
echo "[test_ollama_gpu] if this single (cold-start) call is wildly different, re-time after a few warmup calls"
echo "[test_ollama_gpu] before trusting --time budgets derived from it (DESIGN_DECISIONS.md §9)."
