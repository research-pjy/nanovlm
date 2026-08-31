#!/bin/bash
# Sourced (not executed) by every dgx/*.sbatch script. Conda activation
# pattern and job-private Ollama server helpers, carried forward from
# DGX_GUIDE_nanovlm.md §0/§1.
set -euo pipefail

REPO_DIR="/scratch/cs26d002/repos/nanovlm"
OLLAMA_GPU_BIN="/scratch/cs26d002/software/ollama-gpu/bin/ollama"
OLLAMA_MODELS_DIR="/usr/share/ollama/.ollama/models"  # shared, world-readable store
OLLAMA_PRIVATE_PORT=11435  # never 11434 — that's the ambient CPU-bound daemon

activate_env() {
    . /etc/profile.d/modules.sh
    module load anaconda/2023.03-1
    eval "$(conda shell.bash hook)"
    conda activate /scratch/cs26d002/envs/dgx-research-test
}

# Resolves which ollama binary to run. Loudly warns and falls back to the
# slow CPU-only system binary if the GPU-capable build isn't there yet
# (DGX_GUIDE_nanovlm.md §1 — run dgx/setup_ollama_gpu.sh once on the login
# node first).
resolve_ollama_bin() {
    if [ -x "$OLLAMA_GPU_BIN" ]; then
        echo "$OLLAMA_GPU_BIN"
    else
        echo "[common.sh] WARNING: GPU-capable Ollama not found at $OLLAMA_GPU_BIN." >&2
        echo "[common.sh] WARNING: run dgx/setup_ollama_gpu.sh on the login node first." >&2
        echo "[common.sh] WARNING: falling back to the system 'ollama' binary — THIS HAS NO CUDA BACKEND and will be extremely slow." >&2
        command -v ollama
    fi
}

# Starts a job-private Ollama server on $OLLAMA_PRIVATE_PORT, backgrounded,
# and waits for it to answer before returning. Caller is responsible for
# killing $OLLAMA_SERVER_PID when done (see trap in the calling sbatch
# script) — this is NOT the ambient daemon on 11434.
start_private_ollama_server() {
    local ollama_bin
    ollama_bin=$(resolve_ollama_bin)

    export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
    export OLLAMA_HOST="127.0.0.1:${OLLAMA_PRIVATE_PORT}"

    "$ollama_bin" serve > "logs/ollama-server-${SLURM_JOB_ID:-local}.log" 2>&1 &
    OLLAMA_SERVER_PID=$!
    export OLLAMA_SERVER_PID

    echo "[common.sh] started private Ollama server (pid $OLLAMA_SERVER_PID) on port $OLLAMA_PRIVATE_PORT"

    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:${OLLAMA_PRIVATE_PORT}/api/tags" > /dev/null 2>&1; then
            echo "[common.sh] Ollama server is up"
            return 0
        fi
        sleep 2
    done
    echo "[common.sh] ERROR: Ollama server did not come up after 60s" >&2
    return 1
}

stop_private_ollama_server() {
    if [ -n "${OLLAMA_SERVER_PID:-}" ]; then
        kill "$OLLAMA_SERVER_PID" 2>/dev/null || true
    fi
}
