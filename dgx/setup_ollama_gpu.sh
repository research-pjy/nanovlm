#!/bin/bash
# One-time setup — run on the DGX LOGIN node (needs internet), per
# DGX_GUIDE_nanovlm.md §1. Downloads the official Ollama release (bundles
# CUDA backend libs; the system 'ollama' binary has none) into
# /scratch/cs26d002/software/ollama-gpu/.
set -euo pipefail

INSTALL_DIR="/scratch/cs26d002/software/ollama-gpu"
OLLAMA_RELEASE_URL="https://ollama.com/download/ollama-linux-amd64.tgz"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

if [ -x "$INSTALL_DIR/bin/ollama" ]; then
    echo "[setup_ollama_gpu] already installed at $INSTALL_DIR/bin/ollama"
    "$INSTALL_DIR/bin/ollama" --version
    exit 0
fi

echo "[setup_ollama_gpu] downloading $OLLAMA_RELEASE_URL ..."
curl -fsSL "$OLLAMA_RELEASE_URL" -o ollama-linux-amd64.tgz
tar -xzf ollama-linux-amd64.tgz
rm ollama-linux-amd64.tgz

echo "[setup_ollama_gpu] installed to $INSTALL_DIR"
"$INSTALL_DIR/bin/ollama" --version

echo "[setup_ollama_gpu] done. Verify with dgx/test_ollama_gpu.sbatch on shortq before trusting a real job to it."
