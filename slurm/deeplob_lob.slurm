#!/bin/bash
#SBATCH --job-name=penny-deeplob-lob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-all

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT_DIR"

echo "Penny DeepLOB LOB | host=$(hostname) | start=$(date)"

if command -v module >/dev/null 2>&1; then
  module purge || true
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate"
fi

# cu118: Pascal/Volta clusters (P100 sm_60, V100 sm_70)
# cu126: Ampere/Ada clusters (A100, RTX 30xx/40xx)
UV_EXTRA="${UV_EXTRA:-cu126}"
CONFIG="${CONFIG:-configs/crypto/deeplob/btcusdt_lob.json}"
uv sync --extra "$UV_EXTRA"
uv run python -m crypto.deeplob.train "$CONFIG"

echo "done: $(date)"
