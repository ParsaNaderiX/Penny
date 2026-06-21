#!/bin/bash
# SLURM array job — DeepLOB on all Binance + Nobitex symbols (ofi and lob modes)
#
# Submit:
#   sbatch scripts/slurm/deeplob_crypto.sh
#
# Run a subset (e.g. jobs 0-3 only):
#   sbatch --array=0-3 scripts/slurm/deeplob_crypto.sh
#
# Override GPU count limit (default 4 concurrent):
#   sbatch --array=0-21%8 scripts/slurm/deeplob_crypto.sh

#SBATCH --job-name=deeplob-crypto
#SBATCH --array=0-21%4
#SBATCH --gres=gpu-all:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# ── index → config mapping ────────────────────────────────────────────────────
# Binance (9 symbols × 2 modes = 18)   indices 0-17
# Nobitex (2 symbols × 2 modes = 4)    indices 18-21
configs=(
  # Binance — ADAUSDT
  "configs/crypto/binance/deeplob/adausdt_ofi.json"
  "configs/crypto/binance/deeplob/adausdt_lob.json"
  # Binance — AVAXUSDT
  "configs/crypto/binance/deeplob/avaxusdt_ofi.json"
  "configs/crypto/binance/deeplob/avaxusdt_lob.json"
  # Binance — BNBUSDT
  "configs/crypto/binance/deeplob/bnbusdt_ofi.json"
  "configs/crypto/binance/deeplob/bnbusdt_lob.json"
  # Binance — BTCUSDT
  "configs/crypto/binance/deeplob/btcusdt_ofi.json"
  "configs/crypto/binance/deeplob/btcusdt_lob.json"
  # Binance — DOGEUSDT
  "configs/crypto/binance/deeplob/dogeusdt_ofi.json"
  "configs/crypto/binance/deeplob/dogeusdt_lob.json"
  # Binance — ETHUSDT
  "configs/crypto/binance/deeplob/ethusdt_ofi.json"
  "configs/crypto/binance/deeplob/ethusdt_lob.json"
  # Binance — SOLUSDT
  "configs/crypto/binance/deeplob/solusdt_ofi.json"
  "configs/crypto/binance/deeplob/solusdt_lob.json"
  # Binance — USDCUSDT
  "configs/crypto/binance/deeplob/usdcusdt_ofi.json"
  "configs/crypto/binance/deeplob/usdcusdt_lob.json"
  # Binance — XRPUSDT
  "configs/crypto/binance/deeplob/xrpusdt_ofi.json"
  "configs/crypto/binance/deeplob/xrpusdt_lob.json"
  # Nobitex — BTCIRT
  "configs/crypto/nobitex/deeplob/btcirt_ofi.json"
  "configs/crypto/nobitex/deeplob/btcirt_lob.json"
  # Nobitex — USDTIRT
  "configs/crypto/nobitex/deeplob/usdtirt_ofi.json"
  "configs/crypto/nobitex/deeplob/usdtirt_lob.json"
)

config="${configs[$SLURM_ARRAY_TASK_ID]}"

# ── environment ───────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }
mkdir -p logs/slurm

echo "======================================================================"
echo "Job:     $SLURM_JOB_ID  array task: $SLURM_ARRAY_TASK_ID"
echo "Node:    $(hostname)"
echo "Config:  $config"
echo "Started: $(date)"
echo "======================================================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

uv run python -m crypto.train_deeplob "$config"
EXIT=$?

echo "======================================================================"
echo "Finished: $(date)  exit=$EXIT"
echo "======================================================================"
exit $EXIT
