#!/bin/bash
# SLURM array job — JointDiffusion on all Binance + Nobitex symbols (ofi and lob modes)
#
# Submit:
#   sbatch scripts/slurm/jointdiff_crypto.sh
#
# Run a subset:
#   sbatch --array=0-3 scripts/slurm/jointdiff_crypto.sh
#
# Override concurrency limit (default 4):
#   sbatch --array=0-21%8 scripts/slurm/jointdiff_crypto.sh

#SBATCH --job-name=jointdiff-crypto
#SBATCH --array=0-21%4
#SBATCH --gres=gpu-all:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# ── index → config mapping ────────────────────────────────────────────────────
# Binance (9 symbols × 2 modes = 18)   indices 0-17
# Nobitex (2 symbols × 2 modes = 4)    indices 18-21
configs=(
  # Binance — ADAUSDT
  "configs/crypto/binance/jointdiff/adausdt_ofi.json"
  "configs/crypto/binance/jointdiff/adausdt_lob.json"
  # Binance — AVAXUSDT
  "configs/crypto/binance/jointdiff/avaxusdt_ofi.json"
  "configs/crypto/binance/jointdiff/avaxusdt_lob.json"
  # Binance — BNBUSDT
  "configs/crypto/binance/jointdiff/bnbusdt_ofi.json"
  "configs/crypto/binance/jointdiff/bnbusdt_lob.json"
  # Binance — BTCUSDT
  "configs/crypto/binance/jointdiff/btcusdt_ofi.json"
  "configs/crypto/binance/jointdiff/btcusdt_lob.json"
  # Binance — DOGEUSDT
  "configs/crypto/binance/jointdiff/dogeusdt_ofi.json"
  "configs/crypto/binance/jointdiff/dogeusdt_lob.json"
  # Binance — ETHUSDT
  "configs/crypto/binance/jointdiff/ethusdt_ofi.json"
  "configs/crypto/binance/jointdiff/ethusdt_lob.json"
  # Binance — SOLUSDT
  "configs/crypto/binance/jointdiff/solusdt_ofi.json"
  "configs/crypto/binance/jointdiff/solusdt_lob.json"
  # Binance — USDCUSDT
  "configs/crypto/binance/jointdiff/usdcusdt_ofi.json"
  "configs/crypto/binance/jointdiff/usdcusdt_lob.json"
  # Binance — XRPUSDT
  "configs/crypto/binance/jointdiff/xrpusdt_ofi.json"
  "configs/crypto/binance/jointdiff/xrpusdt_lob.json"
  # Nobitex — BTCIRT
  "configs/crypto/nobitex/jointdiff/btcirt_ofi.json"
  "configs/crypto/nobitex/jointdiff/btcirt_lob.json"
  # Nobitex — USDTIRT
  "configs/crypto/nobitex/jointdiff/usdtirt_ofi.json"
  "configs/crypto/nobitex/jointdiff/usdtirt_lob.json"
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

uv run python -m crypto.train_jointdiff "$config"
EXIT=$?

echo "======================================================================"
echo "Finished: $(date)  exit=$EXIT"
echo "======================================================================"
exit $EXIT
