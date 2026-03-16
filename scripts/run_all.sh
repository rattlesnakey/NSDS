#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B"}
OUTPUT_DIR=${2:-"results"}
DEVICE=${3:-"cuda:0"}
NUM_SAMPLES=${4:-128}

echo "========================================"
echo " Running ALL metrics"
echo " Model : ${MODEL_PATH}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================"

DATA_FREE_METRICS=("nsds" "zd" "mse" "ewq" "kurtboost" "lieq")
CALIB_METRICS=("lim" "lsaq" "llm_mq")

echo ""
echo "[Phase 1] Data-free metrics..."
for METRIC in "${DATA_FREE_METRICS[@]}"; do
    echo "  --> ${METRIC}"
    python main.py \
        --model_path  "${MODEL_PATH}" \
        --metric      "${METRIC}" \
        --output_dir  "${OUTPUT_DIR}" \
        --device      "${DEVICE}"
done

echo ""
echo "[Phase 2] Calibration-based metrics..."
for METRIC in "${CALIB_METRICS[@]}"; do
    echo "  --> ${METRIC}"
    python main.py \
        --model_path  "${MODEL_PATH}" \
        --metric      "${METRIC}" \
        --output_dir  "${OUTPUT_DIR}" \
        --device      "${DEVICE}" \
        --num_samples "${NUM_SAMPLES}"
done

echo ""
echo "[Done] All results saved to ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"
