#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"meta-llama/Llama-3.1-8B"}
OUTPUT_DIR=${2:-"results"}
DEVICE=${3:-"cuda:0"}

echo "========================================"
echo " Running NSDS"
echo " Model : ${MODEL_PATH}"
echo " Output: ${OUTPUT_DIR}"
echo "========================================"

python main.py \
    --model_path  "${MODEL_PATH}" \
    --metric      nsds \
    --output_dir  "${OUTPUT_DIR}" \
    --device      "${DEVICE}"

echo "[Done] NSDS scores saved to ${OUTPUT_DIR}"
