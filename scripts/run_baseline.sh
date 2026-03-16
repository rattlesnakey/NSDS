#!/usr/bin/env bash
set -euo pipefail

METRIC=${1:-"zd"}
MODEL_PATH=${2:-"meta-llama/Llama-3.1-8B"}
OUTPUT_DIR=${3:-"results"}
DEVICE=${4:-"cuda:0"}
NUM_SAMPLES=${5:-128}
BATCH_SIZE=${6:-8}

VALID_METRICS="zd mse ewq lim llm_mq lsaq lieq kurtboost"
if ! echo "${VALID_METRICS}" | grep -qw "${METRIC}"; then
    echo "[Error] Unknown metric '${METRIC}'. Choose from: ${VALID_METRICS}"
    exit 1
fi

echo "========================================"
echo " Running Baseline: ${METRIC}"
echo " Model      : ${MODEL_PATH}"
echo " Output     : ${OUTPUT_DIR}"
echo " Num Samples: ${NUM_SAMPLES}"
echo "========================================"

python main.py \
    --model_path  "${MODEL_PATH}" \
    --metric      "${METRIC}" \
    --output_dir  "${OUTPUT_DIR}" \
    --device      "${DEVICE}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size  "${BATCH_SIZE}"

echo "[Done] ${METRIC} scores saved to ${OUTPUT_DIR}"
