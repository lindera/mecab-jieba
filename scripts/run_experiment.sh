#!/usr/bin/env bash
# Run the full training pipeline: train -> export -> build -> evaluate
#
# Prerequisites:
#   - lindera CLI (cargo install --path lindera-cli --features train)
#   - Python 3.10+ with no extra packages
#   - UD Chinese GSD test data at .tmp/ud-chinese/UD_Chinese-GSD/
#   - Training data prepared in work/train/ (see README.md)
#
# Usage:
#   bash scripts/run_experiment.sh <experiment_name>
#
# Environment variables (override defaults):
#   FEATURE_DEF          - path to feature.def (default: work/train/feature.def)
#   CHAR_DEF             - path to char.def (default: work/train/char.def)
#   UNK_DEF              - path to unk.def (default: work/train/unk.def)
#   CORPUS               - path to corpus.txt (default: work/train/corpus.txt)
#   LAMBDA               - regularization coefficient (default: 0.01)
#   MAX_ITER             - max training iterations (default: 100)
#   REGULARIZATION       - l1, l2, or elasticnet (default: l1)
#   ELASTIC_NET_L1_RATIO - L1 ratio for elastic net (default: 0.5, range: 0.0-1.0)
#
# Examples:
#   # Default L1 training
#   bash scripts/run_experiment.sh baseline
#
#   # Elastic Net with custom L1 ratio
#   REGULARIZATION=elasticnet ELASTIC_NET_L1_RATIO=0.8 \
#     bash scripts/run_experiment.sh elasticnet_r08
#
#   # Custom feature template and lambda
#   FEATURE_DEF=work/experiments/my_feat/feature.def LAMBDA=0.005 \
#     bash scripts/run_experiment.sh my_experiment

set -euo pipefail

EXPERIMENT_NAME="${1:?Usage: $0 <experiment_name>}"

# Project root
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${ROOT}/work"
TRAIN="${WORK}/train"
EXP_DIR="${WORK}/experiments/${EXPERIMENT_NAME}"

# Defaults (overridable via env vars)
FEATURE_DEF="${FEATURE_DEF:-${TRAIN}/feature.def}"
CHAR_DEF="${CHAR_DEF:-${TRAIN}/char.def}"
UNK_DEF="${UNK_DEF:-${TRAIN}/unk.def}"
CORPUS="${CORPUS:-${TRAIN}/corpus.txt}"
LAMBDA="${LAMBDA:-0.01}"
MAX_ITER="${MAX_ITER:-100}"
REGULARIZATION="${REGULARIZATION:-l1}"
ELASTIC_NET_L1_RATIO="${ELASTIC_NET_L1_RATIO:-0.5}"

# Fixed paths
SEED="${TRAIN}/seed.csv"
REWRITE_DEF="${TRAIN}/rewrite.def"
MODEL="${TRAIN}/model.dat"
EXPORT_DIR="${WORK}/export"
DICT_DIR="${WORK}/dict"
METADATA="${WORK}/build/metadata.json"
TEST_FILE="${ROOT}/.tmp/ud-chinese/UD_Chinese-GSD/zh_gsd-ud-test.conllu"
JIEBA_DICT="${ROOT}/dict.txt.big"

echo "============================================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "============================================================"
echo "  FEATURE_DEF : ${FEATURE_DEF}"
echo "  CHAR_DEF    : ${CHAR_DEF}"
echo "  UNK_DEF     : ${UNK_DEF}"
echo "  CORPUS      : ${CORPUS}"
echo "  LAMBDA      : ${LAMBDA}"
echo "  MAX_ITER    : ${MAX_ITER}"
echo "  REGULARIZATION: ${REGULARIZATION}"
echo "  ELASTIC_NET_L1_RATIO: ${ELASTIC_NET_L1_RATIO}"
echo "============================================================"

# Create experiment directory and save config copies
mkdir -p "${EXP_DIR}"
cp -f "${FEATURE_DEF}" "${EXP_DIR}/feature.def" 2>/dev/null || true
cp -f "${CHAR_DEF}" "${EXP_DIR}/char.def" 2>/dev/null || true
cp -f "${UNK_DEF}" "${EXP_DIR}/unk.def" 2>/dev/null || true
echo "lambda=${LAMBDA}" > "${EXP_DIR}/params.txt"
echo "max_iter=${MAX_ITER}" >> "${EXP_DIR}/params.txt"
echo "regularization=${REGULARIZATION}" >> "${EXP_DIR}/params.txt"
echo "elastic_net_l1_ratio=${ELASTIC_NET_L1_RATIO}" >> "${EXP_DIR}/params.txt"

# Step 1: Train
echo ""
echo "[1/4] Training CRF model..."
time lindera train \
  --seed "${SEED}" \
  --corpus "${CORPUS}" \
  --char-def "${CHAR_DEF}" \
  --unk-def "${UNK_DEF}" \
  --feature-def "${FEATURE_DEF}" \
  --rewrite-def "${REWRITE_DEF}" \
  --output "${MODEL}" \
  --lambda "${LAMBDA}" \
  --regularization "${REGULARIZATION}" \
  --elastic-net-l1-ratio "${ELASTIC_NET_L1_RATIO}" \
  --max-iterations "${MAX_ITER}"

# Step 2: Export
echo ""
echo "[2/4] Exporting dictionary..."
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}"
lindera export \
  --model "${MODEL}" \
  --output "${EXPORT_DIR}" \
  --metadata "${METADATA}"

# Copy char.def to export dir (lindera build needs it from source dir)
cp "${CHAR_DEF}" "${EXPORT_DIR}/char.def"

# Step 3: Build
echo ""
echo "[3/4] Building compiled dictionary..."
rm -rf "${DICT_DIR}"
mkdir -p "${DICT_DIR}"
lindera build \
  --src "${EXPORT_DIR}" \
  --dest "${DICT_DIR}" \
  --metadata "${EXPORT_DIR}/metadata.json"

# Step 4: Evaluate
echo ""
echo "[4/4] Evaluating..."
python3 "${ROOT}/scripts/evaluate.py" \
  --test-file "${TEST_FILE}" \
  --dict-dir "${DICT_DIR}" \
  --jieba-dict "${JIEBA_DICT}" \
  2>&1 | tee "${EXP_DIR}/result.txt"

echo ""
echo "Results saved to: ${EXP_DIR}/result.txt"
echo "Done: ${EXPERIMENT_NAME}"
