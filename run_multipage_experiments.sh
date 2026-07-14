#!/bin/bash
# run_multipage_experiments.sh
# Run closed-world multi-page experiments for one defended dataset.
#
# Usage:
#   bash run_multipage_experiments.sh <DATASET_NAME>
#
# Examples:
#   bash run_multipage_experiments.sh CW_mix_K4_deltat0p01_N3
#   bash run_multipage_experiments.sh CW_mix_K4_deltat0p01_N2
#   bash run_multipage_experiments.sh CW_tam_padl1_pin0p005_pout0p005_L0_G0
#
# Expected input file:
#   datasets/<DATASET_NAME>.npz
#
# Behavior:
#   1) Runs dataset_split.py on datasets/<DATASET_NAME>.npz
#   2) Generates TAM features for RF from train/valid/test
#   3) Trains/tests DF, TikTok, VarCNN, RF
#   4) Stores logs in logs_multipage/<DATASET_NAME>/

set -u

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

if [ $# -lt 1 ]; then
    echo "Usage: bash run_multipage_experiments.sh <DATASET_NAME>"
    echo "Example: bash run_multipage_experiments.sh CW_mix_K4_deltat0p01_N3"
    exit 1
fi

DATASET_NAME="$1"
IN_BASE="datasets"
DATASET_NPZ="${IN_BASE}/${DATASET_NAME}.npz"
DATASET_DIR="${IN_BASE}/${DATASET_NAME}"
LOG_DIR="logs_multipage/${DATASET_NAME}"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Running multi-page experiments for DATASET = ${DATASET_NAME}"
echo "Input dataset file: ${DATASET_NPZ}"
echo "Dataset working dir: ${DATASET_DIR}"
echo "Logs directory: ${LOG_DIR}"
echo "============================================"

if [ ! -f "${DATASET_NPZ}" ]; then
    echo "ERROR: ${DATASET_NPZ} not found."
    exit 1
fi

# Split
python exp/dataset_process/dataset_split.py \
    --dataset "${DATASET_NAME}" \
    2>&1 | tee "${LOG_DIR}/split.log" || true

# Generate TAM for RF
for split in train valid test; do
    python -u exp/dataset_process/gen_tam.py \
        --dataset "${DATASET_NAME}" \
        --seq_len 5000 \
        --in_file "${split}" \
        2>&1 | tee "${LOG_DIR}/tam_${split}.log" || true
done

# DF
python -u exp/train.py --dataset "${DATASET_NAME}" --model DF \
    --device cuda:0 --feature DIR --seq_len 5000 \
    --train_epochs 30 --batch_size 512 --learning_rate 2e-3 \
    --optimizer Adamax \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score --save_name max_f1 \
    2>&1 | tee "${LOG_DIR}/DF_train.log" || true

python -u exp/test.py --dataset "${DATASET_NAME}" --model DF \
    --device cuda:0 --feature DIR --seq_len 5000 \
    --batch_size 1024 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    2>&1 | tee "${LOG_DIR}/DF_test.log" || true

# TikTok
python -u exp/train.py --dataset "${DATASET_NAME}" --model TikTok \
    --device cuda:0 --feature DT --seq_len 5000 \
    --train_epochs 30 --batch_size 512 --learning_rate 2e-3 \
    --optimizer Adamax \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score --save_name max_f1 \
    2>&1 | tee "${LOG_DIR}/TikTok_train.log" || true

python -u exp/test.py --dataset "${DATASET_NAME}" --model TikTok \
    --device cuda:0 --feature DT --seq_len 5000 \
    --batch_size 1024 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    2>&1 | tee "${LOG_DIR}/TikTok_test.log" || true

# VarCNN
python -u exp/train.py --dataset "${DATASET_NAME}" --model VarCNN \
    --device cuda:0 --feature DT2 --seq_len 5000 \
    --train_epochs 30 --batch_size 512 --learning_rate 1e-3 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score --save_name max_f1 \
    2>&1 | tee "${LOG_DIR}/VarCNN_train.log" || true

python -u exp/test.py --dataset "${DATASET_NAME}" --model VarCNN \
    --device cuda:0 --feature DT2 --seq_len 5000 \
    --batch_size 1024 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    2>&1 | tee "${LOG_DIR}/VarCNN_test.log" || true

# RF
python -u exp/train.py --dataset "${DATASET_NAME}" --model RF \
    --device cuda:0 --train_file tam_train --valid_file tam_valid \
    --feature TAM --seq_len 1800 \
    --train_epochs 30 --batch_size 512 --learning_rate 5e-4 \
    --optimizer Adam \
    --eval_metrics Accuracy Precision Recall F1-score \
    --save_metric F1-score --save_name max_f1 \
    2>&1 | tee "${LOG_DIR}/RF_train.log" || true

python -u exp/test.py --dataset "${DATASET_NAME}" --model RF \
    --device cuda:0 --test_file tam_test \
    --feature TAM --seq_len 1800 \
    --batch_size 1024 \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    2>&1 | tee "${LOG_DIR}/RF_test.log" || true

echo "============================================"
echo "Finished multi-page experiments for DATASET = ${DATASET_NAME}"
echo "============================================"
