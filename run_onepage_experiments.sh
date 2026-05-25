#!/bin/bash
# run_onepage_experiments.sh
# Run one-page experiments for a Phase-2 defended dataset tag.
#
# Usage:
#   bash run_onepage_experiments.sh padl1_pin0p005_pout0p005_L0_G0
#
# Expected input directory:
#   datasets/CW_tam_<TAG>_pages/
# containing files like:
#   CW_tam_<TAG>_page0.npz
#   CW_tam_<TAG>_page1.npz
#   ...

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash run_onepage_experiments.sh <TAG>"
    echo "Example: bash run_onepage_experiments.sh padl1_pin0p005_pout0p005_L0_G0"
    exit 1
fi

TAG="$1"
IN_BASE="datasets"
PAGES_DIR="${IN_BASE}/CW_tam_${TAG}_pages"
LOG_DIR="logs_onepage/${TAG}"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Running one-page experiments for TAG = ${TAG}"
echo "Input pages directory: ${PAGES_DIR}"
echo "Logs directory: ${LOG_DIR}"
echo "============================================"

if [ ! -d "${PAGES_DIR}" ]; then
    echo "ERROR: ${PAGES_DIR} not found."
    echo "Generate one-page page datasets first."
    exit 1
fi

shopt -s nullglob
PAGE_FILES=("${PAGES_DIR}"/*.npz)

if [ ${#PAGE_FILES[@]} -eq 0 ]; then
    echo "ERROR: No .npz page files found in ${PAGES_DIR}"
    exit 1
fi

for PAGE_FILE in "${PAGE_FILES[@]}"; do
    PAGE_BASENAME=$(basename "${PAGE_FILE}")
    DATASET_NAME="${PAGE_BASENAME%.npz}"   # e.g. CW_tam_padl1_pin0p005_pout0p005_L0_G0_page0

    echo "--------------------------------------------"
    echo "Processing ${DATASET_NAME}"
    echo "--------------------------------------------"

    # Copy for dataset_split.py compatibility
    cp "${PAGE_FILE}" "${IN_BASE}/${DATASET_NAME}.npz"

    # Split
    python exp/dataset_process/dataset_split.py \
        --dataset "${DATASET_NAME}" \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_split.log"

    # Generate TAM for RF
    for split in train valid test; do
        python -u exp/dataset_process/gen_tam.py \
            --dataset "${DATASET_NAME}" \
            --seq_len 5000 \
            --in_file "${split}" \
            2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_tam_${split}.log"
    done

    # DF
    python -u exp/train.py --dataset "${DATASET_NAME}" --model DF \
        --device cuda:0 --feature DIR --seq_len 5000 \
        --train_epochs 30 --batch_size 128 --learning_rate 2e-3 \
        --optimizer Adamax \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score --save_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_DF_train.log"

    python -u exp/test.py --dataset "${DATASET_NAME}" --model DF \
        --device cuda:0 --feature DIR --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_DF_test.log"

    # Tik-Tok
    python -u exp/train.py --dataset "${DATASET_NAME}" --model TikTok \
        --device cuda:0 --feature DT --seq_len 5000 \
        --train_epochs 30 --batch_size 128 --learning_rate 2e-3 \
        --optimizer Adamax \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score --save_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_TikTok_train.log"

    python -u exp/test.py --dataset "${DATASET_NAME}" --model TikTok \
        --device cuda:0 --feature DT --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_TikTok_test.log"

    # Var-CNN
    python -u exp/train.py --dataset "${DATASET_NAME}" --model VarCNN \
        --device cuda:0 --feature DT2 --seq_len 5000 \
        --train_epochs 30 --batch_size 50 --learning_rate 1e-3 \
        --optimizer Adam \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score --save_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_VarCNN_train.log"

    python -u exp/test.py --dataset "${DATASET_NAME}" --model VarCNN \
        --device cuda:0 --feature DT2 --seq_len 5000 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_VarCNN_test.log"

    # RF
    python -u exp/train.py --dataset "${DATASET_NAME}" --model RF \
        --device cuda:0 --train_file tam_train --valid_file tam_valid \
        --feature TAM --seq_len 1800 \
        --train_epochs 30 --batch_size 200 --learning_rate 5e-4 \
        --optimizer Adam \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score --save_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_RF_train.log"

    python -u exp/test.py --dataset "${DATASET_NAME}" --model RF \
        --device cuda:0 --test_file tam_test \
        --feature TAM --seq_len 1800 \
        --batch_size 256 \
        --eval_metrics Accuracy Precision Recall F1-score \
        --load_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_RF_test.log"

done

echo "============================================"
echo "Finished one-page experiments for TAG = ${TAG}"
echo "============================================"
