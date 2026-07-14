#!/bin/bash
# run_onepage_varcnn_only.sh
# Run VarCNN only for a Phase-2 one-page defended dataset tag.
#
# Usage:
#   bash run_onepage_varcnn_only.sh <TAG>
#
# Expected input directory:
#   datasets/CW_tam_<TAG>_pages/
# containing files like:
#   CW_tam_<TAG>_page0.npz
#   CW_tam_<TAG>_page1.npz
#   ...

set -u

if [ $# -lt 1 ]; then
    echo "Usage: bash run_onepage_varcnn_only.sh <TAG>"
    echo "Example: bash run_onepage_varcnn_only.sh padl1_pin0p005_pout0p005_L0_G0"
    exit 1
fi

TAG="$1"
IN_BASE="datasets"
PAGES_DIR="${IN_BASE}/CW_tam_${TAG}_pages"
LOG_DIR="logs_onepage_varcnn/${TAG}"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "Running VarCNN-only one-page experiments for TAG = ${TAG}"
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
    DATASET_NAME="${PAGE_BASENAME%.npz}"   # e.g. CW_tam_mix_K4_deltat0p01_N10_page0

    ROOT_PAGE_NPZ="${IN_BASE}/${DATASET_NAME}.npz"
    PAGE_WORK_DIR="${IN_BASE}/${DATASET_NAME}"
    PAGE_ARCHIVE_DIR="${PAGES_DIR}/${DATASET_NAME}"

    echo "--------------------------------------------"
    echo "Processing ${DATASET_NAME} (VarCNN only)"
    echo "--------------------------------------------"

    # Copy for dataset_split.py compatibility
    cp "${PAGE_FILE}" "${ROOT_PAGE_NPZ}"

    # If split/train/test already exist from DF/TikTok/RF, you can skip dataset_split.py
    # But re-running is safe; it will just overwrite train/valid/test.npz.
    python exp/dataset_process/dataset_split.py \
        --dataset "${DATASET_NAME}" \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_split.log" || true

    # We only need tam_* files for RF, so VarCNN doesn't require gen_tam.py.
    # Skipping TAM generation here.

    # VarCNN - batch_size 256 / test 512 (reduced to avoid CUDA OOM)
    python -u exp/train.py --dataset "${DATASET_NAME}" --model VarCNN \
        --device cuda:0 --feature DT2 --seq_len 5000 \
        --train_epochs 30 --batch_size 256 --learning_rate 1e-3 \
        --optimizer Adam \
        --eval_metrics Accuracy Precision Recall F1-score \
        --save_metric F1-score --save_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_VarCNN_train.log" || true

    python -u exp/test.py --dataset "${DATASET_NAME}" --model VarCNN \
        --device cuda:0 --feature DT2 --seq_len 5000 \
        --batch_size 512 \
        --eval_metrics Accuracy Precision Recall F1-score TPR FPR \
        --load_name max_f1 \
        2>&1 | tee "${LOG_DIR}/${DATASET_NAME}_VarCNN_test.log" || true

    # Move the generated per-page working directory into the *_pages archive,
    # but only if dataset_split.py created one.
    if [ -d "${PAGE_WORK_DIR}" ]; then
        rm -rf "${PAGE_ARCHIVE_DIR}"
        mv "${PAGE_WORK_DIR}" "${PAGES_DIR}/"
    else
        echo "WARNING: Expected working directory ${PAGE_WORK_DIR} was not found."
    fi

    rm -f "${ROOT_PAGE_NPZ}"

    echo "Finished VarCNN for ${DATASET_NAME}"
    echo
done

echo "============================================"
echo "Finished VarCNN-only one-page experiments for TAG = ${TAG}"
echo "============================================"
