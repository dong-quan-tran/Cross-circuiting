#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ID="${1:-LIVE_K2_FIXED5S_FULL}"
MODE="${2:-full}"
K="${3:-2}"
ARRIVAL_MODE="${4:-fixed}"
STAGGER_SECONDS="${5:-5.0}"

DELTA_T="0.01"
N_OUT="1"
N_IN="1"
SEQ_LEN="10000"
SEED="2024"

for SPLIT in train valid test; do
  python -u src/build_kgt1_dataset.py \
    --input_path "datasets/CW/${SPLIT}.npz" \
    --output_path "datasets/${EXPERIMENT_ID}/${SPLIT}.npz" \
    --metadata_path "datasets/${EXPERIMENT_ID}/${SPLIT}_meta.npz" \
    --K "${K}" \
    --mode "${MODE}" \
    --arrival_mode "${ARRIVAL_MODE}" \
    --stagger_seconds "${STAGGER_SECONDS}" \
    --delta_t "${DELTA_T}" \
    --N_out "${N_OUT}" \
    --N_in "${N_IN}" \
    --seq_len "${SEQ_LEN}" \
    --seed "${SEED}" \
    --progress_every 100
done
