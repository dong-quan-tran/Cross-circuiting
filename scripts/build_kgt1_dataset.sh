#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./scripts/build_kgt1_dataset.sh \
#   EXPERIMENT_ID MODE K ARRIVAL_MODE STAGGER_SECONDS \
#   N_OUT N_IN PADDING_PER_ACTIVE_BUCKET \
#   [DELTA_T] [SEQ_LEN] [SEED] [NUM_MIXED]
#
# Example:
# ./scripts/build_kgt1_dataset.sh \
#   SMOKE_K2_FIXED5S_BOUNDED_N3_P1 bounded 2 fixed 5.0 \
#   3 3 1 0.01 10000 2024 100

EXPERIMENT_ID="${1:?experiment ID is required}"
MODE="${2:?mode is required}"
K="${3:?K is required}"
ARRIVAL_MODE="${4:?arrival mode is required}"
STAGGER_SECONDS="${5:?stagger seconds is required}"
N_OUT="${6:?N_out is required}"
N_IN="${7:?N_in is required}"
PADDING_PER_ACTIVE_BUCKET="${8:?padding-per-active-bucket is required}"
DELTA_T="${9:-0.01}"
SEQ_LEN="${10:-10000}"
SEED="${11:-2024}"
NUM_MIXED="${12:-}"

case "${MODE}" in
  concurrent|scheduled|full|bounded)
    ;;
  *)
    echo "ERROR: unsupported MODE='${MODE}'." >&2
    exit 2
    ;;
esac

if [[ "${K}" -lt 2 ]]; then
  echo "ERROR: K must be at least 2; got ${K}." >&2
  exit 2
fi

if [[ "${N_OUT}" -lt 1 || "${N_IN}" -lt 1 ]]; then
  echo "ERROR: N_OUT and N_IN must both be at least 1." >&2
  exit 2
fi

if [[ "${PADDING_PER_ACTIVE_BUCKET}" -lt 0 ]]; then
  echo "ERROR: PADDING_PER_ACTIVE_BUCKET must be nonnegative." >&2
  exit 2
fi

echo "Building ${EXPERIMENT_ID}"
echo "  mode=${MODE}, K=${K}, arrival=${ARRIVAL_MODE}, stagger=${STAGGER_SECONDS}s"
echo "  dt=${DELTA_T}s, N_out=${N_OUT}, N_in=${N_IN}, P=${PADDING_PER_ACTIVE_BUCKET}"
echo "  seq_len=${SEQ_LEN}, seed=${SEED}, num_mixed=${NUM_MIXED:-source-count}"

for SPLIT in train valid test; do
  EXTRA_ARGS=()

  if [[ -n "${NUM_MIXED}" ]]; then
    EXTRA_ARGS+=(--num_mixed "${NUM_MIXED}")
  fi

  python -u src/build_kgt1_dataset.py \
    --input_path "datasets/CW/${SPLIT}.npz" \
    --output_path "datasets/${EXPERIMENT_ID}/${SPLIT}.npz" \
    --metadata_path "datasets/${EXPERIMENT_ID}/${SPLIT}_meta.npz" \
    --K "${K}" \
    --mode "${MODE}" \
    --padding_per_active_bucket "${PADDING_PER_ACTIVE_BUCKET}" \
    --arrival_mode "${ARRIVAL_MODE}" \
    --stagger_seconds "${STAGGER_SECONDS}" \
    --delta_t "${DELTA_T}" \
    --N_out "${N_OUT}" \
    --N_in "${N_IN}" \
    --seq_len "${SEQ_LEN}" \
    --seed "${SEED}" \
    --progress_every 100 \
    "${EXTRA_ARGS[@]}"
done
