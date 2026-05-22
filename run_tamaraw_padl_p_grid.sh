#!/usr/bin/env bash
set -euo pipefail

PADLS=(1 50 100 150 200 300 400 500)
P_VALUES=(0.005 0.010 0.020)   # 5ms, 10ms, 20ms
L_VALUE=0
G_VALUE=0

LOG_DIR="logs_tamaraw"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/padl_p_grid_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "Logging to ${LOG_FILE}"

{
  echo "=================================================="
  echo "Tamaraw PadL x p grid run"
  echo "Start time: $(date)"
  echo "Working directory: $(pwd)"
  echo "PadL values: ${PADLS[*]}"
  echo "p_in = p_out values: ${P_VALUES[*]}"
  echo "L=${L_VALUE}, G=${G_VALUE}"
  echo "=================================================="
  echo

  for PADL in "${PADLS[@]}"; do
    for P in "${P_VALUES[@]}"; do
      echo "=================================================="
      echo "Running: padL=${PADL}, p_in=${P}, p_out=${P}, L=${L_VALUE}, G=${G_VALUE}"
      echo "Start: $(date)"
      echo "--------------------------------------------------"

      python run_tamaraw_CW.py \
        --padl "${PADL}" \
        --p_in "${P}" \
        --p_out "${P}" \
        --L "${L_VALUE}" \
        --G "${G_VALUE}"

      echo "--------------------------------------------------"
      echo "Finished: padL=${PADL}, p_in=${P}, p_out=${P}, L=${L_VALUE}, G=${G_VALUE}"
      echo "End: $(date)"
      echo
    done
  done

  echo "=================================================="
  echo "All PadL x p datasets generated."
  echo "Finish time: $(date)"
  echo "Log file: ${LOG_FILE}"
  echo "=================================================="
} | tee "${LOG_FILE}"
