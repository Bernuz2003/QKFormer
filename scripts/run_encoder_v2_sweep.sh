#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Mini_QKFormer_128}"
DATA_DIR="${DATA_DIR:-${DATA:-}}"
OUT_ROOT="${OUT_ROOT:-${RUNROOT:-runs_encoder_v2}}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
PRINT_FREQ="${PRINT_FREQ:-64}"
LR="${LR:-0.005}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Set DATA_DIR or DATA to the CIFAR10-DVS root directory." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/_terminal_logs"
SWEEP_LOG="${OUT_ROOT}/_terminal_logs/encoder_v2_sweep.log"
exec > >(tee -a "${SWEEP_LOG}") 2>&1

echo "=== Encoder v2 sweep started at $(date) ==="
echo "host=$(hostname)"
echo "model=${MODEL}"
echo "data_dir=${DATA_DIR}"
echo "out_root=${OUT_ROOT}"
echo "device=${DEVICE}"
echo "batch_size=${BATCH_SIZE}"
echo "workers=${WORKERS}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"

COMMON=(
  --model "${MODEL}"
  --data-path "${DATA_DIR}"
  --output-dir "${OUT_ROOT}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --workers "${WORKERS}"
  --print-freq "${PRINT_FREQ}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --split-by number
  --profile-input-activity
)

run_train() {
  local tag="$1"
  local encoder="$2"
  local source_t="$3"
  local encoded_t="$4"
  local in_channels="$5"
  local run_log="${OUT_ROOT}/_terminal_logs/${tag}.train.log"

  echo "=== ${tag}: encoder=${encoder}, source_T=${source_t}, T=${encoded_t}, C=${in_channels} ==="
  python -u train.py \
    "${COMMON[@]}" \
    --encoder "${encoder}" \
    --source-time-steps "${source_t}" \
    --time-steps "${encoded_t}" \
    --in-channels "${in_channels}" \
    --experiment-tag "${tag}" \
    2>&1 | tee "${run_log}"
}

run_train E0_count_T16 count_number 16 16 2
run_train E1_count_T8 count_number 8 8 2
run_train E2_count_T4 count_number 4 4 2
run_train E3_occupancy_T16 occupancy_number 16 16 2
run_train E4_temporal_pack2_T8 temporal_pack_2 16 8 4

echo "=== Encoder v2 sweep finished at $(date) ==="
