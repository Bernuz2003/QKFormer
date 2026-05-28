#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 RUN_DIR ENCODER SOURCE_T ENCODED_T IN_CHANNELS" >&2
  echo "Example: $0 runs_encoder_v2/E4_temporal_pack2_T8 temporal_pack_2 16 8 4" >&2
  exit 1
fi

RUN_DIR="$1"
ENCODER="$2"
SOURCE_T="$3"
ENCODED_T="$4"
IN_CHANNELS="$5"

MODEL="${MODEL:-Mini_QKFormer_128}"
DATA_DIR="${DATA_DIR:-${DATA:-}}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
MAX_BATCHES="${MAX_BATCHES:-16}"
LR="${LR:-0.005}"

if [[ -z "${DATA_DIR}" ]]; then
  echo "Set DATA_DIR or DATA to the CIFAR10-DVS root directory." >&2
  exit 1
fi

CHECKPOINT="${RUN_DIR}/lr${LR}/checkpoint_max_test_acc1.pth"
PROFILE_DIR="${RUN_DIR}/profile"

python -u activity_profile.py \
  --model "${MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --data-path "${DATA_DIR}" \
  --output-dir "${PROFILE_DIR}" \
  --run-id "$(basename "${RUN_DIR}")" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --split test \
  --split-by number \
  --encoder "${ENCODER}" \
  --source-time-steps "${SOURCE_T}" \
  --time-steps "${ENCODED_T}" \
  --in-channels "${IN_CHANNELS}"

python -u analyze_run.py --run-dir "${RUN_DIR}"
