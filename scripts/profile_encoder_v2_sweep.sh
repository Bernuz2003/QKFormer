#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-${RUNROOT:-runs_encoder_v2}}"

"$(dirname "$0")/profile_encoder_v2_run.sh" "${OUT_ROOT}/E0_count_T16" count_number 16 16 2
"$(dirname "$0")/profile_encoder_v2_run.sh" "${OUT_ROOT}/E1_count_T8" count_number 8 8 2
"$(dirname "$0")/profile_encoder_v2_run.sh" "${OUT_ROOT}/E2_count_T4" count_number 4 4 2
"$(dirname "$0")/profile_encoder_v2_run.sh" "${OUT_ROOT}/E3_occupancy_T16" occupancy_number 16 16 2
"$(dirname "$0")/profile_encoder_v2_run.sh" "${OUT_ROOT}/E4_temporal_pack2_T8" temporal_pack_2 16 8 4
