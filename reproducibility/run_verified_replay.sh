#!/usr/bin/env bash
# Replay the checked-in PICO and SenseGlove evidence to an already running simulator.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT_DIR}"

unset LD_LIBRARY_PATH PYTHONPATH
export ESROBO_BODY_UDP_HOST="${ESROBO_BODY_UDP_HOST:-127.0.0.1}"
export ESROBO_BODY_UDP_PORT="${ESROBO_BODY_UDP_PORT:-15050}"

exec ./scripts/run_esrobo_arm_hand_recording_replay.sh \
    --body-record-path reproducibility/samples/pico-body-success-20260811.jsonl \
    --hand-record-path reproducibility/samples/senseglove-hands-success-20260818.jsonl \
    --hand-imu-mode relative \
    --print-interval 0 \
    "$@"
