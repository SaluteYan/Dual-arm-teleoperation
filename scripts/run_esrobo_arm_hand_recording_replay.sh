#!/usr/bin/env bash
# Replay aligned body and hand recordings to the ESROBO IsaacLab UDP device.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UDP_HOST="${ESROBO_BODY_UDP_HOST:-127.0.0.1}"
UDP_PORT="${ESROBO_BODY_UDP_PORT:-15050}"

exec python3 -u "${ROOT_DIR}/scripts/replay_esrobo_arm_hand_recordings.py" \
    --host "${UDP_HOST}" \
    --port "${UDP_PORT}" \
    "$@"
