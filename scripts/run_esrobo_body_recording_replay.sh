#!/usr/bin/env bash
# Replay recorded PICO/XRoboToolkit body-tracking data to IsaacLab bodytracking_udp.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UDP_HOST="${ESROBO_BODY_UDP_HOST:-127.0.0.1}"
UDP_PORT="${ESROBO_BODY_UDP_PORT:-15050}"

exec python3 -u "${ROOT_DIR}/scripts/replay_esrobo_body_recording.py" \
    --host "${UDP_HOST}" \
    --port "${UDP_PORT}" \
    "$@"
