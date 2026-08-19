#!/usr/bin/env bash
# Launch an IsaacLab viewer for XRoboToolkit motion tracker wrist poses.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ISAACLAB_DIR="${ROOT_DIR}/external/IsaacLab"
ISAACLAB_ENV_NAME="${ISAACLAB_ENV_NAME:-env_isaaclab}"
DEVICE="${DEVICE:-cuda:0}"
PORT="${XROBOTOOLKIT_TRACKER_UDP_PORT:-15200}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

conda activate "${ISAACLAB_ENV_NAME}"
UDP_BIND_HOST="${XROBOTOOLKIT_TRACKER_UDP_BIND_HOST:-0.0.0.0}"
if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
    export TERM=xterm-256color
fi

echo "Starting IsaacLab XRoboToolkit motion tracker viewer."
echo "Listening for UDP on ${UDP_BIND_HOST}:${PORT}."
echo "Run scripts/run_xrobotoolkit_motion_tracker_udp_bridge.sh in another terminal."
echo

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh -p "${ROOT_DIR}/scripts/isaaclab_xrobotoolkit_motion_tracker_viewer.py" \
    --device "${DEVICE}" \
    --host "${UDP_BIND_HOST}" \
    --port "${PORT}" \
    "$@"
