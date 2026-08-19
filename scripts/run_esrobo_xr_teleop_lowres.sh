#!/usr/bin/env bash
# Low-resolution OpenXR/CloudXR launcher for ESROBO fixed-base dual-arm teleoperation.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ISAACLAB_DIR="${ROOT_DIR}/external/IsaacLab"
CLOUDXR_ENV="${CLOUDXR_ENV:-${HOME}/.cloudxr/run/cloudxr.env}"

PER_EYE_WIDTH="${PER_EYE_WIDTH:-1280}"
PER_EYE_HEIGHT="${PER_EYE_HEIGHT:-1280}"
DEVICE_FPS="${DEVICE_FPS:-72}"
BITRATE_MBPS="${BITRATE_MBPS:-60}"
CODEC="${CODEC:-h265}"
IMMERSIVE_MODE="${IMMERSIVE_MODE:-vr}"
CLIENT_PATH="${CLIENT_PATH:-sim/isaaclab}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
TELEOP_DEVICE="${TELEOP_DEVICE:-handtracking}"
TASK="${TASK:-DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0}"
DEFAULT_KIT_ARGS="--/persistent/physics/visualizationDisplayJoints=0"
ISAACLAB_KIT_ARGS="${ISAACLAB_KIT_ARGS-${DEFAULT_KIT_ARGS}}"

if [[ "${TELEOP_DEVICE}" == "bodytracking_udp" ]]; then
    echo "bodytracking_udp does not use IsaacLab XR rendering/streaming."
    echo "Forwarding to scripts/run_esrobo_bodytracking_teleop.sh."
    exec "${ROOT_DIR}/scripts/run_esrobo_bodytracking_teleop.sh" "$@"
fi

CLIENT_URL="https://${HOST_IP}:48322/client/?perEyeWidth=${PER_EYE_WIDTH}&perEyeHeight=${PER_EYE_HEIGHT}&deviceFrameRate=${DEVICE_FPS}&maxStreamingBitrateMbps=${BITRATE_MBPS}&codec=${CODEC}&immersiveMode=${IMMERSIVE_MODE}#/${CLIENT_PATH}"

echo "Open this URL in the PICO browser before pressing Connect:"
echo "${CLIENT_URL}"
echo
echo "Low-res profile: per-eye ${PER_EYE_WIDTH}x${PER_EYE_HEIGHT}, ${DEVICE_FPS} FPS target,"
echo "${BITRATE_MBPS} Mbps, ${CODEC}, ${IMMERSIVE_MODE}."
echo "Task: ${TASK}"
echo "Teleop device: ${TELEOP_DEVICE}"
echo "Kit args: ${ISAACLAB_KIT_ARGS}"
echo "After the XR scene appears, hold both hands at the teleop zero pose, then press Play to calibrate and step."
echo

if [[ -f "${CLOUDXR_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${CLOUDXR_ENV}"
else
    echo "Warning: ${CLOUDXR_ENV} not found. Start CloudXR first, then re-run this script." >&2
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export DUAL_ARM_TELEOP_ROOT="${DUAL_ARM_TELEOP_ROOT:-${ROOT_DIR}}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

cd "${ISAACLAB_DIR}"
exec ./isaaclab.sh -p "${ROOT_DIR}/scripts/esrobo_teleop_se3_agent.py" \
    --task "${TASK}" \
    --teleop_device "${TELEOP_DEVICE}" \
    --num_envs 1 \
    --device cuda:0 \
    --kit_args="${ISAACLAB_KIT_ARGS}" \
    --enable_pinocchio \
    --debug_xr_data \
    --debug_xr_interval 60 \
    --debug_perf \
    "$@"
