#!/usr/bin/env bash
# Run ESROBO hand/fixed-position wrist teleoperation from real SenseGlove data only.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export ESROBO_BODY_UDP_HOST="${ESROBO_BODY_UDP_HOST:-0.0.0.0}"
export ESROBO_BODY_UDP_PORT="${ESROBO_BODY_UDP_PORT:-15050}"
export ESROBO_HAND_ONLY_MODE=1
export ESROBO_USE_HAND_IMU_ORIENTATION=1
export ESROBO_HAND_IMU_MAX_STALE_TIME_S="${ESROBO_HAND_IMU_MAX_STALE_TIME_S:-0.5}"
export ESROBO_HAND_IMU_LEFT_LOCAL_AXIS_SIGNS="${ESROBO_HAND_IMU_LEFT_LOCAL_AXIS_SIGNS:--1 -1 1}"
export ESROBO_HAND_IMU_LEFT_AXIS_ORDER="${ESROBO_HAND_IMU_LEFT_AXIS_ORDER:-2 0 1}"
export ESROBO_HAND_IMU_RIGHT_LOCAL_AXIS_SIGNS="${ESROBO_HAND_IMU_RIGHT_LOCAL_AXIS_SIGNS:--1 1 1}"
export ESROBO_HAND_IMU_RIGHT_SWAP_XY="${ESROBO_HAND_IMU_RIGHT_SWAP_XY:-0}"
export ESROBO_ENABLE_ELBOW_IK_TASKS=0
export ESROBO_WRIST_ONLY_IK=1
export ESROBO_BODY_REQUIRE_CALIBRATION=0
export ESROBO_BODY_AUTO_START_REFERENCE=0
export ESROBO_BODY_RETARGETING_MODE=wrist_delta
export ESROBO_BODY_ENABLE_VISUALIZATION=0
export ESROBO_HAND_SKELETON_VISUALIZATION=1
export ESROBO_HAND_SKELETON_VISUALIZATION_OFFSET="${ESROBO_HAND_SKELETON_VISUALIZATION_OFFSET:-0.55 0 0.18}"
export ESROBO_HAND_SKELETON_VISUALIZATION_SCALE="${ESROBO_HAND_SKELETON_VISUALIZATION_SCALE:-1.6}"

# Remove translation from the IK objective; only distal wrist rotation is controllable in this mode.
export ESROBO_IK_POSITION_COST="${ESROBO_IK_POSITION_COST:-0.0}"
export ESROBO_IK_ORIENTATION_COST="${ESROBO_IK_ORIENTATION_COST:-8.0}"
export ESROBO_IK_FRAME_TASK_GAIN="${ESROBO_IK_FRAME_TASK_GAIN:-1.0}"
export ESROBO_IK_FRAME_TASK_LM_DAMPING="${ESROBO_IK_FRAME_TASK_LM_DAMPING:-0.012}"
export ESROBO_IK_STATIC_TARGET_ORIENTATION_ERROR_RAD="${ESROBO_IK_STATIC_TARGET_ORIENTATION_ERROR_RAD:-0.05}"

echo "Starting ESROBO SenseGlove-only hand and wrist teleoperation."
echo "No PICO/XRoboToolkit body stream is required."
echo "Arm joints 1-4 stay locked; wrist joints 5-7 follow calibrated Nova 2 IMU orientation."
echo "Live packets use fixed Nova 2 glove axes plus a mandatory per-run IMU neutral pose."
echo "Legacy static receiver settings are used only for recordings or packets without calibrated-axis metadata."
echo "Glove measurements drive the 20 active hand joints; wrist translation is not controlled."
echo

exec "${ROOT_DIR}/scripts/run_esrobo_bodytracking_teleop.sh" "$@"
