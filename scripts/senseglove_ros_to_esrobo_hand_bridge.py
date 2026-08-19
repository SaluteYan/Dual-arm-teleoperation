#!/usr/bin/env python3
"""Bridge official SenseGlove ROS 2 state topics to ESROBO IsaacLab hand joints."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FLEX_SUFFIXES = ("mcp", "pip", "dip")
DEFAULT_LEFT_SERIAL = "01001"
DEFAULT_RIGHT_SERIAL = "01002"
DEFAULT_CALIBRATION_PATH = Path("config/senseglove_esrobo_calibration.json")

ESROBO_HAND_JOINT_ORDER = (
    "left_thumb_cmc_roll",
    "left_thumb_cmc_yaw",
    "left_thumb_cmc_pitch",
    "left_index_mcp_roll",
    "left_index_mcp_pitch",
    "left_middle_mcp_pitch",
    "left_ring_mcp_roll",
    "left_ring_mcp_pitch",
    "left_pinky_mcp_roll",
    "left_pinky_mcp_pitch",
    "right_thumb_cmc_roll",
    "right_thumb_cmc_yaw",
    "right_thumb_cmc_pitch",
    "right_index_mcp_roll",
    "right_index_mcp_pitch",
    "right_middle_mcp_pitch",
    "right_ring_mcp_roll",
    "right_ring_mcp_pitch",
    "right_pinky_mcp_roll",
    "right_pinky_mcp_pitch",
)

ESROBO_HAND_JOINT_LIMITS = {
    "left_thumb_cmc_roll": (0.0, 1.03),
    "left_thumb_cmc_yaw": (0.0, 1.40),
    "left_thumb_cmc_pitch": (0.0, 0.52),
    "left_index_mcp_roll": (0.0, 0.19),
    "left_index_mcp_pitch": (0.0, 1.36),
    "left_middle_mcp_pitch": (0.0, 1.36),
    "left_ring_mcp_roll": (0.0, 0.20),
    "left_ring_mcp_pitch": (0.0, 1.36),
    "left_pinky_mcp_roll": (0.0, 0.30),
    "left_pinky_mcp_pitch": (0.0, 1.36),
    "right_thumb_cmc_roll": (0.0, 1.03),
    "right_thumb_cmc_yaw": (0.0, 1.40),
    "right_thumb_cmc_pitch": (0.0, 0.52),
    "right_index_mcp_roll": (0.0, 0.19),
    "right_index_mcp_pitch": (0.0, 1.36),
    "right_middle_mcp_pitch": (0.0, 1.36),
    "right_ring_mcp_roll": (0.0, 0.20),
    "right_ring_mcp_pitch": (0.0, 1.36),
    "right_pinky_mcp_roll": (0.0, 0.30),
    "right_pinky_mcp_pitch": (0.0, 1.36),
}


@dataclass
class SideState:
    topic: str
    msg: Any | None = None
    stamp_monotonic: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    imu_quat_wxyz: list[float] | None = None
    imu_raw_norm: float | None = None
    imu_w_repaired: bool = False
    hand_positions_mm: list[list[float]] = field(default_factory=list)
    callback_sequence: int = 0


class SenseGloveUdpBridge:
    """ROS subscriber state and SenseGlove-to-ESROBO hand retargeting."""

    def __init__(
        self, node: Any, msg_type: Any, args: argparse.Namespace, qos_profile: Any
    ):
        self.args = args
        self.left = SideState(args.left_topic)
        self.right = SideState(args.right_topic)
        self._sent_count = 0
        self._last_targets = [0.0] * len(ESROBO_HAND_JOINT_ORDER)

        node.create_subscription(
            msg_type, args.left_topic, self._make_callback(self.left), qos_profile
        )
        node.create_subscription(
            msg_type, args.right_topic, self._make_callback(self.right), qos_profile
        )

    def _make_callback(self, state: SideState):
        def callback(msg: Any) -> None:
            state.msg = msg
            state.stamp_monotonic = time.monotonic()
            state.features = extract_features(msg)
            imu_diagnostics: dict[str, Any] = {}
            state.imu_quat_wxyz = extract_imu_quat_wxyz(
                msg,
                self.args.imu_correction,
                self.args.imu_w_repair,
                imu_diagnostics,
            )
            state.imu_raw_norm = imu_diagnostics.get("raw_norm")
            state.imu_w_repaired = bool(imu_diagnostics.get("w_repaired", False))
            state.hand_positions_mm = extract_hand_positions_mm(msg)
            state.features.update(extract_hand_vector_features(state.hand_positions_mm))
            state.callback_sequence += 1

        return callback

    def has_both_hands(self, max_age_s: float) -> bool:
        return self._is_fresh(self.left, max_age_s) and self._is_fresh(
            self.right, max_age_s
        )

    def _is_fresh(self, state: SideState, max_age_s: float) -> bool:
        if state.msg is None or state.stamp_monotonic is None:
            return False
        if time.monotonic() - state.stamp_monotonic > max_age_s:
            return False
        return (
            self.args.retargeting_mode != "vector" or len(state.hand_positions_mm) == 20
        )

    def topic_status(self) -> str:
        return (
            f"left={self._age_text(self.left)}/points={len(self.left.hand_positions_mm)}"
            f"/imu={self._imu_status(self.left)}({self.left.topic}) "
            f"right={self._age_text(self.right)}/points={len(self.right.hand_positions_mm)}"
            f"/imu={self._imu_status(self.right)}({self.right.topic})"
        )

    def _age_text(self, state: SideState) -> str:
        if state.stamp_monotonic is None:
            return "none"
        return f"{time.monotonic() - state.stamp_monotonic:.3f}s"

    @staticmethod
    def _imu_status(state: SideState) -> str:
        if state.imu_raw_norm is None:
            return "none"
        suffix = "/w-repaired" if state.imu_w_repaired else ""
        return f"norm={state.imu_raw_norm:.3f}{suffix}"

    def build_targets(
        self, calibration: dict[str, Any]
    ) -> tuple[list[float], dict[str, dict[str, float]]]:
        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        left_targets, left_curls = retarget_side(
            source_left.features,
            calibration["open"][
                "right" if self.args.swap_left_right_targets else "left"
            ],
            calibration["closed"][
                "right" if self.args.swap_left_right_targets else "left"
            ],
            "left",
            self.args,
        )
        right_targets, right_curls = retarget_side(
            source_right.features,
            calibration["open"][
                "left" if self.args.swap_left_right_targets else "right"
            ],
            calibration["closed"][
                "left" if self.args.swap_left_right_targets else "right"
            ],
            "right",
            self.args,
        )
        targets = left_targets + right_targets
        if self.args.smoothing_alpha < 1.0:
            alpha = max(0.0, self.args.smoothing_alpha)
            targets = [
                alpha * current + (1.0 - alpha) * previous
                for current, previous in zip(targets, self._last_targets)
            ]
        self._last_targets = targets
        return targets, {"left": left_curls, "right": right_curls}

    def build_orientation_deltas(
        self, calibration: dict[str, Any]
    ) -> dict[str, list[float]]:
        if self.args.disable_imu_orientation:
            return {}

        imu_neutral = calibration.get("imu_neutral")
        if not isinstance(imu_neutral, dict):
            return {}

        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        left_key = "right" if self.args.swap_left_right_targets else "left"
        right_key = "left" if self.args.swap_left_right_targets else "right"

        left_delta = orientation_delta_wxyz(
            source_left.imu_quat_wxyz, imu_neutral.get(left_key)
        )
        right_delta = orientation_delta_wxyz(
            source_right.imu_quat_wxyz, imu_neutral.get(right_key)
        )
        if left_delta is None or right_delta is None:
            return {}
        return {"left": left_delta, "right": right_delta}

    def send(
        self,
        sock: socket.socket,
        calibration: dict[str, Any],
    ) -> tuple[
        list[float],
        dict[str, dict[str, float]],
        dict[str, list[float]],
        dict[str, Any],
    ]:
        targets, curls = self.build_targets(calibration)
        orientation_deltas = self.build_orientation_deltas(calibration)
        packet = {
            "timestamp": time.time(),
            "source": "senseglove_ros",
            "type": "esrobo_hand_joints",
            "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
            "hand_joints": targets,
            "retargeting": {
                "method": (
                    "calibrated_hand_vectors_to_esrobo_10_active_joints_per_hand"
                    if self.args.retargeting_mode == "vector"
                    else "calibrated_finger_curl_to_esrobo_active_hand_joints"
                ),
                "robot_left_source": (
                    "right_glove" if self.args.swap_left_right_targets else "left_glove"
                ),
                "robot_right_source": (
                    "left_glove" if self.args.swap_left_right_targets else "right_glove"
                ),
            },
            "curls": curls,
        }
        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        if (
            len(source_left.hand_positions_mm) == 20
            and len(source_right.hand_positions_mm) == 20
        ):
            packet["hand_skeleton_positions"] = {
                "left": canonicalize_hand_points_for_robot(
                    source_left.hand_positions_mm, "left"
                ),
                "right": canonicalize_hand_points_for_robot(
                    source_right.hand_positions_mm, "right"
                ),
            }
            packet["hand_skeleton_layout"] = (
                "finger_major_5x4_thumb_to_pinky_proximal_to_distal"
            )
            packet["hand_skeleton_unit"] = "mm"
            packet["hand_skeleton_frame"] = "robot_hand_local"
            packet["hand_skeleton_source"] = (
                "senseglove_ros/SenseGloveState.hand_position"
            )
        if orientation_deltas:
            packet["hand_orientation_deltas"] = orientation_deltas
            packet["hand_orientation_delta_order"] = "wxyz"
            packet["hand_orientation_source_frame"] = self.args.imu_correction
            packet["hand_orientation_retargeting"] = {
                "method": "calibrated_senseglove_imu_delta",
                "robot_left_source": (
                    "right_glove" if self.args.swap_left_right_targets else "left_glove"
                ),
                "robot_right_source": (
                    "left_glove" if self.args.swap_left_right_targets else "right_glove"
                ),
            }
        encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(encoded, (self.args.host, self.args.port))
        self._sent_count += 1
        return targets, curls, orientation_deltas, packet

    def raw_recording_snapshot(self) -> dict[str, Any]:
        """Capture both physical glove messages that produced the current UDP target."""
        return {
            "left": side_recording_snapshot(self.left),
            "right": side_recording_snapshot(self.right),
        }


class HandRecordingWriter:
    """Write replay-compatible hand packets plus same-frame raw SenseGlove data."""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        calibration: dict[str, Any],
    ):
        self.path = path.expanduser()
        if self.path.exists() and not args.record_overwrite:
            raise FileExistsError(
                f"recording already exists: {self.path}; use --record-overwrite to replace it"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", buffering=1)
        self.start_delay_s = args.record_start_delay_s
        self.duration_s = args.record_duration_s
        self.armed_monotonic: float | None = None
        self.started_monotonic: float | None = None
        self.sequence = 0
        self.complete = False
        self._last_countdown: int | None = None
        header = {
            "type": "esrobo_hand_recording_header",
            "format_version": 2,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "real_senseglove_ros",
            "left_serial": args.left_serial,
            "right_serial": args.right_serial,
            "left_topic": args.left_topic,
            "right_topic": args.right_topic,
            "nominal_rate_hz": args.rate_hz,
            "record_start_delay_s": self.start_delay_s,
            "record_duration_s": self.duration_s,
            "imu_orientation_included": not args.disable_imu_orientation,
            "imu_correction": args.imu_correction,
            "imu_w_repair": args.imu_w_repair,
            "retargeting_mode": args.retargeting_mode,
            "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
            "raw_fields": {
                "joint_position": "SenseGloveState.position as published",
                "joint_absolute_velocity": "SenseGloveState.absolute_velocity as published",
                "hand_position": "millimeters, physical glove wrist-local frame",
                "finger_tip_position": "millimeters, physical glove wrist-local frame",
                "imu_orientation_raw": "xyzw, exactly as published",
                "imu_orientation_corrected": "wxyz, repaired/normalized and coordinate-corrected",
            },
            "calibration": calibration,
        }
        self.stream.write(json.dumps(header, separators=(",", ":")) + "\n")

    def update(
        self,
        now: float,
        packet: dict[str, Any],
        raw_snapshot: dict[str, Any],
    ) -> str:
        if self.complete:
            return "complete"
        if self.armed_monotonic is None:
            self.armed_monotonic = now
            print(
                "[senseglove_bridge] RECORDING ARMED: keep both hands still for "
                f"{self.start_delay_s:.1f}s; teleoperation is already active.",
                flush=True,
            )

        delay_elapsed = now - self.armed_monotonic
        if delay_elapsed < self.start_delay_s:
            remaining = max(self.start_delay_s - delay_elapsed, 0.0)
            countdown = max(1, math.ceil(remaining))
            if countdown != self._last_countdown:
                print(
                    f"[senseglove_bridge] Recording starts in {countdown}s; hold still.",
                    flush=True,
                )
                self._last_countdown = countdown
            return "countdown"

        if self.started_monotonic is None:
            self.started_monotonic = now
            print(
                f"[senseglove_bridge] RECORDING ACTIVE: move now; saving to {self.path}.",
                flush=True,
            )

        t_rel_s = now - self.started_monotonic
        if self.duration_s > 0.0 and t_rel_s > self.duration_s:
            self.complete = True
            self.stream.flush()
            print(
                "[senseglove_bridge] RECORDING COMPLETE: "
                f"saved={self.path} frames={self.sequence} duration={self.duration_s:.3f}s.",
                flush=True,
            )
            return "complete"

        item = {
            "type": "esrobo_hand_packet",
            "record_sequence": self.sequence,
            "t_rel_s": t_rel_s,
            "record_phase": "motion",
            "packet": packet,
            "senseglove_raw": raw_snapshot,
        }
        self.stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.sequence += 1
        return "recording"

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    env_swap = _env_bool(
        "ESROBO_HAND_SWAP_LEFT_RIGHT_TARGETS",
        _env_bool("ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS", False),
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("ESROBO_BODY_UDP_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050"))
    )
    parser.add_argument(
        "--left-serial",
        default=os.environ.get("SENSEGLOVE_LEFT_SERIAL", DEFAULT_LEFT_SERIAL),
    )
    parser.add_argument(
        "--right-serial",
        default=os.environ.get("SENSEGLOVE_RIGHT_SERIAL", DEFAULT_RIGHT_SERIAL),
    )
    parser.add_argument("--left-topic", default=None)
    parser.add_argument("--right-topic", default=None)
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=float(os.environ.get("SENSEGLOVE_BRIDGE_RATE_HZ", "60")),
    )
    parser.add_argument("--qos-depth", type=int, default=5)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Write replay-compatible JSONL with raw finger, skeleton, and IMU samples.",
    )
    parser.add_argument(
        "--record-start-delay-s",
        type=float,
        default=3.0,
        help="Keep streaming but wait this long after fresh data before recording.",
    )
    parser.add_argument(
        "--record-duration-s",
        type=float,
        default=0.0,
        help="Stop recording after this duration; 0 records until Ctrl-C.",
    )
    parser.add_argument(
        "--record-overwrite",
        action="store_true",
        help="Replace an existing --record-path instead of refusing to start.",
    )
    parser.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_PATH
    )
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--calibration-duration", type=float, default=10.0)
    parser.add_argument("--calibration-sample-start", type=float, default=6.0)
    parser.add_argument(
        "--calibration-method", choices=("last", "average"), default="last"
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Start each calibration phase without waiting for ENTER.",
    )
    parser.add_argument(
        "--disable-imu-orientation",
        action="store_true",
        help="Send only finger joints; do not send calibrated SenseGlove IMU hand orientation.",
    )
    parser.add_argument(
        "--imu-correction",
        choices=("unity_to_ros", "raw"),
        default=os.environ.get("SENSEGLOVE_IMU_CORRECTION", "unity_to_ros"),
        help="Coordinate correction for SenseGlove imu_orientation before calibration.",
    )
    parser.add_argument(
        "--imu-w-repair",
        choices=("auto", "off", "reconstruct"),
        default=os.environ.get("SENSEGLOVE_IMU_W_REPAIR", "auto"),
        help=(
            "Repair the observed Nova 2/SGCore fault where quaternion w duplicates z. "
            "Auto repairs only duplicated, non-unit samples."
        ),
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="1.0 disables smoothing; lower values smooth more.",
    )
    parser.add_argument(
        "--retargeting-mode",
        choices=("vector", "curl"),
        default=os.environ.get("ESROBO_HAND_RETARGETING_MODE", "vector"),
        help="Use 5x4 hand-segment vectors (default) or the legacy averaged joint-curl mapping.",
    )
    parser.add_argument(
        "--spread-scale",
        type=float,
        default=None,
        help="Scale finger-vector spread into ESROBO roll joints; defaults to 1 for vector mode and 0 for curl mode.",
    )
    parser.add_argument("--vector-spread-range-deg", type=float, default=18.0)
    parser.add_argument("--thumb-roll-gain", type=float, default=0.35)
    parser.add_argument("--thumb-yaw-gain", type=float, default=0.50)
    parser.add_argument(
        "--swap-left-right-targets",
        dest="swap_left_right_targets",
        action="store_true",
        default=env_swap,
    )
    parser.add_argument(
        "--no-swap-left-right-targets",
        dest="swap_left_right_targets",
        action="store_false",
    )
    args = parser.parse_args(argv)
    args.left_topic = (
        args.left_topic or f"/senseglove/glove{args.left_serial}/lh/senseglove_states"
    )
    args.right_topic = (
        args.right_topic or f"/senseglove/glove{args.right_serial}/rh/senseglove_states"
    )
    args.rate_hz = max(args.rate_hz, 1.0)
    args.calibration_duration = max(args.calibration_duration, 0.1)
    args.calibration_sample_start = min(
        max(args.calibration_sample_start, 0.0), args.calibration_duration
    )
    args.smoothing_alpha = min(max(args.smoothing_alpha, 0.0), 1.0)
    args.spread_scale = (
        (1.0 if args.retargeting_mode == "vector" else 0.0)
        if args.spread_scale is None
        else args.spread_scale
    )
    args.vector_spread_range_deg = max(args.vector_spread_range_deg, 1.0)
    args.record_start_delay_s = max(args.record_start_delay_s, 0.0)
    args.record_duration_s = max(args.record_duration_s, 0.0)
    return args


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def extract_features(msg: Any) -> dict[str, float]:
    names = list(getattr(msg, "joint_names", []))
    positions = list(getattr(msg, "position", []))
    joints: dict[str, float] = {}
    for name, position in zip(names, positions):
        try:
            value = float(position)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            joints[_normalize_senseglove_joint_name(name)] = value

    features: dict[str, float] = {}
    for finger in FINGERS:
        flex_values = [
            joints[f"{finger}_{suffix}"]
            for suffix in FLEX_SUFFIXES
            if f"{finger}_{suffix}" in joints
        ]
        features[finger] = _mean(flex_values) if flex_values else 0.0
        brake_name = f"{finger}_brake"
        features[brake_name] = joints.get(brake_name, 0.0)
    return features


def side_recording_snapshot(state: SideState) -> dict[str, Any]:
    """Serialize the original ROS message and the corrected values used by retargeting."""
    msg = state.msg
    if msg is None:
        return {}
    orientation = getattr(msg, "imu_orientation", None)
    raw_imu_xyzw = (
        [
            _finite_float(getattr(orientation, axis, None))
            for axis in ("x", "y", "z", "w")
        ]
        if orientation is not None
        else None
    )
    return {
        "topic": state.topic,
        "callback_sequence": state.callback_sequence,
        "received_monotonic_ns": (
            int(state.stamp_monotonic * 1.0e9)
            if state.stamp_monotonic is not None
            else None
        ),
        "ros_stamp": _ros_stamp_dict(msg),
        "joint_names": [str(value) for value in getattr(msg, "joint_names", [])],
        "position": _finite_float_list(getattr(msg, "position", [])),
        "absolute_velocity": _finite_float_list(getattr(msg, "absolute_velocity", [])),
        "hand_position_mm_raw": _point_list(getattr(msg, "hand_position", [])),
        "finger_tip_position_mm_raw": _point_list(
            getattr(msg, "finger_tip_position", [])
        ),
        "hand_position_mm_used": [list(point) for point in state.hand_positions_mm],
        "imu_orientation_raw_xyzw": raw_imu_xyzw,
        "imu_raw_norm": state.imu_raw_norm,
        "imu_w_repaired": state.imu_w_repaired,
        "imu_orientation_corrected_wxyz": (
            list(state.imu_quat_wxyz) if state.imu_quat_wxyz is not None else None
        ),
        "retargeting_features": dict(state.features),
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_float_list(values: Any) -> list[float | None]:
    return [_finite_float(value) for value in list(values)]


def _point_list(points: Any) -> list[list[float | None]]:
    return [
        [_finite_float(getattr(point, axis, None)) for axis in ("x", "y", "z")]
        for point in list(points)
    ]


def _ros_stamp_dict(msg: Any) -> dict[str, int] | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
    except (AttributeError, TypeError, ValueError):
        return None


def extract_imu_quat_wxyz(
    msg: Any,
    correction: str,
    imu_w_repair: str = "auto",
    diagnostics: dict[str, Any] | None = None,
) -> list[float] | None:
    orientation = getattr(msg, "imu_orientation", None)
    if orientation is None:
        return None
    raw_xyzw = [
        getattr(orientation, "x", 0.0),
        getattr(orientation, "y", 0.0),
        getattr(orientation, "z", 0.0),
        getattr(orientation, "w", 1.0),
    ]
    raw_xyzw, raw_norm, w_repaired = repair_senseglove_imu_w(raw_xyzw, imu_w_repair)
    if diagnostics is not None:
        diagnostics["raw_norm"] = raw_norm
        diagnostics["w_repaired"] = w_repaired
    corrected_xyzw = correct_imu_quat_xyzw(raw_xyzw, correction)
    if corrected_xyzw is None:
        return None
    x, y, z, w = corrected_xyzw
    return normalize_quat_wxyz([w, x, y, z])


def repair_senseglove_imu_w(
    raw_xyzw: list[float], mode: str
) -> tuple[list[float], float | None, bool]:
    """Repair Nova 2 samples where SGCore returns the z component as w."""
    try:
        values = [float(value) for value in raw_xyzw]
    except (TypeError, ValueError):
        return raw_xyzw, None, False
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return values, None, False

    x, y, z, w = values
    raw_norm = math.sqrt(sum(value * value for value in values))
    xyz_norm_sq = x * x + y * y + z * z
    duplicated_w = math.isclose(w, z, rel_tol=1.0e-5, abs_tol=1.0e-5)
    non_unit = abs(raw_norm - 1.0) > 0.02
    should_repair = mode == "reconstruct" or (
        mode == "auto" and duplicated_w and non_unit
    )
    if not should_repair or xyz_norm_sq > 1.02:
        return values, raw_norm, False

    repaired_w = math.sqrt(max(0.0, 1.0 - min(xyz_norm_sq, 1.0)))
    if w < 0.0:
        repaired_w = -repaired_w
    return [x, y, z, repaired_w], raw_norm, True


def extract_hand_positions_mm(msg: Any) -> list[list[float]]:
    """Extract SGCore's 5x4 hand joints, which are millimeters relative to the wrist."""
    positions: list[list[float]] = []
    for point in list(getattr(msg, "hand_position", [])):
        xyz = [float(getattr(point, axis, float("nan"))) for axis in ("x", "y", "z")]
        if not all(math.isfinite(value) for value in xyz):
            return []
        positions.append(xyz)
    if len(positions) != 20:
        return []

    # Prefer the separately published SGCore fingertips when they are geometrically
    # plausible. This also repairs recordings made with the older 19-point guard,
    # where the fourth pinky point was left at zero.
    fingertips = list(getattr(msg, "finger_tip_position", []))
    if len(fingertips) == 5:
        for finger_index, point in enumerate(fingertips):
            xyz = [
                float(getattr(point, axis, float("nan"))) for axis in ("x", "y", "z")
            ]
            distal = positions[finger_index * 4 + 2]
            distal_to_tip = math.dist(distal, xyz)
            if (
                all(math.isfinite(value) for value in xyz)
                and 1.0 <= distal_to_tip <= 120.0
            ):
                positions[finger_index * 4 + 3] = xyz
    return positions


def extract_hand_vector_features(
    hand_positions_mm: list[list[float]],
) -> dict[str, float]:
    """Describe a hand using palm-relative segment directions, independent of hand size."""
    if len(hand_positions_mm) != 20:
        return {}
    chains = {
        finger: [
            [float(value) for value in point]
            for point in hand_positions_mm[index * 4 : index * 4 + 4]
        ]
        for index, finger in enumerate(FINGERS)
    }
    lateral = _vector_normalize(
        _vector_subtract(chains["index"][0], chains["pinky"][0])
    )
    forward = _vector_normalize(chains["middle"][0])
    if lateral is None or forward is None:
        return {}
    normal = _vector_normalize(_vector_cross(lateral, forward))
    if normal is None:
        return {}
    forward = _vector_normalize(_vector_cross(normal, lateral))
    if forward is None:
        return {}

    features: dict[str, float] = {}
    for finger, points in chains.items():
        segments = [
            _vector_normalize(_vector_subtract(points[index + 1], points[index]))
            for index in range(3)
        ]
        if any(segment is None for segment in segments):
            return {}
        first, second, third = segments
        base_flex = math.atan2(_vector_dot(first, normal), _vector_dot(first, forward))
        pip_flex = _unsigned_vector_angle(first, second)
        dip_flex = _unsigned_vector_angle(second, third)
        features[f"vector_{finger}_curl"] = base_flex + pip_flex + dip_flex
        features[f"vector_{finger}_spread"] = math.atan2(
            _vector_dot(first, lateral),
            _vector_dot(first, forward),
        )
        features[f"vector_{finger}_elevation"] = math.atan2(
            _vector_dot(first, normal),
            math.hypot(_vector_dot(first, forward), _vector_dot(first, lateral)),
        )
    return features


def canonicalize_hand_points_for_robot(
    hand_positions_mm: list[list[float]], side: str
) -> list[list[float]]:
    """Express SGCore points in the mirrored ESROBO hand-link axis convention."""
    if len(hand_positions_mm) != 20 or side not in ("left", "right"):
        return []
    index_base = [float(value) for value in hand_positions_mm[4]]
    middle_base = [float(value) for value in hand_positions_mm[8]]
    pinky_base = [float(value) for value in hand_positions_mm[16]]
    lateral = _vector_normalize(_vector_subtract(index_base, pinky_base))
    forward = _vector_normalize(middle_base)
    if lateral is None or forward is None:
        return []
    normal = _vector_normalize(_vector_cross(lateral, forward))
    if normal is None:
        return []
    forward = _vector_normalize(_vector_cross(normal, lateral))
    if forward is None:
        return []
    robot_y_sign = -1.0 if side == "left" else 1.0
    return [
        [
            _vector_dot(point, normal),
            robot_y_sign * _vector_dot(point, lateral),
            _vector_dot(point, forward),
        ]
        for point in hand_positions_mm
    ]


def _vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _vector_dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _vector_cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _vector_normalize(vector: list[float]) -> list[float] | None:
    norm = math.sqrt(_vector_dot(vector, vector))
    if norm < 1.0e-6 or not math.isfinite(norm):
        return None
    return [value / norm for value in vector]


def _unsigned_vector_angle(first: list[float], second: list[float]) -> float:
    return math.acos(min(max(_vector_dot(first, second), -1.0), 1.0))


def correct_imu_quat_xyzw(raw_xyzw: list[float], correction: str) -> list[float] | None:
    q_in = normalize_quat_xyzw(raw_xyzw)
    if q_in is None:
        return None
    if correction == "raw":
        return q_in

    # Matches the official senseglove_interaction/common/imu_tf_broadcaster.py conversion.
    x, y, z, w = q_in
    unity_to_ros_q = normalize_quat_xyzw([z, -y, x, w])
    if unity_to_ros_q is None:
        return None
    q_corr = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    return normalize_quat_xyzw(quat_multiply_xyzw(q_corr, unity_to_ros_q))


def _normalize_senseglove_joint_name(name: Any) -> str:
    text = str(name).strip().lower()
    if len(text) > 2 and text[1] == "_" and text[0] in ("l", "r"):
        text = text[2:]
    return text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_or_collect_calibration(
    bridge: SenseGloveUdpBridge,
    node: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    calibration_path = args.calibration_file.expanduser()
    if calibration_path.exists() and not args.recalibrate:
        with calibration_path.open("r", encoding="utf-8") as stream:
            calibration = json.load(stream)
        has_required_vectors = args.retargeting_mode != "vector" or all(
            f"vector_{finger}_curl" in calibration.get(pose, {}).get(side, {})
            for pose in ("open", "closed")
            for side in ("left", "right")
            for finger in FINGERS
        )
        imu_neutral = calibration.get("imu_neutral")
        has_required_imu = args.disable_imu_orientation or (
            isinstance(imu_neutral, dict)
            and all(
                isinstance(imu_neutral.get(side), list) and len(imu_neutral[side]) == 4
                for side in ("left", "right")
            )
        )
        repair_mode_matches = args.disable_imu_orientation or (
            calibration.get("imu_w_repair") == args.imu_w_repair
        )
        if has_required_vectors and has_required_imu and repair_mode_matches:
            print(
                f"[senseglove_bridge] Loaded calibration from {calibration_path}.",
                flush=True,
            )
            return calibration
        if not has_required_vectors:
            print(
                f"[senseglove_bridge] Loaded {calibration_path}, but it has no hand-vector reference; "
                "starting the required vector recalibration.",
                flush=True,
            )
            args.recalibrate = True
            return load_or_collect_calibration(bridge, node, args)
        if not repair_mode_matches:
            print(
                f"[senseglove_bridge] Loaded {calibration_path}, but its IMU w-repair mode "
                f"does not match {args.imu_w_repair!r}; starting recalibration.",
                flush=True,
            )
            args.recalibrate = True
            return load_or_collect_calibration(bridge, node, args)
        print(
            f"[senseglove_bridge] Loaded {calibration_path}, but it has no IMU neutral pose; "
            "starting recalibration.",
            flush=True,
        )

    if args.recalibrate:
        print(
            "[senseglove_bridge] Recalibration requested; existing file will be replaced.",
            flush=True,
        )
    elif calibration_path.exists():
        print(
            f"[senseglove_bridge] Existing calibration at {calibration_path} will be replaced.",
            flush=True,
        )
    else:
        print(
            f"[senseglove_bridge] No calibration file at {calibration_path}; starting calibration.",
            flush=True,
        )

    wait_for_both_hands(bridge, node, args)
    open_features, imu_neutral = collect_calibration_pose(
        bridge,
        node,
        args,
        label="1/2 open hand + IMU neutral",
        instruction=(
            "open both hands naturally, keep the fingers straight, and hold the hand/wrist orientation "
            "you want to map to the robot initial hand orientation"
        ),
        require_imu=not args.disable_imu_orientation,
    )
    closed_features, _closed_imu = collect_calibration_pose(
        bridge,
        node,
        args,
        label="2/2 closed fist",
        instruction="close both hands into a fist",
        require_imu=False,
    )
    calibration = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "senseglove_ros/SenseGloveState",
        "method": "two_pose_open_closed_with_imu_neutral",
        "calibration_method": args.calibration_method,
        "valid_window": {
            "sample_start_s": args.calibration_sample_start,
            "duration_s": args.calibration_duration,
        },
        "open": open_features,
        "closed": closed_features,
        "imu_neutral": imu_neutral,
        "imu_correction": args.imu_correction,
        "imu_w_repair": args.imu_w_repair,
        "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[senseglove_bridge] Saved calibration to {calibration_path}.", flush=True)
    return calibration


def wait_for_both_hands(
    bridge: SenseGloveUdpBridge, node: Any, args: argparse.Namespace
) -> None:
    next_print = 0.0
    while True:
        spin_once(node, timeout_sec=0.05)
        if bridge.has_both_hands(args.max_state_age):
            print(
                f"[senseglove_bridge] Receiving both gloves: {bridge.topic_status()}.",
                flush=True,
            )
            return
        now = time.monotonic()
        if now >= next_print:
            print(
                f"[senseglove_bridge] Waiting for SenseGlove topics: {bridge.topic_status()}.",
                flush=True,
            )
            next_print = now + 1.0


def collect_calibration_pose(
    bridge: SenseGloveUdpBridge,
    node: Any,
    args: argparse.Namespace,
    label: str,
    instruction: str,
    require_imu: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]]]:
    print(f"[senseglove_bridge] Calibration {label}: {instruction}.", flush=True)
    if not args.no_prompt:
        input(
            "[senseglove_bridge] Press ENTER when ready to start this "
            f"{args.calibration_duration:.1f}s pose capture..."
        )

    left_samples: list[dict[str, float]] = []
    right_samples: list[dict[str, float]] = []
    left_imu_samples: list[list[float]] = []
    right_imu_samples: list[list[float]] = []
    start = time.monotonic()
    next_print = 0.0
    while True:
        spin_once(node, timeout_sec=0.01)
        now = time.monotonic()
        elapsed = now - start
        if elapsed >= args.calibration_sample_start and bridge.has_both_hands(
            args.max_state_age
        ):
            left_samples.append(dict(bridge.left.features))
            right_samples.append(dict(bridge.right.features))
            if (
                bridge.left.imu_quat_wxyz is not None
                and bridge.right.imu_quat_wxyz is not None
            ):
                left_imu_samples.append(list(bridge.left.imu_quat_wxyz))
                right_imu_samples.append(list(bridge.right.imu_quat_wxyz))
        if now >= next_print:
            remaining = max(0.0, args.calibration_duration - elapsed)
            if elapsed < args.calibration_sample_start:
                until_record = math.ceil(args.calibration_sample_start - elapsed)
                print(
                    f"[senseglove_bridge] Hold {label}: {until_record}s until recording starts, "
                    f"{math.ceil(remaining)}s until this pose locks.",
                    flush=True,
                )
            else:
                print(
                    f"[senseglove_bridge] Recording {label}: {math.ceil(remaining)}s remaining, "
                    f"samples={len(left_samples)} imu_samples={len(left_imu_samples)}.",
                    flush=True,
                )
            next_print = now + 1.0
        if elapsed >= args.calibration_duration:
            break

    if not left_samples or not right_samples:
        raise RuntimeError(
            f"no valid SenseGlove samples captured for calibration {label}"
        )
    if require_imu and (not left_imu_samples or not right_imu_samples):
        raise RuntimeError(
            f"no valid SenseGlove IMU samples captured for calibration {label}"
        )

    left_features = reduce_samples(left_samples, args.calibration_method)
    right_features = reduce_samples(right_samples, args.calibration_method)
    imu_neutral: dict[str, list[float]] = {}
    if left_imu_samples and right_imu_samples:
        imu_neutral = {
            "left": reduce_quat_samples(left_imu_samples, args.calibration_method),
            "right": reduce_quat_samples(right_imu_samples, args.calibration_method),
        }
    print(
        f"[senseglove_bridge] Recorded {label}: "
        f"left={format_feature_preview(left_features)} right={format_feature_preview(right_features)} "
        f"imu_left={format_quat_preview(imu_neutral.get('left'))} "
        f"imu_right={format_quat_preview(imu_neutral.get('right'))}.",
        flush=True,
    )
    return {"left": left_features, "right": right_features}, imu_neutral


def spin_once(node: Any, timeout_sec: float) -> None:
    import rclpy

    rclpy.spin_once(node, timeout_sec=timeout_sec)


def reduce_samples(samples: list[dict[str, float]], method: str) -> dict[str, float]:
    if method == "last":
        return dict(samples[-1])
    keys = sorted({key for sample in samples for key in sample.keys()})
    return {
        key: _mean([sample[key] for sample in samples if key in sample]) for key in keys
    }


def reduce_quat_samples(samples: list[list[float]], method: str) -> list[float]:
    if method == "last":
        return normalize_quat_wxyz(samples[-1])

    reference = normalize_quat_wxyz(samples[0])
    aligned: list[list[float]] = []
    for sample in samples:
        quat = normalize_quat_wxyz(sample)
        if quat_dot_wxyz(reference, quat) < 0.0:
            quat = [-value for value in quat]
        aligned.append(quat)
    averaged = [
        sum(quat[index] for quat in aligned) / len(aligned) for index in range(4)
    ]
    return normalize_quat_wxyz(averaged)


def format_feature_preview(features: dict[str, float]) -> str:
    return ",".join(f"{finger}={features.get(finger, 0.0):.3f}" for finger in FINGERS)


def format_quat_preview(quat: list[float] | None) -> str:
    if quat is None:
        return "none"
    return "[" + ",".join(f"{value:.3f}" for value in quat) + "]"


def retarget_side(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    robot_side: str,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, float]]:
    if args.retargeting_mode == "vector":
        return retarget_side_vectors(
            features, open_features, closed_features, robot_side, args
        )

    curls = {
        finger: normalize_feature(features, open_features, closed_features, finger)
        for finger in FINGERS
    }
    spreads = {
        finger: normalize_feature(
            features, open_features, closed_features, f"{finger}_brake"
        )
        for finger in FINGERS
    }
    joint_values = {
        f"{robot_side}_thumb_cmc_roll": limit_lerp(
            f"{robot_side}_thumb_cmc_roll", args.thumb_roll_gain * curls["thumb"]
        ),
        f"{robot_side}_thumb_cmc_yaw": limit_lerp(
            f"{robot_side}_thumb_cmc_yaw", args.thumb_yaw_gain * curls["thumb"]
        ),
        f"{robot_side}_thumb_cmc_pitch": limit_lerp(
            f"{robot_side}_thumb_cmc_pitch", curls["thumb"]
        ),
        f"{robot_side}_index_mcp_roll": limit_lerp(
            f"{robot_side}_index_mcp_roll", args.spread_scale * spreads["index"]
        ),
        f"{robot_side}_index_mcp_pitch": limit_lerp(
            f"{robot_side}_index_mcp_pitch", curls["index"]
        ),
        f"{robot_side}_middle_mcp_pitch": limit_lerp(
            f"{robot_side}_middle_mcp_pitch", curls["middle"]
        ),
        f"{robot_side}_ring_mcp_roll": limit_lerp(
            f"{robot_side}_ring_mcp_roll", args.spread_scale * spreads["ring"]
        ),
        f"{robot_side}_ring_mcp_pitch": limit_lerp(
            f"{robot_side}_ring_mcp_pitch", curls["ring"]
        ),
        f"{robot_side}_pinky_mcp_roll": limit_lerp(
            f"{robot_side}_pinky_mcp_roll", args.spread_scale * spreads["pinky"]
        ),
        f"{robot_side}_pinky_mcp_pitch": limit_lerp(
            f"{robot_side}_pinky_mcp_pitch", curls["pinky"]
        ),
    }
    order = [
        joint for joint in ESROBO_HAND_JOINT_ORDER if joint.startswith(f"{robot_side}_")
    ]
    return [joint_values[joint] for joint in order], curls


def retarget_side_vectors(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    robot_side: str,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, float]]:
    """Map palm-relative human finger vectors onto one hand's ten active robot joints."""
    curls = {
        finger: normalize_feature(
            features, open_features, closed_features, f"vector_{finger}_curl"
        )
        for finger in FINGERS
    }
    thumb_roll = normalize_feature_with_fallback(
        features,
        open_features,
        closed_features,
        "vector_thumb_elevation",
        args.thumb_roll_gain * curls["thumb"],
    )
    thumb_yaw = normalize_feature_with_fallback(
        features,
        open_features,
        closed_features,
        "vector_thumb_spread",
        args.thumb_yaw_gain * curls["thumb"],
    )

    spread_range = math.radians(args.vector_spread_range_deg)
    outward_sign = {"index": 1.0, "ring": -1.0, "pinky": -1.0}
    spreads: dict[str, float] = {}
    for finger, sign in outward_sign.items():
        key = f"vector_{finger}_spread"
        delta = float(features.get(key, open_features.get(key, 0.0))) - float(
            open_features.get(key, 0.0)
        )
        spreads[finger] = clamp01(sign * delta / spread_range) * args.spread_scale

    joint_values = {
        f"{robot_side}_thumb_cmc_roll": limit_lerp(
            f"{robot_side}_thumb_cmc_roll", thumb_roll
        ),
        f"{robot_side}_thumb_cmc_yaw": limit_lerp(
            f"{robot_side}_thumb_cmc_yaw", thumb_yaw
        ),
        f"{robot_side}_thumb_cmc_pitch": limit_lerp(
            f"{robot_side}_thumb_cmc_pitch", curls["thumb"]
        ),
        f"{robot_side}_index_mcp_roll": limit_lerp(
            f"{robot_side}_index_mcp_roll", spreads["index"]
        ),
        f"{robot_side}_index_mcp_pitch": limit_lerp(
            f"{robot_side}_index_mcp_pitch", curls["index"]
        ),
        f"{robot_side}_middle_mcp_pitch": limit_lerp(
            f"{robot_side}_middle_mcp_pitch", curls["middle"]
        ),
        f"{robot_side}_ring_mcp_roll": limit_lerp(
            f"{robot_side}_ring_mcp_roll", spreads["ring"]
        ),
        f"{robot_side}_ring_mcp_pitch": limit_lerp(
            f"{robot_side}_ring_mcp_pitch", curls["ring"]
        ),
        f"{robot_side}_pinky_mcp_roll": limit_lerp(
            f"{robot_side}_pinky_mcp_roll", spreads["pinky"]
        ),
        f"{robot_side}_pinky_mcp_pitch": limit_lerp(
            f"{robot_side}_pinky_mcp_pitch", curls["pinky"]
        ),
    }
    order = [
        joint for joint in ESROBO_HAND_JOINT_ORDER if joint.startswith(f"{robot_side}_")
    ]
    diagnostics = dict(curls)
    diagnostics.update({f"{finger}_spread": value for finger, value in spreads.items()})
    return [joint_values[joint] for joint in order], diagnostics


def normalize_feature_with_fallback(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    key: str,
    fallback: float,
) -> float:
    open_value = float(open_features.get(key, 0.0))
    closed_value = float(closed_features.get(key, open_value))
    if abs(closed_value - open_value) < 1.0e-4:
        return clamp01(fallback)
    return normalize_feature(features, open_features, closed_features, key)


def normalize_feature(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    key: str,
) -> float:
    open_value = float(open_features.get(key, 0.0))
    closed_value = float(closed_features.get(key, open_value))
    denominator = closed_value - open_value
    if abs(denominator) < 1.0e-6:
        return 0.0
    value = float(features.get(key, open_value))
    return clamp01((value - open_value) / denominator)


def limit_lerp(joint_name: str, normalized_value: float) -> float:
    lower, upper = ESROBO_HAND_JOINT_LIMITS[joint_name]
    return lower + clamp01(normalized_value) * (upper - lower)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def orientation_delta_wxyz(
    current: list[float] | None, neutral: Any
) -> list[float] | None:
    if current is None or neutral is None:
        return None
    try:
        neutral_quat = [float(value) for value in neutral]
    except (TypeError, ValueError):
        return None
    current_quat = normalize_quat_wxyz(current)
    neutral_quat = normalize_quat_wxyz(neutral_quat)
    return normalize_quat_wxyz(
        quat_multiply_wxyz(current_quat, quat_conjugate_wxyz(neutral_quat))
    )


def normalize_quat_wxyz(raw: list[float]) -> list[float]:
    values = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in values]


def normalize_quat_xyzw(raw: list[float]) -> list[float] | None:
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return None
    return [value / norm for value in values]


def quat_dot_wxyz(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def quat_conjugate_wxyz(quat: list[float]) -> list[float]:
    w, x, y, z = quat
    return [w, -x, -y, -z]


def quat_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_multiply_xyzw(left: list[float], right: list[float]) -> list[float]:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from senseglove_msgs.msg import SenseGloveState
    except ImportError as exc:
        print(
            f"[senseglove_bridge] Missing ROS 2 Python module: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[senseglove_bridge] Source /opt/ros/humble/setup.bash and the built senseglove_ros install/setup.bash.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node("senseglove_to_esrobo_hand_bridge")
    qos_profile = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=args.qos_depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    bridge = SenseGloveUdpBridge(node, SenseGloveState, args, qos_profile)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[senseglove_bridge] Subscribing left topic: {args.left_topic}", flush=True)
    print(
        f"[senseglove_bridge] Subscribing right topic: {args.right_topic}", flush=True
    )
    print(
        f"[senseglove_bridge] Subscription QoS: best_effort keep_last depth={args.qos_depth}.",
        flush=True,
    )
    print(
        f"[senseglove_bridge] Sending ESROBO hand joints to UDP {args.host}:{args.port}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Hand retargeting: "
        f"mode={args.retargeting_mode}, robot_active_joints=10/hand, "
        f"vector_spread_range={args.vector_spread_range_deg:.1f}deg, spread_scale={args.spread_scale:.2f}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Hand IMU orientation: "
        f"{'disabled' if args.disable_imu_orientation else 'enabled'} "
        f"(correction={args.imu_correction}, w_repair={args.imu_w_repair}).",
        flush=True,
    )
    print(
        "[senseglove_bridge] Robot hand mapping: "
        f"left<={'right_glove' if args.swap_left_right_targets else 'left_glove'} "
        f"right<={'left_glove' if args.swap_left_right_targets else 'right_glove'}.",
        flush=True,
    )
    if args.record_path is not None:
        print(
            "[senseglove_bridge] Hand recording requested: "
            f"path={args.record_path.expanduser()} start_delay={args.record_start_delay_s:.1f}s "
            f"duration={'until Ctrl-C' if args.record_duration_s <= 0.0 else f'{args.record_duration_s:.1f}s'}.",
            flush=True,
        )

    record_writer: HandRecordingWriter | None = None
    try:
        calibration = load_or_collect_calibration(bridge, node, args)
        if args.record_path is not None:
            record_writer = HandRecordingWriter(args.record_path, args, calibration)
        period = 1.0 / args.rate_hz
        next_send = time.monotonic()
        next_print = 0.0
        sent_count = 0
        last_print_sent = 0
        last_print_time = time.monotonic()
        while rclpy.ok():
            now = time.monotonic()
            spin_timeout = min(max(next_send - now, 0.0), 0.01)
            spin_once(node, timeout_sec=spin_timeout)
            now = time.monotonic()
            if now < next_send:
                continue
            next_send += period
            if next_send < now - period:
                next_send = now + period

            if not bridge.has_both_hands(args.max_state_age):
                if args.print_interval > 0.0 and now >= next_print:
                    print(
                        f"[senseglove_bridge] Waiting for fresh glove data: {bridge.topic_status()}.",
                        flush=True,
                    )
                    next_print = now + args.print_interval
                continue

            targets, curls, orientation_deltas, packet = bridge.send(sock, calibration)
            sent_count += 1
            recording_status = "off"
            if record_writer is not None:
                recording_status = record_writer.update(
                    now, packet, bridge.raw_recording_snapshot()
                )

            if args.print_interval > 0.0 and now >= next_print:
                elapsed = max(now - last_print_time, 1.0e-6)
                hz = (sent_count - last_print_sent) / elapsed
                print(
                    "[senseglove_bridge] "
                    f"sent={sent_count} hz={hz:.1f} "
                    f"imu={'on' if orientation_deltas else 'off'} "
                    f"imu_w_repair=left:{bridge.left.imu_w_repaired}/right:{bridge.right.imu_w_repaired} "
                    f"skeleton={'sgcore_5x4' if len(bridge.left.hand_positions_mm) == 20 and len(bridge.right.hand_positions_mm) == 20 else 'unavailable'} "
                    f"recording={recording_status} "
                    f"left_curl={format_feature_preview(curls['left'])} "
                    f"right_curl={format_feature_preview(curls['right'])}",
                    flush=True,
                )
                if args.print_raw:
                    print(
                        "[senseglove_bridge raw] "
                        f"left={format_feature_preview(bridge.left.features)} "
                        f"right={format_feature_preview(bridge.right.features)} "
                        f"targets0={[round(value, 4) for value in targets[:6]]} "
                        f"left_imu_delta={format_quat_preview(orientation_deltas.get('left'))} "
                        f"right_imu_delta={format_quat_preview(orientation_deltas.get('right'))}",
                        flush=True,
                    )
                last_print_time = now
                last_print_sent = sent_count
                next_print = now + args.print_interval
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\n[senseglove_bridge] stopped.", flush=True)
    except RuntimeError as exc:
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
        print("\n[senseglove_bridge] stopped during ROS 2 shutdown.", flush=True)
    except OSError as exc:
        print(f"[senseglove_bridge] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if record_writer is not None:
            record_writer.close()
            if record_writer.sequence > 0 and not record_writer.complete:
                duration = (
                    time.monotonic() - record_writer.started_monotonic
                    if record_writer.started_monotonic is not None
                    else 0.0
                )
                print(
                    "[senseglove_bridge] RECORDING SAVED: "
                    f"path={record_writer.path} frames={record_writer.sequence} duration={duration:.3f}s.",
                    flush=True,
                )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
