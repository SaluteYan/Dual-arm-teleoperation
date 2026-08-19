#!/usr/bin/env python3
"""Generate time-aligned synthetic SenseGlove hand targets for an ESROBO body recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from typing import Any

from senseglove_ros_to_esrobo_hand_bridge import (
    ESROBO_HAND_JOINT_LIMITS,
    ESROBO_HAND_JOINT_ORDER,
    FINGERS,
    canonicalize_hand_points_for_robot,
    extract_hand_vector_features,
    retarget_side_vectors,
)


GESTURE_KEYS = (
    (0.0, 0.0, 0.0, "both_open"),
    (3.0, 0.0, 0.0, "both_open"),
    (5.5, 1.0, 0.0, "left_close"),
    (7.0, 1.0, 0.0, "left_hold"),
    (9.5, 0.0, 1.0, "right_close"),
    (11.0, 0.0, 1.0, "right_hold"),
    (13.0, 0.0, 0.0, "both_open"),
    (16.0, 1.0, 1.0, "both_close"),
    (18.0, 1.0, 1.0, "both_hold"),
    (20.0, 0.0, 0.0, "both_open"),
    (23.0, 0.75, 0.25, "left_dominant"),
    (25.0, 0.25, 0.75, "right_dominant"),
    (27.5, 0.85, 0.85, "both_close"),
    (30.0, 0.0, 0.0, "both_open"),
)

# Relative wrist orientation in the synthetic IMU frame, expressed as roll/pitch/yaw degrees.
WRIST_ORIENTATION_KEYS = (
    (0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
    (3.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
    (6.0, (35.0, 0.0, 0.0), (0.0, 0.0, 0.0), "left_roll"),
    (9.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
    (12.0, (0.0, 0.0, 0.0), (-35.0, 0.0, 0.0), "right_roll"),
    (15.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
    (18.0, (0.0, 30.0, 0.0), (0.0, 30.0, 0.0), "both_pitch"),
    (21.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
    (24.0, (0.0, 0.0, 30.0), (0.0, 0.0, -30.0), "mirrored_yaw"),
    (27.0, (20.0, -20.0, 20.0), (-20.0, -20.0, -20.0), "combined"),
    (30.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "wrists_neutral"),
)

FINGER_SPREAD_KEYS = (
    (0.0, 0.0, 0.0),
    (4.0, 1.0, 0.0),
    (7.0, 0.0, 0.0),
    (10.0, 0.0, 1.0),
    (13.0, 0.0, 0.0),
    (21.0, 1.0, 1.0),
    (24.0, 0.0, 0.0),
    (30.0, 0.0, 0.0),
)

VECTOR_RETARGETING_ARGS = SimpleNamespace(
    retargeting_mode="vector",
    vector_spread_range_deg=18.0,
    spread_scale=1.0,
    thumb_roll_gain=0.35,
    thumb_yaw_gain=0.50,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-record-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_body_timeline(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    packets: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "esrobo_body_recording_header":
                header = item
            elif item.get("type") == "esrobo_body_packet" and isinstance(
                item.get("packet"), dict
            ):
                packets.append(item)
            elif item.get("packet") is not None and "t_rel_s" in item:
                packets.append(item)
    if header is None:
        raise ValueError(f"body recording has no header: {path}")
    if not packets:
        raise ValueError(f"body recording has no packets: {path}")
    for index, item in enumerate(packets):
        if "t_rel_s" not in item:
            raise ValueError(f"body packet {index} has no t_rel_s")
    return header, packets


def _smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def _gesture_at(time_s: float) -> tuple[float, float, str]:
    if time_s <= GESTURE_KEYS[0][0]:
        return GESTURE_KEYS[0][1], GESTURE_KEYS[0][2], GESTURE_KEYS[0][3]
    for left_key, right_key in zip(GESTURE_KEYS, GESTURE_KEYS[1:]):
        if time_s <= right_key[0]:
            span = max(right_key[0] - left_key[0], 1.0e-6)
            blend = _smoothstep((time_s - left_key[0]) / span)
            left_curl = left_key[1] + blend * (right_key[1] - left_key[1])
            right_curl = left_key[2] + blend * (right_key[2] - left_key[2])
            return left_curl, right_curl, right_key[3]
    return GESTURE_KEYS[-1][1], GESTURE_KEYS[-1][2], GESTURE_KEYS[-1][3]


def _wrist_orientation_at(
    time_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    if time_s <= WRIST_ORIENTATION_KEYS[0][0]:
        return (
            WRIST_ORIENTATION_KEYS[0][1],
            WRIST_ORIENTATION_KEYS[0][2],
            WRIST_ORIENTATION_KEYS[0][3],
        )
    for left_key, right_key in zip(WRIST_ORIENTATION_KEYS, WRIST_ORIENTATION_KEYS[1:]):
        if time_s <= right_key[0]:
            span = max(right_key[0] - left_key[0], 1.0e-6)
            blend = _smoothstep((time_s - left_key[0]) / span)
            left_rpy = tuple(
                a + blend * (b - a) for a, b in zip(left_key[1], right_key[1])
            )
            right_rpy = tuple(
                a + blend * (b - a) for a, b in zip(left_key[2], right_key[2])
            )
            return left_rpy, right_rpy, right_key[3]
    return (
        WRIST_ORIENTATION_KEYS[-1][1],
        WRIST_ORIENTATION_KEYS[-1][2],
        WRIST_ORIENTATION_KEYS[-1][3],
    )


def _spread_at(time_s: float) -> tuple[float, float]:
    if time_s <= FINGER_SPREAD_KEYS[0][0]:
        return FINGER_SPREAD_KEYS[0][1], FINGER_SPREAD_KEYS[0][2]
    for left_key, right_key in zip(FINGER_SPREAD_KEYS, FINGER_SPREAD_KEYS[1:]):
        if time_s <= right_key[0]:
            span = max(right_key[0] - left_key[0], 1.0e-6)
            blend = _smoothstep((time_s - left_key[0]) / span)
            return (
                left_key[1] + blend * (right_key[1] - left_key[1]),
                left_key[2] + blend * (right_key[2] - left_key[2]),
            )
    return FINGER_SPREAD_KEYS[-1][1], FINGER_SPREAD_KEYS[-1][2]


def _rpy_degrees_to_quat_wxyz(rpy_degrees: tuple[float, float, float]) -> list[float]:
    roll, pitch, yaw = (math.radians(value) for value in rpy_degrees)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def _synthetic_hand_points(
    side: str, closure: float, spread: float
) -> list[list[float]]:
    """Generate mirrored SGCore-layout points in millimeters for vector-retargeting tests."""
    closure = min(max(float(closure), 0.0), 1.0)
    spread = min(max(float(spread), 0.0), 1.0)
    mirror = 1.0 if side == "left" else -1.0
    lateral = (0.0, mirror, 0.0)
    normal = (mirror, 0.0, 0.0)
    forward = (0.0, 0.0, 1.0)
    # SGCore hand points and the segment lengths below are expressed in millimeters.
    # Keep the palm anchors in the same unit so the five chains remain separated.
    bases = {
        "thumb": (40.0, 12.0),
        "index": (32.0, 60.0),
        "middle": (0.0, 65.0),
        "ring": (-20.0, 58.0),
        "pinky": (-38.0, 50.0),
    }
    curl_scales = {
        "thumb": 0.95,
        "index": 1.0,
        "middle": 1.02,
        "ring": 0.98,
        "pinky": 0.92,
    }
    segment_lengths = {
        "thumb": (36.0, 28.0, 22.0),
        "index": (42.0, 28.0, 21.0),
        "middle": (45.0, 30.0, 22.0),
        "ring": (42.0, 28.0, 21.0),
        "pinky": (36.0, 24.0, 18.0),
    }
    spread_signs = {
        "thumb": 1.0,
        "index": 1.0,
        "middle": 0.0,
        "ring": -1.0,
        "pinky": -1.0,
    }
    points: list[list[float]] = []
    for finger in FINGERS:
        base_lateral, base_forward = bases[finger]
        current = [
            base_lateral * lateral[axis] + base_forward * forward[axis]
            for axis in range(3)
        ]
        points.append(current.copy())
        finger_curl = min(closure * curl_scales[finger], 1.0)
        cumulative_angles = (0.55 * finger_curl, 1.10 * finger_curl, 1.55 * finger_curl)
        lateral_component = math.tan(math.radians(16.0) * spread * spread_signs[finger])
        if finger == "thumb":
            lateral_component += 0.55 - 0.35 * finger_curl
        for length, angle in zip(segment_lengths[finger], cumulative_angles):
            direction = [
                math.cos(angle) * forward[axis]
                + math.sin(angle) * normal[axis]
                + lateral_component * lateral[axis]
                for axis in range(3)
            ]
            direction_norm = math.sqrt(sum(value * value for value in direction))
            direction = [value / direction_norm for value in direction]
            current = [current[axis] + length * direction[axis] for axis in range(3)]
            points.append(current.copy())
    return points


def _vector_calibration(side: str) -> tuple[dict[str, float], dict[str, float]]:
    return (
        extract_hand_vector_features(_synthetic_hand_points(side, 0.0, 0.0)),
        extract_hand_vector_features(_synthetic_hand_points(side, 1.0, 0.0)),
    )


def _side_targets(
    side: str,
    closure: float,
    spread: float,
    open_features: dict[str, float],
    closed_features: dict[str, float],
) -> tuple[list[float], dict[str, float], list[list[float]]]:
    points = _synthetic_hand_points(side, closure, spread)
    targets, diagnostics = retarget_side_vectors(
        extract_hand_vector_features(points),
        open_features,
        closed_features,
        side,
        VECTOR_RETARGETING_ARGS,
    )
    return targets, diagnostics, points


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(body_path: Path, output_path: Path, overwrite: bool) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path}; pass --overwrite to replace it"
        )
    body_header, body_items = _load_body_timeline(body_path)
    times = [float(item["t_rel_s"]) for item in body_items]
    periods = [
        current - previous
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    median_rate_hz = 1.0 / statistics.median(periods) if periods else 0.0
    duration_s = times[-1] - times[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "esrobo_hand_recording_header",
        "version": 1,
        "source": "synthetic_senseglove_retargeted",
        "synthetic": True,
        "description": (
            "Deterministic dual-hand and wrist-orientation integration-test gestures; "
            "not physical SenseGlove measurements."
        ),
        "body_recording": str(body_path),
        "body_recording_sha256": _file_sha256(body_path),
        "aligned_packet_count": len(body_items),
        "duration_s": duration_s,
        "median_rate_hz": median_rate_hz,
        "time_field": "t_rel_s",
        "packet_field": "packet",
        "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
        "hand_joint_limits": {
            name: list(ESROBO_HAND_JOINT_LIMITS[name])
            for name in ESROBO_HAND_JOINT_ORDER
        },
        "imu_orientation_included": True,
        "imu_orientation_kind": "synthetic_calibrated_relative_delta",
        "imu_orientation_order": "wxyz",
        "hand_retargeting_method": "calibrated_hand_vectors_to_esrobo_10_active_joints_per_hand",
        "hand_skeleton_included": True,
        "hand_skeleton_layout": "finger_major_5x4_thumb_to_pinky_proximal_to_distal",
        "gesture_keys": [
            {
                "t_rel_s": time_s,
                "left_closure": left,
                "right_closure": right,
                "label": label,
            }
            for time_s, left, right, label in GESTURE_KEYS
        ],
        "wrist_orientation_keys": [
            {
                "t_rel_s": time_s,
                "left_rpy_deg": list(left_rpy),
                "right_rpy_deg": list(right_rpy),
                "label": label,
            }
            for time_s, left_rpy, right_rpy, label in WRIST_ORIENTATION_KEYS
        ],
        "body_neutral_start_s": body_header.get("neutral_start_s"),
    }

    left_open, left_closed = _vector_calibration("left")
    right_open, right_closed = _vector_calibration("right")
    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, separators=(",", ":")) + "\n")
        for sequence, (time_s, body_item) in enumerate(zip(times, body_items)):
            left_closure, right_closure, gesture = _gesture_at(time_s)
            left_spread, right_spread = _spread_at(time_s)
            left_wrist_rpy, right_wrist_rpy, wrist_gesture = _wrist_orientation_at(
                time_s
            )
            left_targets, left_curls, left_points = _side_targets(
                "left", left_closure, left_spread, left_open, left_closed
            )
            right_targets, right_curls, right_points = _side_targets(
                "right", right_closure, right_spread, right_open, right_closed
            )
            packet = {
                "timestamp": body_item.get("packet", {}).get("timestamp"),
                "source": "synthetic_senseglove_retargeted",
                "type": "esrobo_hand_joints",
                "sequence": sequence,
                "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
                "hand_joints": left_targets + right_targets,
                "hand_skeleton_positions": {
                    "left": canonicalize_hand_points_for_robot(left_points, "left"),
                    "right": canonicalize_hand_points_for_robot(right_points, "right"),
                },
                "hand_skeleton_layout": "finger_major_5x4_thumb_to_pinky_proximal_to_distal",
                "hand_skeleton_unit": "mm",
                "hand_skeleton_frame": "robot_hand_local",
                "hand_skeleton_source": "synthetic_mirrored_hand_vectors",
                "retargeting": {
                    "method": "calibrated_hand_vectors_to_esrobo_10_active_joints_per_hand",
                    "robot_left_source": "left_synthetic_hand_vectors",
                    "robot_right_source": "right_synthetic_hand_vectors",
                },
                "hand_orientation_deltas": {
                    "left": _rpy_degrees_to_quat_wxyz(left_wrist_rpy),
                    "right": _rpy_degrees_to_quat_wxyz(right_wrist_rpy),
                },
                "hand_orientation_delta_order": "wxyz",
                "hand_orientation_source_frame": "synthetic_robot_aligned",
                "hand_orientation_retargeting": {
                    "method": "synthetic_calibrated_imu_delta",
                    "robot_left_source": "left_synthetic_imu",
                    "robot_right_source": "right_synthetic_imu",
                },
                "curls": {"left": left_curls, "right": right_curls},
                "synthetic_gesture": gesture,
                "synthetic_closure": {"left": left_closure, "right": right_closure},
                "synthetic_spread": {"left": left_spread, "right": right_spread},
                "synthetic_wrist_gesture": wrist_gesture,
                "synthetic_wrist_rpy_deg": {
                    "left": list(left_wrist_rpy),
                    "right": list(right_wrist_rpy),
                },
            }
            item = {
                "type": "esrobo_hand_packet",
                "record_sequence": sequence,
                "t_rel_s": time_s,
                "record_phase": body_item.get("record_phase", "motion"),
                "packet": packet,
            }
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    return header


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        header = generate(
            args.body_record_path.expanduser(),
            args.output_path.expanduser(),
            args.overwrite,
        )
    except Exception as exc:
        print(f"[generate_senseglove_recording] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[generate_senseglove_recording] "
        f"saved={args.output_path} packets={header['aligned_packet_count']} "
        f"duration={header['duration_s']:.3f}s median_rate={header['median_rate_hz']:.1f}Hz",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
