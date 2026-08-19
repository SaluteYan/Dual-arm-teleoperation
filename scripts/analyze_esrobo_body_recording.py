#!/usr/bin/env python3
"""Analyze recorded PICO/XRoboToolkit body data and ESROBO retargeted targets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


BODY_FRAME_ALIASES = {
    "pelvis": "waist",
    "waist": "waist",
    "spine1": "waist",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "neck": "neck",
    "head": "head",
}

DEFAULT_SOURCE_TO_ROBOT_ROTATION = (
    0.0,
    0.0,
    -1.0,
    -1.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
)
DEFAULT_POSITION_DELTA_SIGNS = (1.0, 1.0, 1.0)
DEFAULT_ROOT_Z_OFFSET = 0.3364894688129425
DEFAULT_ROBOT_LEFT_SHOULDER_POSITION = (-0.146646, 0.245584, 1.529004)
DEFAULT_ROBOT_RIGHT_SHOULDER_POSITION = (-0.146803, -0.235015, 1.529003)
DEFAULT_INITIAL_LEFT_ELBOW_POSITION = (-0.146645, 0.245585, 1.219004)
DEFAULT_INITIAL_RIGHT_ELBOW_POSITION = (-0.146802, -0.235014, 1.219003)
DEFAULT_INITIAL_LEFT_WRIST_POSITION = (-0.14664, 0.24538, 0.48300)
DEFAULT_INITIAL_RIGHT_WRIST_POSITION = (-0.14680, -0.23480, 0.48300)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _parse_floats(raw: str | None, expected_len: int, default: tuple[float, ...], name: str) -> tuple[float, ...]:
    if raw is None or not raw.strip():
        return default
    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} numbers, got {len(values)} from {raw!r}")
    return values


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-path", required=True, help="JSONL recording from xrobotoolkit_body_udp_bridge.py.")
    parser.add_argument(
        "--swap-left-right-targets",
        type=int,
        default=1 if _env_bool("ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS", False) else 0,
        choices=(0, 1),
        help="Use the same robot-left/robot-right target swap setting as IsaacLab. Default reads env or 0.",
    )
    parser.add_argument(
        "--position-delta-signs",
        default=os.environ.get("ESROBO_BODY_POSITION_DELTA_SIGNS", "1 1 1"),
        help="Three robot-frame signs applied to wrist/reach deltas, for example '1 1 1' or '-1 -1 1'.",
    )
    parser.add_argument(
        "--source-to-robot-rotation",
        default=os.environ.get("ESROBO_BODY_SOURCE_TO_ROBOT_ROTATION", ""),
        help="Nine numbers, row-major source-to-robot rotation. Empty uses the project default.",
    )
    parser.add_argument(
        "--use-waist-frame",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("ESROBO_BODY_USE_WAIST_FRAME", "0").strip().lower()
        not in ("0", "false", "no", "off"),
        help="Analyze arm points relative to waist instead of using PICO absolute body directions.",
    )
    parser.add_argument("--position-scale", type=float, default=float(os.environ.get("ESROBO_BODY_POSITION_SCALE", "1.0")))
    parser.add_argument("--reference-phase", default="neutral_start")
    parser.add_argument("--reference-start-s", type=float, default=0.75)
    parser.add_argument("--reference-end-s", type=float, default=2.0)
    parser.add_argument("--motion-phase", default="motion")
    parser.add_argument(
        "--arm-vector-position-mode",
        default=os.environ.get("ESROBO_BODY_ARM_VECTOR_POSITION_MODE", "segment_direction_absolute"),
        choices=("segment_direction_absolute", "segment_direction_relative", "segment_direction", "delta"),
        help="Match IsaacLab body arm-vector position retargeting mode.",
    )
    parser.add_argument(
        "--compare-both-swaps",
        action="store_true",
        default=True,
        help="Print both swap=0 and swap=1 target mappings for direction comparison.",
    )
    return parser.parse_args(argv)


def _quat_xyzw_to_wxyz(quat_xyzw: Any) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1.0e-10:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _invert_pose_matrix(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def _packet_time(item: dict[str, Any], fallback_index: int) -> float:
    for key in ("t_rel_s", "record_t_rel_s"):
        value = item.get(key)
        if value is not None:
            return float(value)
    packet = item.get("packet", item)
    if isinstance(packet, dict):
        sender_ns = packet.get("sender_monotonic_ns")
        if sender_ns is not None:
            return float(sender_ns) / 1.0e9
        timestamp = packet.get("timestamp")
        if timestamp is not None:
            return float(timestamp)
    return float(fallback_index)


def _packet_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    packet = item.get("packet")
    if packet is None and ("frames" in item or "joint_names" in item):
        packet = item
    if isinstance(packet, dict):
        return packet
    return None


def _frame_pose_to_matrix(raw_pose: Any, source_to_robot_rotation: np.ndarray) -> np.ndarray | None:
    if isinstance(raw_pose, dict):
        if not raw_pose.get("valid", True):
            return None
        position = raw_pose.get("pos", raw_pose.get("position"))
        if position is None:
            return None
        if "quat_wxyz" in raw_pose:
            quat = np.asarray(raw_pose["quat_wxyz"], dtype=np.float64).reshape(4)
        elif "quat_xyzw" in raw_pose:
            quat = _quat_xyzw_to_wxyz(raw_pose["quat_xyzw"])
        else:
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        values = np.asarray(raw_pose, dtype=np.float64).reshape(-1)
        if values.shape[0] != 7:
            return None
        position = values[:3]
        quat = values[3:]

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = source_to_robot_rotation @ _quat_wxyz_to_matrix(quat) @ source_to_robot_rotation.T
    matrix[:3, 3] = source_to_robot_rotation @ np.asarray(position, dtype=np.float64).reshape(3)
    return matrix


def _parse_array_frames(packet: dict[str, Any], source_to_robot_rotation: np.ndarray) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    joint_names = packet.get("joint_names")
    joint_positions = packet.get("joint_positions")
    if joint_names is None or joint_positions is None:
        return frames
    positions = np.asarray(joint_positions, dtype=np.float64)
    orientations = np.asarray(packet.get("joint_orientations", []), dtype=np.float64)
    valid = np.asarray(packet.get("joint_valid", np.ones(len(joint_names))), dtype=bool)
    for index, raw_name in enumerate(joint_names):
        if index >= len(positions) or index >= len(valid) or not valid[index]:
            continue
        frame_name = BODY_FRAME_ALIASES.get(str(raw_name).lower())
        if frame_name is None:
            continue
        if orientations.ndim == 2 and orientations.shape[0] > index and orientations.shape[1] >= 4:
            quat = _quat_xyzw_to_wxyz(orientations[index, :4])
        else:
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = source_to_robot_rotation @ _quat_wxyz_to_matrix(quat) @ source_to_robot_rotation.T
        matrix[:3, 3] = source_to_robot_rotation @ positions[index, :3]
        frames[frame_name] = matrix
    return frames


def _parse_packet_frames(packet: dict[str, Any], source_to_robot_rotation: np.ndarray) -> dict[str, np.ndarray]:
    frames = _parse_array_frames(packet, source_to_robot_rotation)
    raw_frames = packet.get("frames")
    if isinstance(raw_frames, dict):
        for raw_name, raw_pose in raw_frames.items():
            frame_name = BODY_FRAME_ALIASES.get(str(raw_name).lower())
            if frame_name is None:
                continue
            matrix = _frame_pose_to_matrix(raw_pose, source_to_robot_rotation)
            if matrix is not None:
                frames[frame_name] = matrix
    return frames


def _arm_points_from_packet(
    packet: dict[str, Any],
    source_to_robot_rotation: np.ndarray,
    use_waist_frame: bool,
) -> dict[str, dict[str, np.ndarray]] | None:
    frames = _parse_packet_frames(packet, source_to_robot_rotation)
    required = [
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ]
    if use_waist_frame:
        required.append("waist")
    if any(name not in frames for name in required):
        return None

    world_to_body = _invert_pose_matrix(frames["waist"]) if use_waist_frame else np.eye(4, dtype=np.float64)
    points: dict[str, dict[str, np.ndarray]] = {}
    for side in ("left", "right"):
        side_points = {}
        for joint in ("shoulder", "elbow", "wrist"):
            side_points[joint] = (world_to_body @ frames[f"{side}_{joint}"])[:3, 3]
        points[side] = side_points
    return points


def _load_samples(
    path: Path,
    source_to_robot_rotation: np.ndarray,
    use_waist_frame: bool,
) -> list[tuple[float, str, dict[str, dict[str, np.ndarray]]]]:
    samples: list[tuple[float, str, dict[str, dict[str, np.ndarray]]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            packet = _packet_from_item(item)
            if packet is None:
                continue
            points = _arm_points_from_packet(packet, source_to_robot_rotation, use_waist_frame)
            if points is None:
                continue
            samples.append((_packet_time(item, len(samples)), str(item.get("record_phase", "unknown")), points))

    if not samples:
        raise ValueError(f"no usable full-arm body samples found in {path}")
    first_time = samples[0][0]
    return [(max(0.0, t - first_time), phase, points) for t, phase, points in samples]


def _mean_points(samples: list[dict[str, dict[str, np.ndarray]]]) -> dict[str, dict[str, np.ndarray]]:
    return {
        side: {
            joint: np.mean([sample[side][joint] for sample in samples], axis=0)
            for joint in ("shoulder", "elbow", "wrist")
        }
        for side in ("left", "right")
    }


def _fmt(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{value: .4f}" for value in vec) + "]"


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-10:
        return None
    return vector / norm


def _scale_vector_to_length(vector: np.ndarray, length: float, fallback: np.ndarray) -> np.ndarray:
    direction = _normalize_vector(vector)
    if direction is None:
        direction = _normalize_vector(fallback)
    if direction is None:
        return np.zeros(3, dtype=np.float64)
    return direction * float(length)


def _arm_points_relative_to_shoulder(
    points: dict[str, np.ndarray], position_delta_signs: np.ndarray
) -> dict[str, np.ndarray]:
    shoulder = points["shoulder"]
    return {
        "shoulder": np.zeros(3, dtype=np.float64),
        "elbow": (points["elbow"] - shoulder) * position_delta_signs,
        "wrist": (points["wrist"] - shoulder) * position_delta_signs,
    }


def _arm_points_to_rotation(points: dict[str, np.ndarray]) -> np.ndarray | None:
    reach_axis = _normalize_vector(points["wrist"] - points["shoulder"])
    upper_axis = _normalize_vector(points["elbow"] - points["shoulder"])
    if reach_axis is None:
        return None

    bend_axis = None
    if upper_axis is not None:
        bend_axis = _normalize_vector(upper_axis - float(np.dot(upper_axis, reach_axis)) * reach_axis)
    if bend_axis is None:
        for reference in (
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        ):
            bend_axis = _normalize_vector(reference - float(np.dot(reference, reach_axis)) * reach_axis)
            if bend_axis is not None:
                break
    if bend_axis is None:
        return None

    plane_normal = _normalize_vector(np.cross(bend_axis, reach_axis))
    if plane_normal is None:
        return None
    side_axis = _normalize_vector(np.cross(plane_normal, reach_axis))
    if side_axis is None:
        return None
    return np.stack([reach_axis, side_axis, plane_normal], axis=1)


def _segment_direction_targets(
    sample_points: dict[str, np.ndarray],
    reference_points: dict[str, np.ndarray] | None,
    robot_shoulder: np.ndarray,
    robot_elbow: np.ndarray,
    robot_wrist: np.ndarray,
    position_delta_signs: np.ndarray,
    relative_reference: bool,
) -> tuple[np.ndarray, np.ndarray]:
    signed_sample = _arm_points_relative_to_shoulder(sample_points, position_delta_signs)
    robot_reference = {"shoulder": robot_shoulder, "elbow": robot_elbow, "wrist": robot_wrist}

    alignment_rotation = np.eye(3, dtype=np.float64)
    if relative_reference and reference_points is not None:
        signed_reference = _arm_points_relative_to_shoulder(reference_points, position_delta_signs)
        sample_reference_rotation = _arm_points_to_rotation(signed_reference)
        robot_reference_rotation = _arm_points_to_rotation(robot_reference)
        if sample_reference_rotation is not None and robot_reference_rotation is not None:
            alignment_rotation = robot_reference_rotation @ sample_reference_rotation.T

    upper_length = float(np.linalg.norm(robot_elbow - robot_shoulder))
    forearm_length = float(np.linalg.norm(robot_wrist - robot_elbow))
    current_upper = signed_sample["elbow"] - signed_sample["shoulder"]
    current_forearm = signed_sample["wrist"] - signed_sample["elbow"]
    upper_vector = _scale_vector_to_length(current_upper @ alignment_rotation.T, upper_length, robot_elbow - robot_shoulder)
    forearm_vector = _scale_vector_to_length(
        current_forearm @ alignment_rotation.T,
        forearm_length,
        robot_wrist - robot_elbow,
    )
    target_elbow = robot_shoulder + upper_vector
    target_wrist = target_elbow + forearm_vector
    return target_elbow, target_wrist


def _print_delta_stats(label: str, values: np.ndarray) -> None:
    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    mean = values.mean(axis=0)
    print(f"{label}: min={_fmt(minimum)} max={_fmt(maximum)} range={_fmt(maximum - minimum)} mean={_fmt(mean)}")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.arm_vector_position_mode == "segment_direction":
        args.arm_vector_position_mode = "segment_direction_absolute"
    path = Path(args.record_path).expanduser()
    if not path.exists():
        print(f"[analyze_body_recording] ERROR: recording does not exist: {path}", file=sys.stderr)
        return 2

    try:
        source_to_robot_rotation = np.asarray(
            _parse_floats(
                args.source_to_robot_rotation,
                9,
                DEFAULT_SOURCE_TO_ROBOT_ROTATION,
                "source-to-robot rotation",
            ),
            dtype=np.float64,
        ).reshape(3, 3)
        position_delta_signs = np.asarray(
            _parse_floats(args.position_delta_signs, 3, DEFAULT_POSITION_DELTA_SIGNS, "position delta signs"),
            dtype=np.float64,
        ).reshape(3)
        samples = _load_samples(path, source_to_robot_rotation, args.use_waist_frame)
    except Exception as exc:
        print(f"[analyze_body_recording] ERROR: {exc}", file=sys.stderr)
        return 2

    duration = samples[-1][0] if samples else 0.0
    deltas = np.diff([sample[0] for sample in samples])
    positive_deltas = deltas[deltas > 1.0e-6]
    median_hz = 1.0 / float(np.median(positive_deltas)) if positive_deltas.size else 0.0
    phase_counts = Counter(phase for _t, phase, _points in samples)

    reference_samples = [
        points
        for t, phase, points in samples
        if phase == args.reference_phase and args.reference_start_s <= t <= args.reference_end_s
    ]
    if not reference_samples:
        reference_samples = [
            points for t, _phase, points in samples if args.reference_start_s <= t <= args.reference_end_s
        ]
    if not reference_samples:
        reference_samples = [points for _t, _phase, points in samples[: min(len(samples), 120)]]

    motion_samples = [points for _t, phase, points in samples if phase == args.motion_phase]
    if not motion_samples:
        motion_samples = [points for t, _phase, points in samples if t > args.reference_end_s]
    if not motion_samples:
        motion_samples = [points for _t, _phase, points in samples]

    reference = _mean_points(reference_samples)
    print(f"recording={path}")
    print(f"samples={len(samples)} duration={duration:.3f}s median_hz={median_hz:.1f} phases={dict(phase_counts)}")
    print(
        "mapping "
        f"swap_left_right_targets={args.swap_left_right_targets} "
        f"position_delta_signs={_fmt(position_delta_signs)} "
        f"position_scale={args.position_scale:.3f} "
        f"arm_vector_position_mode={args.arm_vector_position_mode} "
        f"use_waist_frame={args.use_waist_frame}"
    )
    print(f"source_to_robot_rotation_det={np.linalg.det(source_to_robot_rotation):.3f}")
    print(
        "reference "
        f"phase={args.reference_phase!r} window={args.reference_start_s:.2f}-{args.reference_end_s:.2f}s "
        f"samples={len(reference_samples)}"
    )
    if args.use_waist_frame:
        print("robot-frame axes below are waist-relative x/y/z after source-to-robot rotation.")
    else:
        print("robot-frame axes below use PICO absolute body directions after source-to-robot rotation.")

    for side in ("left", "right"):
        reach_ref = reference[side]["wrist"] - reference[side]["shoulder"]
        upper_ref = reference[side]["elbow"] - reference[side]["shoulder"]
        fore_ref = reference[side]["wrist"] - reference[side]["elbow"]
        print(
            f"{side}_reference shoulder={_fmt(reference[side]['shoulder'])} "
            f"elbow={_fmt(reference[side]['elbow'])} wrist={_fmt(reference[side]['wrist'])} "
            f"reach={_fmt(reach_ref)} reach_len={np.linalg.norm(reach_ref):.3f} "
            f"upper_len={np.linalg.norm(upper_ref):.3f} forearm_len={np.linalg.norm(fore_ref):.3f}"
        )

    for side in ("left", "right"):
        reach_ref = reference[side]["wrist"] - reference[side]["shoulder"]
        reach_delta = np.asarray(
            [(sample[side]["wrist"] - sample[side]["shoulder"]) - reach_ref for sample in motion_samples]
        )
        wrist_delta = np.asarray([sample[side]["wrist"] - reference[side]["wrist"] for sample in motion_samples])
        elbow_delta = np.asarray([sample[side]["elbow"] - reference[side]["elbow"] for sample in motion_samples])
        print(f"\nhuman_{side} motion deltas:")
        _print_delta_stats("  reach_delta shoulder_to_wrist", reach_delta)
        _print_delta_stats("  wrist_delta", wrist_delta)
        _print_delta_stats("  elbow_delta", elbow_delta)

    swaps = (0, 1) if args.compare_both_swaps else (args.swap_left_right_targets,)
    robot_shoulder_positions = {
        "left": np.asarray(DEFAULT_ROBOT_LEFT_SHOULDER_POSITION, dtype=np.float64),
        "right": np.asarray(DEFAULT_ROBOT_RIGHT_SHOULDER_POSITION, dtype=np.float64),
    }
    initial_elbow_positions = {
        "left": np.asarray(DEFAULT_INITIAL_LEFT_ELBOW_POSITION, dtype=np.float64),
        "right": np.asarray(DEFAULT_INITIAL_RIGHT_ELBOW_POSITION, dtype=np.float64),
    }
    initial_wrist_positions = {
        "left": np.asarray(DEFAULT_INITIAL_LEFT_WRIST_POSITION, dtype=np.float64),
        "right": np.asarray(DEFAULT_INITIAL_RIGHT_WRIST_POSITION, dtype=np.float64),
    }
    root_z_offset = float(os.environ.get("ESROBO_ROOT_Z_OFFSET", str(DEFAULT_ROOT_Z_OFFSET)))
    initial_wrist_positions["left"][2] += root_z_offset
    initial_wrist_positions["right"][2] += root_z_offset
    for swap in swaps:
        mapping = {"left": "right", "right": "left"} if swap else {"left": "left", "right": "right"}
        current_marker = " (current)" if swap == args.swap_left_right_targets else ""
        print(f"\nretargeted robot elbow/wrist targets with swap={swap}{current_marker}:")
        for target_side, source_side in mapping.items():
            reach_ref = reference[source_side]["wrist"] - reference[source_side]["shoulder"]
            upper_ref = reference[source_side]["elbow"] - reference[source_side]["shoulder"]
            if args.arm_vector_position_mode in ("segment_direction_absolute", "segment_direction_relative"):
                relative_reference = args.arm_vector_position_mode == "segment_direction_relative"
                elbow_target_position_rows = []
                wrist_target_position_rows = []
                for sample in motion_samples:
                    elbow_target, wrist_target = _segment_direction_targets(
                        sample[source_side],
                        reference[source_side] if relative_reference else None,
                        robot_shoulder_positions[target_side],
                        initial_elbow_positions[target_side],
                        initial_wrist_positions[target_side],
                        position_delta_signs,
                        relative_reference,
                    )
                    elbow_target_position_rows.append(elbow_target)
                    wrist_target_position_rows.append(wrist_target)
                elbow_target_position = np.asarray(elbow_target_position_rows)
                wrist_target_position = np.asarray(wrist_target_position_rows)
                elbow_target_delta = elbow_target_position - initial_elbow_positions[target_side]
                wrist_target_delta = wrist_target_position - initial_wrist_positions[target_side]
            else:
                wrist_target_delta = np.asarray(
                    [
                        ((sample[source_side]["wrist"] - sample[source_side]["shoulder"]) - reach_ref)
                        * position_delta_signs
                        * args.position_scale
                        for sample in motion_samples
                    ]
                )
                elbow_target_delta = np.asarray(
                    [
                        ((sample[source_side]["elbow"] - sample[source_side]["shoulder"]) - upper_ref)
                        * position_delta_signs
                        * args.position_scale
                        for sample in motion_samples
                    ]
                )
                wrist_target_position = initial_wrist_positions[target_side] + wrist_target_delta
                elbow_target_position = initial_elbow_positions[target_side] + elbow_target_delta
            _print_delta_stats(
                f"  robot_{target_side}_elbow <= human_{source_side} upper_arm_delta", elbow_target_delta
            )
            _print_delta_stats(
                f"  robot_{target_side}_elbow <= human_{source_side} target_position", elbow_target_position
            )
            _print_delta_stats(
                f"  robot_{target_side}_wrist <= human_{source_side} reach_delta", wrist_target_delta
            )
            _print_delta_stats(
                f"  robot_{target_side}_wrist <= human_{source_side} target_position", wrist_target_position
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
