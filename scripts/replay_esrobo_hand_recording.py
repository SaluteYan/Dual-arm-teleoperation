#!/usr/bin/env python3
"""Replay an ESROBO hand recording without sending PICO body packets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from typing import Any

from senseglove_ros_to_esrobo_hand_bridge import (
    DEFAULT_ESROBO_URDF_PATH,
    ESROBO_HAND_JOINT_ORDER,
    ESROBOHandDexVectorRetargeter,
    FINGERS,
    apply_imu_axis_calibration,
    build_fixed_imu_axis_calibration,
    canonicalize_hand_points_for_robot,
    orientation_delta_local_wxyz,
    orientation_delta_wxyz,
)


RecordingFrame = tuple[float, dict[str, Any], dict[str, Any]]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-record-path", required=True, type=Path)
    parser.add_argument("--host", default=os.environ.get("ESROBO_BODY_UDP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050")))
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument(
        "--retargeting-mode",
        choices=("recorded", "dex_vector"),
        default="recorded",
        help=(
            "Replay stored robot targets, or recompute them from each frame's "
            "raw 20-point SenseGlove skeleton using dex_vector."
        ),
    )
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_ESROBO_URDF_PATH)
    parser.add_argument("--dex-iterations", type=int, default=1)
    parser.add_argument("--dex-damping", type=float, default=0.025)
    parser.add_argument("--dex-max-joint-step", type=float, default=0.16)
    parser.add_argument("--dex-normal-delta", type=float, default=0.003)
    parser.add_argument("--dex-jacobian-eps", type=float, default=0.0001)
    parser.add_argument("--dex-direction-weight", type=float, default=0.35)
    parser.add_argument(
        "--dex-vector-set", choices=("fingertip", "dense"), default="fingertip"
    )
    parser.add_argument("--dex-curl-weight", type=float, default=0.015)
    parser.add_argument("--dex-huber-delta", type=float, default=0.025)
    parser.add_argument("--dex-joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--dex-joint-limit-weight", type=float, default=0.001)
    return parser.parse_args(argv)


def load_recording(path: Path) -> tuple[dict[str, Any] | None, list[RecordingFrame]]:
    header: dict[str, Any] | None = None
    frames: list[RecordingFrame] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("type") == "esrobo_hand_recording_header":
                header = item
                continue
            packet = item.get("packet")
            if not isinstance(packet, dict) or "t_rel_s" not in item:
                continue
            if packet.get("type") != "esrobo_hand_joints":
                raise ValueError(f"line {line_number} is not an esrobo_hand_joints packet")
            raw = item.get("senseglove_raw")
            frames.append(
                (
                    float(item["t_rel_s"]),
                    packet,
                    raw if isinstance(raw, dict) else {},
                )
            )
    if not frames:
        raise ValueError(f"recording contains no hand packets: {path}")
    return header, frames


def select_packets(
    frames: list[RecordingFrame], start_offset: float, duration: float
) -> list[RecordingFrame]:
    end_time = start_offset + duration if duration > 0.0 else None
    selected = [
        (packet_time, packet, raw)
        for packet_time, packet, raw in frames
        if packet_time >= start_offset and (end_time is None or packet_time <= end_time)
    ]
    if not selected:
        raise ValueError("no packets selected by --start-offset/--duration")
    base_time = selected[0][0]
    return [
        (packet_time - base_time, packet, raw)
        for packet_time, packet, raw in selected
    ]


def _valid_raw_side(raw_side: Any) -> bool:
    return (
        isinstance(raw_side, dict)
        and len(raw_side.get("hand_position_mm_used", [])) == 20
        and isinstance(raw_side.get("retargeting_features"), dict)
    )


def _select_open_reference(
    frames: list[RecordingFrame], side: str
) -> tuple[int, list[list[float]]]:
    candidates = [
        (index, raw[side])
        for index, (_packet_time, _packet, raw) in enumerate(frames)
        if _valid_raw_side(raw.get(side))
    ]
    if not candidates:
        raise ValueError(f"recording has no valid raw 20-point {side} hand frames")
    best_index, best_side = min(
        candidates,
        key=lambda candidate: sum(
            float(
                candidate[1]["retargeting_features"].get(
                    f"vector_{finger}_curl", 0.0
                )
            )
            for finger in FINGERS
        ),
    )
    return best_index, [
        [float(value) for value in point]
        for point in best_side["hand_position_mm_used"]
    ]


def _dex_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dex_iterations=max(args.dex_iterations, 1),
        dex_damping=max(args.dex_damping, 1.0e-6),
        dex_max_joint_step=max(args.dex_max_joint_step, 1.0e-4),
        dex_normal_delta=max(args.dex_normal_delta, 0.0),
        dex_jacobian_eps=max(args.dex_jacobian_eps, 1.0e-6),
        dex_direction_weight=max(args.dex_direction_weight, 0.0),
        dex_vector_set=args.dex_vector_set,
        dex_curl_weight=max(args.dex_curl_weight, 0.0),
        dex_huber_delta=max(args.dex_huber_delta, 1.0e-6),
        dex_joint_limit_margin=max(args.dex_joint_limit_margin, 0.0),
        dex_joint_limit_weight=max(args.dex_joint_limit_weight, 0.0),
    )


def retarget_dex_vector(
    header: dict[str, Any] | None,
    all_frames: list[RecordingFrame],
    selected_frames: list[RecordingFrame],
    args: argparse.Namespace,
) -> tuple[list[RecordingFrame], dict[str, int]]:
    if not isinstance(header, dict):
        raise ValueError("dex_vector replay requires a recording header")
    calibration = header.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("dex_vector replay requires calibration data in the header")

    header_open = calibration.get("hand_position_open_mm")
    references: dict[str, list[list[float]]] = {}
    reference_indices: dict[str, int] = {}
    for side in ("left", "right"):
        points = header_open.get(side) if isinstance(header_open, dict) else None
        if isinstance(points, list) and len(points) == 20:
            references[side] = [
                [float(value) for value in point] for point in points
            ]
            reference_indices[side] = -1
        else:
            reference_indices[side], references[side] = _select_open_reference(
                all_frames, side
            )

    dex_args = _dex_args(args)
    urdf_path = args.urdf_path.expanduser().resolve()
    retargeters = {
        side: ESROBOHandDexVectorRetargeter(
            urdf_path,
            side,
            references[side],
            dex_args,
            {
                "open": calibration.get("open", {}).get(side, {}),
                "closed": calibration.get("closed", {}).get(side, {}),
            },
        )
        for side in ("left", "right")
    }
    imu_neutral = calibration.get("imu_neutral")
    recorded_axis_calibration = calibration.get("imu_axis_calibration")
    imu_axis_calibration = (
        recorded_axis_calibration
        if isinstance(recorded_axis_calibration, dict)
        else build_fixed_imu_axis_calibration()
    )
    has_axis_calibration = True
    axis_calibration_source = (
        "recorded" if isinstance(recorded_axis_calibration, dict) else "current_fixed"
    )
    processed: list[RecordingFrame] = []
    for packet_time, packet, raw in selected_frames:
        if not all(_valid_raw_side(raw.get(side)) for side in ("left", "right")):
            raise ValueError(
                f"frame at {packet_time:.6f}s lacks raw two-hand 20-point data"
            )

        targets: list[float] = []
        diagnostics: dict[str, Any] = {}
        skeleton_positions: dict[str, list[list[float]]] = {}
        absolute_orientations: dict[str, list[float]] = {}
        neutral_orientations: dict[str, list[float]] = {}
        orientation_deltas: dict[str, list[float]] = {}
        for side in ("left", "right"):
            raw_side = raw[side]
            points = raw_side["hand_position_mm_used"]
            side_targets, side_diagnostics = retargeters[side].retarget(points)
            targets.extend(side_targets)
            diagnostics[side] = side_diagnostics
            skeleton_positions[side] = canonicalize_hand_points_for_robot(points, side)

            current_imu = raw_side.get("imu_orientation_corrected_wxyz")
            neutral_imu = (
                imu_neutral.get(side) if isinstance(imu_neutral, dict) else None
            )
            delta_function = (
                orientation_delta_local_wxyz
                if has_axis_calibration
                else orientation_delta_wxyz
            )
            delta = delta_function(current_imu, neutral_imu)
            if delta is not None:
                if has_axis_calibration:
                    delta = apply_imu_axis_calibration(
                        delta, imu_axis_calibration.get(side), target_side=side
                    )
                absolute_orientations[side] = [float(value) for value in current_imu]
                neutral_orientations[side] = [float(value) for value in neutral_imu]
                orientation_deltas[side] = delta

        replay_packet = dict(packet)
        replay_packet["hand_joint_order"] = list(ESROBO_HAND_JOINT_ORDER)
        replay_packet["hand_joints"] = targets
        replay_packet["curls"] = diagnostics
        replay_packet["retargeting"] = {
            "method": "offline_raw_recording_dex_style_fk_vector_optimization",
            "mode": "dex_vector",
            "robot_left_source": "recorded_left_glove",
            "robot_right_source": "recorded_right_glove",
        }
        replay_packet["hand_skeleton_positions"] = skeleton_positions
        replay_packet["hand_skeleton_layout"] = (
            "finger_major_5x4_thumb_to_pinky_proximal_to_distal"
        )
        replay_packet["hand_skeleton_unit"] = "mm"
        replay_packet["hand_skeleton_frame"] = "robot_hand_local"
        replay_packet["hand_skeleton_source"] = (
            "recorded_senseglove_raw.hand_position_mm_used"
        )
        if len(orientation_deltas) == 2:
            replay_packet["hand_orientation_deltas"] = orientation_deltas
            replay_packet["hand_orientation_axis_calibrated"] = has_axis_calibration
            replay_packet["hand_orientations_absolute"] = absolute_orientations
            replay_packet["hand_orientation_neutral"] = neutral_orientations
            replay_packet["hand_orientation_delta_order"] = "wxyz"
            replay_packet["hand_orientation_retargeting"] = {
                "method": "recorded_corrected_imu_relative_to_recorded_neutral",
                "axis_calibration_source": axis_calibration_source,
            }
        processed.append((packet_time, replay_packet, raw))
    return processed, reference_indices


def send_loop(
    sock: socket.socket,
    frames: list[RecordingFrame],
    args: argparse.Namespace,
    loop_index: int,
    sequence: int,
) -> int:
    start = time.perf_counter()
    next_print = start + max(args.print_interval, 0.0)
    speed = max(args.speed, 1.0e-6)
    for local_index, (packet_time, packet, _raw) in enumerate(frames):
        delay = start + packet_time / speed - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        replay = dict(packet)
        replay["replay_sequence"] = sequence
        replay["replay_loop"] = loop_index
        replay["replay_stream"] = "hand"
        replay["replay_timestamp"] = time.time()
        replay_time_ns = time.monotonic_ns()
        replay["replay_sender_monotonic_ns"] = replay_time_ns
        if isinstance(replay.get("hand_orientations_absolute"), dict):
            replay["hand_orientation_sample_monotonic_ns"] = {
                side: replay_time_ns
                for side in replay["hand_orientations_absolute"]
            }
        sock.sendto(json.dumps(replay, separators=(",", ":")).encode("utf-8"), (args.host, args.port))
        sequence += 1
        now = time.perf_counter()
        if args.print_interval > 0.0 and now >= next_print:
            print(
                "[esrobo_hand_replay] "
                f"loop={loop_index} packet={local_index + 1}/{len(frames)} t={packet_time:.3f}s "
                f"hand={packet.get('synthetic_gesture', 'recorded')} "
                f"wrist={packet.get('synthetic_wrist_gesture', 'recorded')}",
                flush=True,
            )
            next_print = now + args.print_interval
    return sequence


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        header, all_frames = load_recording(args.hand_record_path.expanduser())
        frames = select_packets(all_frames, args.start_offset, args.duration)
        reference_indices: dict[str, int] = {}
        if args.retargeting_mode == "dex_vector":
            print(
                "[esrobo_hand_replay] Preparing dex_vector targets from recorded "
                f"raw hand skeletons ({len(frames)} frames); no live input is used...",
                flush=True,
            )
            frames, reference_indices = retarget_dex_vector(
                header, all_frames, frames, args
            )
    except Exception as exc:
        print(f"[esrobo_hand_replay] ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "[esrobo_hand_replay] "
        f"packets={len(frames)} duration={frames[-1][0]:.3f}s speed={args.speed:.3f} "
        f"destination={args.host}:{args.port} "
        f"retargeting={args.retargeting_mode} "
        f"imu_orientation={bool(header and header.get('imu_orientation_included'))}",
        flush=True,
    )
    if args.retargeting_mode == "dex_vector":
        print(
            "[esrobo_hand_replay] dex open references: "
            + " ".join(
                f"{side}={'header' if index < 0 else f'recording_frame_{index}'}"
                for side, index in reference_indices.items()
            ),
            flush=True,
        )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    loop_index = 0
    try:
        while True:
            sequence = send_loop(sock, frames, args, loop_index, sequence)
            loop_index += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[esrobo_hand_replay] stopped.", flush=True)
    finally:
        sock.close()
    print(f"[esrobo_hand_replay] done. sent={sequence}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
