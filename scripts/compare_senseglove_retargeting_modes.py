#!/usr/bin/env python3
"""Compare vector and dex-vector retargeting on one SenseGlove recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from senseglove_ros_to_esrobo_hand_bridge import (
    DEFAULT_ESROBO_URDF_PATH,
    ESROBOHandDexVectorRetargeter,
    FINGERS,
    retarget_side,
    senseglove_points_with_wrist_m,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_path", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
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
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_ESROBO_URDF_PATH)
    return parser.parse_args(argv)


def load_recording(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            item = json.loads(line)
            if item.get("type") == "esrobo_hand_recording_header":
                header = item
            elif item.get("type") == "esrobo_hand_packet":
                raw = item.get("senseglove_raw")
                if not isinstance(raw, dict):
                    continue
                if all(_valid_side(raw.get(side)) for side in ("left", "right")):
                    frames.append(item)
            elif line_number == 1:
                raise ValueError("first JSONL item is not a hand recording header")
    if header is None:
        raise ValueError("recording header is missing")
    if not frames:
        raise ValueError("recording has no frames with two valid 20-point hands")
    return header, frames


def _valid_side(raw_side: Any) -> bool:
    return (
        isinstance(raw_side, dict)
        and len(raw_side.get("hand_position_mm_used", [])) == 20
        and isinstance(raw_side.get("retargeting_features"), dict)
    )


def select_open_reference(frames: list[dict[str, Any]], side: str) -> tuple[int, list[list[float]]]:
    best_index = min(
        range(len(frames)),
        key=lambda index: sum(
            float(
                frames[index]["senseglove_raw"][side]["retargeting_features"].get(
                    f"vector_{finger}_curl", 0.0
                )
            )
            for finger in FINGERS
        ),
    )
    points = frames[best_index]["senseglove_raw"][side]["hand_position_mm_used"]
    return best_index, [[float(value) for value in point] for point in points]


def mode_args(args: argparse.Namespace, mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        retargeting_mode=mode,
        thumb_roll_gain=0.35,
        thumb_yaw_gain=0.50,
        spread_scale=1.0 if mode == "vector" else 0.0,
        vector_spread_range_deg=18.0,
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


def vector_errors(
    evaluator: ESROBOHandDexVectorRetargeter,
    qpos: list[float],
    hand_points: list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    human_points = evaluator._target_vectors(
        senseglove_points_with_wrist_m(hand_points, evaluator.side)
    )
    robot_vectors = evaluator._robot_vectors(np.asarray(qpos, dtype=np.float64))
    distance_errors = np.linalg.norm(robot_vectors - human_points, axis=1)
    robot_norms = np.linalg.norm(robot_vectors, axis=1)
    target_norms = np.linalg.norm(human_points, axis=1)
    denominators = np.maximum(robot_norms * target_norms, 1.0e-12)
    cosines = np.sum(robot_vectors * human_points, axis=1) / denominators
    angular_errors = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return distance_errors, angular_errors


def percentile(values: list[float], level: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def summarize_mode(
    qpos: dict[str, list[np.ndarray]],
    distances: list[float],
    angles: list[float],
    runtimes_ms: list[float],
    human_speeds: list[float],
    times: list[float],
    limits: dict[str, np.ndarray],
) -> dict[str, float]:
    joint_speeds: list[float] = []
    static_joint_speeds: list[float] = []
    lower_limit_count = 0
    upper_limit_count = 0
    value_count = 0
    for side in ("left", "right"):
        side_q = qpos[side]
        lower, upper = limits[side][:, 0], limits[side][:, 1]
        for values in side_q:
            span = np.maximum(upper - lower, 1.0e-9)
            lower_limit_count += int(np.count_nonzero((values - lower) / span < 0.01))
            upper_limit_count += int(np.count_nonzero((upper - values) / span < 0.01))
            value_count += values.size
        for index in range(1, len(side_q)):
            dt = max(times[index] - times[index - 1], 1.0e-6)
            speed = float(
                np.sqrt(
                    np.mean(((side_q[index] - side_q[index - 1]) / dt) ** 2)
                )
            )
            joint_speeds.append(speed)
            if human_speeds[index - 1] < 20.0:
                static_joint_speeds.append(speed)
    return {
        "mean_vector_error_mm": statistics.fmean(distances) * 1000.0,
        "p95_vector_error_mm": percentile(distances, 95.0) * 1000.0,
        "mean_angular_error_deg": statistics.fmean(angles),
        "p95_angular_error_deg": percentile(angles, 95.0),
        "mean_runtime_ms_per_two_hands": statistics.fmean(runtimes_ms),
        "p95_runtime_ms_per_two_hands": percentile(runtimes_ms, 95.0),
        "joint_speed_rms_mean_rad_s": statistics.fmean(joint_speeds),
        "static_joint_speed_mean_rad_s": (
            statistics.fmean(static_joint_speeds) if static_joint_speeds else 0.0
        ),
        "static_interval_count": float(len(static_joint_speeds) // 2),
        "joint_lower_limit_percent": 100.0 * lower_limit_count / max(value_count, 1),
        "joint_upper_limit_percent": 100.0 * upper_limit_count / max(value_count, 1),
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    header, all_frames = load_recording(args.record_path)
    stride = max(args.stride, 1)
    frames = all_frames[::stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    calibration = header.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("recording calibration is missing")

    open_references: dict[str, list[list[float]]] = {}
    reference_indices: dict[str, int] = {}
    header_open = calibration.get("hand_position_open_mm")
    for side in ("left", "right"):
        if isinstance(header_open, dict) and len(header_open.get(side, [])) == 20:
            open_references[side] = header_open[side]
            reference_indices[side] = -1
        else:
            index, points = select_open_reference(all_frames, side)
            reference_indices[side] = index
            open_references[side] = points

    vector_args = mode_args(args, "vector")
    dex_args = mode_args(args, "dex_vector")
    evaluators = {
        side: ESROBOHandDexVectorRetargeter(
            args.urdf_path.expanduser().resolve(),
            side,
            open_references[side],
            dex_args,
            {
                "open": calibration.get("open", {}).get(side, {}),
                "closed": calibration.get("closed", {}).get(side, {}),
            },
        )
        for side in ("left", "right")
    }
    modes = ("vector", "dex_vector")
    qpos = {mode: {side: [] for side in ("left", "right")} for mode in modes}
    distances = {mode: [] for mode in modes}
    angles = {mode: [] for mode in modes}
    runtimes = {mode: [] for mode in modes}
    times = [float(frame["t_rel_s"]) for frame in frames]
    human_speeds: list[float] = []
    previous_points: np.ndarray | None = None
    previous_time: float | None = None

    print(
        f"[retarget_compare] recording={args.record_path} valid_frames={len(all_frames)} "
        f"analyzed_frames={len(frames)} stride={stride}",
        flush=True,
    )
    print(
        "[retarget_compare] dex open references: "
        + " ".join(
            f"{side}={'header' if index < 0 else f'recording_frame_{index}'}"
            for side, index in reference_indices.items()
        ),
        flush=True,
    )

    for frame_index, frame in enumerate(frames):
        raw = frame["senseglove_raw"]
        stacked_points = np.asarray(
            raw["left"]["hand_position_mm_used"]
            + raw["right"]["hand_position_mm_used"],
            dtype=np.float64,
        )
        if previous_points is not None and previous_time is not None:
            dt = max(times[frame_index] - previous_time, 1.0e-6)
            human_speeds.append(
                float(np.sqrt(np.mean((stacked_points - previous_points) ** 2)) / dt)
            )
        previous_points = stacked_points
        previous_time = times[frame_index]

        start = time.perf_counter()
        vector_values: dict[str, list[float]] = {}
        for side in ("left", "right"):
            vector_values[side], _ = retarget_side(
                raw[side]["retargeting_features"],
                calibration["open"][side],
                calibration["closed"][side],
                side,
                vector_args,
            )
        runtimes["vector"].append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        dex_values = {
            side: evaluators[side].retarget(raw[side]["hand_position_mm_used"])[0]
            for side in ("left", "right")
        }
        runtimes["dex_vector"].append((time.perf_counter() - start) * 1000.0)

        for mode, values_by_side in (
            ("vector", vector_values),
            ("dex_vector", dex_values),
        ):
            for side in ("left", "right"):
                values = values_by_side[side]
                qpos[mode][side].append(np.asarray(values, dtype=np.float64))
                distance, angle = vector_errors(
                    evaluators[side], values, raw[side]["hand_position_mm_used"]
                )
                distances[mode].extend(distance.astype(float).tolist())
                angles[mode].extend(angle.astype(float).tolist())
        if (frame_index + 1) % 300 == 0:
            print(
                f"[retarget_compare] processed={frame_index + 1}/{len(frames)}",
                flush=True,
            )

    if len(frames) > 1:
        human_speeds.append(human_speeds[-1] if human_speeds else 0.0)
    else:
        human_speeds.append(0.0)
    limits = {side: evaluators[side].kinematics.joint_limits for side in ("left", "right")}
    summaries = {
        mode: summarize_mode(
            qpos[mode], distances[mode], angles[mode], runtimes[mode], human_speeds, times, limits
        )
        for mode in modes
    }
    print(
        "mode,mean_err_mm,p95_err_mm,mean_angle_deg,p95_angle_deg,runtime_ms,"
        "static_qspeed_rad_s,lower_limit_pct,upper_limit_pct"
    )
    for mode in modes:
        result = summaries[mode]
        print(
            f"{mode},{result['mean_vector_error_mm']:.3f},"
            f"{result['p95_vector_error_mm']:.3f},"
            f"{result['mean_angular_error_deg']:.3f},"
            f"{result['p95_angular_error_deg']:.3f},"
            f"{result['mean_runtime_ms_per_two_hands']:.3f},"
            f"{result['static_joint_speed_mean_rad_s']:.4f},"
            f"{result['joint_lower_limit_percent']:.2f},"
            f"{result['joint_upper_limit_percent']:.2f}"
        )
    print(
        "[retarget_compare] static intervals use hand-skeleton RMS speed <20 mm/s; "
        f"count={int(summaries['vector']['static_interval_count'])}."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"[retarget_compare] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
