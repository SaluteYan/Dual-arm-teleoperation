#!/usr/bin/env python3
"""Validate raw finger and IMU content in a recorded ESROBO hand JSONL file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_path", type=Path)
    return parser.parse_args(argv)


def _quat_angle_deg(left: list[float], right: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return math.degrees(2.0 * math.acos(min(max(dot, -1.0), 1.0)))


def _side_summary(items: list[dict[str, Any]], side: str) -> dict[str, Any]:
    snapshots = [
        item.get("senseglove_raw", {}).get(side, {})
        for item in items
        if isinstance(item.get("senseglove_raw", {}).get(side), dict)
    ]
    callbacks = {
        snapshot.get("callback_sequence")
        for snapshot in snapshots
        if snapshot.get("callback_sequence") is not None
    }
    skeleton_count = sum(
        len(snapshot.get("hand_position_mm_used", [])) == 20 for snapshot in snapshots
    )
    fingertip_count = sum(
        len(snapshot.get("finger_tip_position_mm_raw", [])) == 5
        for snapshot in snapshots
    )
    corrected_imus = [
        snapshot["imu_orientation_corrected_wxyz"]
        for snapshot in snapshots
        if isinstance(snapshot.get("imu_orientation_corrected_wxyz"), list)
        and len(snapshot["imu_orientation_corrected_wxyz"]) == 4
    ]
    raw_norms = [
        float(snapshot["imu_raw_norm"])
        for snapshot in snapshots
        if snapshot.get("imu_raw_norm") is not None
    ]
    return {
        "frames": len(snapshots),
        "unique_ros_callbacks": len(callbacks),
        "skeleton_20x3_frames": skeleton_count,
        "fingertip_5x3_frames": fingertip_count,
        "imu_frames": len(corrected_imus),
        "imu_w_repaired_frames": sum(
            bool(snapshot.get("imu_w_repaired")) for snapshot in snapshots
        ),
        "raw_imu_norm_median": statistics.median(raw_norms) if raw_norms else None,
        "imu_motion_deg": (
            max(_quat_angle_deg(corrected_imus[0], quat) for quat in corrected_imus)
            if corrected_imus
            else None
        ),
    }


def inspect(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    header: dict[str, Any] | None = None
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "esrobo_hand_recording_header":
                header = item
            elif item.get("type") == "esrobo_hand_packet":
                if not isinstance(item.get("packet"), dict):
                    raise ValueError(f"line {line_number} has no packet")
                items.append(item)
    if header is None:
        raise ValueError("missing esrobo_hand_recording_header")
    if not items:
        raise ValueError("recording contains no hand packets")

    times = [float(item["t_rel_s"]) for item in items]
    positive_periods = [b - a for a, b in zip(times, times[1:]) if b > a]
    targets = [item["packet"].get("hand_joints", []) for item in items]
    if any(len(values) != 20 for values in targets):
        raise ValueError("one or more packets do not contain 20 active hand joints")
    target_ranges = [
        max(values[index] for values in targets)
        - min(values[index] for values in targets)
        for index in range(20)
    ]
    summary = {
        "frames": len(items),
        "duration_s": times[-1] - times[0],
        "median_rate_hz": (
            1.0 / statistics.median(positive_periods) if positive_periods else 0.0
        ),
        "max_active_joint_motion_rad": max(target_ranges),
        "moving_active_joints": sum(value > 0.01 for value in target_ranges),
        "left": _side_summary(items, "left"),
        "right": _side_summary(items, "right"),
    }
    return header, items, summary


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        header, _items, summary = inspect(args.record_path.expanduser())
    except Exception as exc:
        print(f"[senseglove_recording_check] ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "[senseglove_recording_check] "
        f"source={header.get('source')} frames={summary['frames']} "
        f"duration={summary['duration_s']:.3f}s rate={summary['median_rate_hz']:.1f}Hz "
        f"moving_joints={summary['moving_active_joints']}/20 "
        f"max_joint_motion={summary['max_active_joint_motion_rad']:.3f}rad"
    )
    for side in ("left", "right"):
        data = summary[side]
        imu_motion = data["imu_motion_deg"]
        imu_motion_text = f"{imu_motion:.2f}deg" if imu_motion is not None else "none"
        print(
            f"[senseglove_recording_check] {side}: "
            f"callbacks={data['unique_ros_callbacks']} "
            f"skeleton={data['skeleton_20x3_frames']}/{data['frames']} "
            f"fingertips={data['fingertip_5x3_frames']}/{data['frames']} "
            f"imu={data['imu_frames']}/{data['frames']} "
            f"w_repaired={data['imu_w_repaired_frames']} "
            f"raw_norm_median={data['raw_imu_norm_median']} "
            f"imu_motion={imu_motion_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
