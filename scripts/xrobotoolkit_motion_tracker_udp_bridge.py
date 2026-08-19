#!/usr/bin/env python3
"""Stream XRoboToolkit motion tracker poses to IsaacLab over UDP."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from collections import deque
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="IsaacLab UDP host.")
    parser.add_argument("--port", type=int, default=15200, help="IsaacLab UDP port.")
    parser.add_argument("--rate-hz", type=float, default=120.0, help="Polling/sending rate limit.")
    parser.add_argument("--print-interval", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument("--left-serial", default="", help="Serial number mounted on the left wrist.")
    parser.add_argument("--right-serial", default="", help="Serial number mounted on the right wrist.")
    parser.add_argument(
        "--emit-format",
        choices=("tracker", "bodytracking"),
        default="tracker",
        help="UDP payload format. 'bodytracking' feeds IsaacLab bodytracking_udp teleop.",
    )
    parser.add_argument(
        "--auto-map-by-y",
        action="store_true",
        help="When serials are not provided and exactly two trackers are online, map larger Y to left wrist.",
    )
    parser.add_argument("--only-on-change", action="store_true", help="Only send packets when tracker data changes.")
    parser.add_argument("--samples", type=int, default=0, help="Number of packets to send. 0 means forever.")
    parser.add_argument(
        "--print-poses",
        action="store_true",
        help="Include mapped wrist position/quaternion values in the periodic status output.",
    )
    return parser.parse_args(argv)


def _as_float_list(values: Any, length: int) -> list[float]:
    row = [float(value) for value in values or []]
    if len(row) < length:
        return []
    row = row[:length]
    if not all(math.isfinite(value) for value in row):
        return []
    return row


def _rate(values: deque[float], now: float, window_s: float = 1.0) -> float:
    while values and now - values[0] > window_s:
        values.popleft()
    if len(values) < 2:
        return 0.0
    return (len(values) - 1) / max(values[-1] - values[0], 1.0e-9)


def _tracker_signature(trackers: list[dict[str, Any]], timestamp_ns: int) -> tuple:
    pose_sig = tuple(
        (tracker["serial"], tuple(round(value, 5) for value in tracker["pose"]))
        for tracker in trackers
    )
    return timestamp_ns, pose_sig


def _resolve_wrist_trackers(
    trackers: list[dict[str, Any]],
    left_serial: str,
    right_serial: str,
    auto_map_by_y: bool,
) -> dict[str, dict[str, Any] | None]:
    by_serial = {tracker["serial"]: tracker for tracker in trackers}
    left_tracker = by_serial.get(left_serial) if left_serial else None
    right_tracker = by_serial.get(right_serial) if right_serial else None

    if (left_tracker is None or right_tracker is None) and auto_map_by_y and len(trackers) == 2:
        ordered = sorted(trackers, key=lambda tracker: tracker["pose"][1])
        right_tracker = right_tracker or ordered[0]
        left_tracker = left_tracker or ordered[1]

    return {"left": left_tracker, "right": right_tracker}


def _build_tracker_packet(
    trackers: list[dict[str, Any]],
    timestamp_ns: int,
    sequence: int,
    left_serial: str,
    right_serial: str,
) -> dict[str, Any]:
    return {
        "type": "xrobotoolkit_motion_tracker",
        "sequence": sequence,
        "timestamp": time.time(),
        "sender_monotonic_ns": time.monotonic_ns(),
        "xrt_motion_timestamp_ns": timestamp_ns,
        "trackers": trackers,
        "wrist_map": {
            "left": left_serial or None,
            "right": right_serial or None,
        },
    }


def _build_bodytracking_packet(
    trackers: list[dict[str, Any]],
    timestamp_ns: int,
    sequence: int,
    left_serial: str,
    right_serial: str,
    auto_map_by_y: bool,
) -> dict[str, Any]:
    wrists = _resolve_wrist_trackers(trackers, left_serial, right_serial, auto_map_by_y)
    frames: dict[str, Any] = {}
    for side, tracker in wrists.items():
        if tracker is None:
            continue
        pose = tracker["pose"]
        frames[f"{side}_wrist"] = {
            "serial": tracker["serial"],
            "pos": pose[:3],
            "quat_xyzw": pose[3:7],
            "valid": True,
        }

    return {
        "timestamp": time.time(),
        "sender_monotonic_ns": time.monotonic_ns(),
        "source": "xrobotoolkit_motion_tracker",
        "source_frame": "pico_tracking",
        "xrt_motion_timestamp_ns": timestamp_ns,
        "sequence": sequence,
        "quat_order": "xyzw",
        "frames": frames,
        "wrist_map": {
            "left": wrists["left"]["serial"] if wrists["left"] else None,
            "right": wrists["right"]["serial"] if wrists["right"] else None,
        },
    }


def _build_trackers(xrt) -> tuple[list[dict[str, Any]], int]:
    count = int(xrt.num_motion_data_available())
    if count <= 0:
        return [], int(xrt.get_motion_timestamp_ns())

    poses = xrt.get_motion_tracker_pose()
    velocities = xrt.get_motion_tracker_velocity()
    accelerations = xrt.get_motion_tracker_acceleration()
    serials = xrt.get_motion_tracker_serial_numbers()
    timestamp_ns = int(xrt.get_motion_timestamp_ns())

    trackers: list[dict[str, Any]] = []
    for index in range(min(count, len(poses), len(serials))):
        pose = _as_float_list(poses[index], 7)
        if not pose:
            continue
        serial = str(serials[index] or f"tracker_{index}")
        trackers.append(
            {
                "index": index,
                "serial": serial,
                "pose": pose,
                "velocity": _as_float_list(velocities[index], 6) if index < len(velocities) else [],
                "acceleration": _as_float_list(accelerations[index], 6) if index < len(accelerations) else [],
                "pose_order": "xyzw",
            }
        )
    return trackers, timestamp_ns


def _format_mapped_pose_summary(packet: dict[str, Any]) -> str:
    frames = packet.get("frames")
    if not isinstance(frames, dict):
        return "none"

    chunks: list[str] = []
    for frame_name in ("left_wrist", "right_wrist"):
        frame = frames.get(frame_name)
        if not isinstance(frame, dict):
            continue
        pos = frame.get("pos", [])
        quat = frame.get("quat_xyzw", [])
        serial = frame.get("serial", "unknown")
        try:
            pos_text = ",".join(f"{float(value):+.3f}" for value in pos[:3])
            quat_text = ",".join(f"{float(value):+.3f}" for value in quat[:4])
        except (TypeError, ValueError):
            continue
        chunks.append(f"{frame_name}:{serial}:pos=[{pos_text}] quat_xyzw=[{quat_text}]")

    return " | ".join(chunks) if chunks else "none"


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rate_hz = max(args.rate_hz, 1.0)
    step_s = 1.0 / rate_hz

    try:
        import xrobotoolkit_sdk as xrt
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(f"[xrobotoolkit_tracker_bridge] Missing Python module: {missing}", file=sys.stderr)
        print("[xrobotoolkit_tracker_bridge] Activate env_xrobotoolkit and install xrobotoolkit_sdk first.", file=sys.stderr)
        return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[xrobotoolkit_tracker_bridge] Sending motion tracker UDP to {args.host}:{args.port}.")
    print("[xrobotoolkit_tracker_bridge] Keep XRoboToolkit-PC-Service and the PICO app connected.")
    if args.left_serial or args.right_serial or args.auto_map_by_y:
        print(
            "[xrobotoolkit_tracker_bridge] Wrist mapping: "
            f"left={args.left_serial or 'auto'} right={args.right_serial or 'auto'} "
            f"auto_map_by_y={args.auto_map_by_y}"
        )
    print(f"[xrobotoolkit_tracker_bridge] Emit format: {args.emit_format}")

    xrt.init()
    sent_times: deque[float] = deque()
    source_change_times: deque[float] = deque()
    last_signature: tuple | None = None
    last_print = time.monotonic()
    sent_total = 0
    latest_packet_size = 0
    latest_serials: list[str] = []
    latest_mapped_frames: list[str] = []
    latest_wrist_map: dict[str, Any] = {"left": None, "right": None}
    latest_pose_summary = "none"

    try:
        while args.samples <= 0 or sent_total < args.samples:
            loop_start = time.perf_counter()
            now = time.monotonic()
            trackers, timestamp_ns = _build_trackers(xrt)
            signature = _tracker_signature(trackers, timestamp_ns)
            source_changed = signature != last_signature
            if source_changed:
                source_change_times.append(now)
                last_signature = signature

            if source_changed or not args.only_on_change:
                if args.emit_format == "bodytracking":
                    packet = _build_bodytracking_packet(
                        trackers,
                        timestamp_ns,
                        sent_total,
                        args.left_serial,
                        args.right_serial,
                        args.auto_map_by_y,
                    )
                    latest_mapped_frames = sorted(packet.get("frames", {}).keys())
                    latest_wrist_map = packet.get("wrist_map", {"left": None, "right": None})
                    latest_pose_summary = _format_mapped_pose_summary(packet)
                else:
                    packet = _build_tracker_packet(
                        trackers,
                        timestamp_ns,
                        sent_total,
                        args.left_serial,
                        args.right_serial,
                    )
                    latest_mapped_frames = []
                    latest_wrist_map = packet.get("wrist_map", {"left": None, "right": None})
                    latest_pose_summary = "none"
                encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                sock.sendto(encoded, (args.host, args.port))
                latest_packet_size = len(encoded)
                sent_total += 1
                sent_times.append(now)
                latest_serials = [tracker["serial"] for tracker in trackers]

            if now - last_print >= args.print_interval:
                serial_text = ",".join(latest_serials) if latest_serials else "none"
                mapped_text = ",".join(latest_mapped_frames) if latest_mapped_frames else "none"
                wrist_map_text = (
                    f"left={latest_wrist_map.get('left') or 'none'} "
                    f"right={latest_wrist_map.get('right') or 'none'}"
                )
                missing_requested: list[str] = []
                if args.emit_format == "bodytracking":
                    if args.left_serial and args.left_serial not in latest_serials:
                        missing_requested.append(f"left:{args.left_serial}")
                    if args.right_serial and args.right_serial not in latest_serials:
                        missing_requested.append(f"right:{args.right_serial}")
                missing_text = ",".join(missing_requested) if missing_requested else "none"
                print(
                    "[xrobotoolkit_tracker_bridge] "
                    f"send_hz={_rate(sent_times, now):5.1f} "
                    f"xrt_update_hz={_rate(source_change_times, now):5.1f} "
                    f"trackers={len(latest_serials)} serials=[{serial_text}] "
                    f"mapped=[{mapped_text}] wrist_map=[{wrist_map_text}] "
                    f"missing_requested=[{missing_text}] "
                    f"timestamp_ns={timestamp_ns} bytes={latest_packet_size}"
                )
                if args.print_poses:
                    print(f"[xrobotoolkit_tracker_bridge] mapped_poses {latest_pose_summary}")
                last_print = now

            elapsed = time.perf_counter() - loop_start
            if elapsed < step_s:
                time.sleep(step_s - elapsed)
    except KeyboardInterrupt:
        print("\n[xrobotoolkit_tracker_bridge] stopped.")
    finally:
        xrt.close()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
