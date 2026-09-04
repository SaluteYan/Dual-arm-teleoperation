#!/usr/bin/env python3
"""Synchronously replay aligned ESROBO body and hand JSONL recordings over UDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-record-path", required=True, type=Path)
    parser.add_argument("--hand-record-path", required=True, type=Path)
    parser.add_argument(
        "--host", default=os.environ.get("ESROBO_BODY_UDP_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050"))
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--body-start-offset", type=float, default=None)
    parser.add_argument("--hand-start-offset", type=float, default=None)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument(
        "--hand-imu-mode",
        choices=("relative", "synchronized", "disabled"),
        default="relative",
        help=(
            "Interpret a separately recorded glove IMU as wrist-relative motion, "
            "use synchronized for a hand stream captured with the same body motion, "
            "or disable hand orientation entirely."
        ),
    )
    return parser.parse_args(argv)


def _load_packets(
    path: Path, expected_packet_type: str
) -> tuple[dict[str, Any] | None, list[tuple[float, dict[str, Any]]]]:
    header: dict[str, Any] | None = None
    packets: list[tuple[float, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if str(item.get("type", "")).endswith("recording_header"):
                header = item
                continue
            packet = item.get("packet")
            if not isinstance(packet, dict) or "t_rel_s" not in item:
                continue
            if (
                expected_packet_type == "hand"
                and packet.get("type") != "esrobo_hand_joints"
            ):
                raise ValueError(
                    f"{path}: line {line_number} is not an esrobo_hand_joints packet"
                )
            packets.append((float(item["t_rel_s"]), packet))
    if not packets:
        raise ValueError(
            f"recording contains no {expected_packet_type} packets: {path}"
        )
    return header, packets


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select(
    body_packets: list[tuple[float, dict[str, Any]]],
    hand_packets: list[tuple[float, dict[str, Any]]],
    body_start_offset: float,
    hand_start_offset: float,
    duration: float,
) -> list[tuple[float, str, dict[str, Any]]]:
    def localize(
        packets: list[tuple[float, dict[str, Any]]], offset: float, stream: str
    ) -> list[tuple[float, str, dict[str, Any]]]:
        selected = [
            (packet_time, packet)
            for packet_time, packet in packets
            if packet_time >= offset
        ]
        if not selected:
            raise ValueError(
                f"no {stream} packets selected at start offset {offset:.3f}s"
            )
        base_time = selected[0][0]
        return [
            (packet_time - base_time, stream, packet)
            for packet_time, packet in selected
        ]

    body_events = localize(body_packets, max(body_start_offset, 0.0), "body")
    hand_events = localize(hand_packets, max(hand_start_offset, 0.0), "hand")
    shared_duration = min(body_events[-1][0], hand_events[-1][0])
    if duration > 0.0:
        shared_duration = min(shared_duration, duration)
    events = [
        event
        for event in body_events + hand_events
        if event[0] <= shared_duration + 1.0e-9
    ]
    stream_priority = {"body": 0, "hand": 1}
    events.sort(key=lambda event: (event[0], stream_priority[event[1]]))
    if not events:
        raise ValueError("no body/hand events selected by offsets and duration")
    return events


def _quat_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    result = [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]
    norm = math.sqrt(sum(value * value for value in result))
    return [value / norm for value in result] if norm > 1.0e-9 else [1.0, 0.0, 0.0, 0.0]


def _replay_packet(
    packet: dict[str, Any],
    sequence: int,
    loop_index: int,
    stream: str,
    hand_imu_neutral: dict[str, list[float]] | None,
    hand_imu_mode: str = "relative",
) -> dict[str, Any]:
    replay = dict(packet)
    replay["replay_sequence"] = sequence
    replay["replay_loop"] = loop_index
    replay["replay_stream"] = stream
    replay["replay_timestamp"] = time.time()
    replay_time_ns = time.monotonic_ns()
    replay["replay_sender_monotonic_ns"] = replay_time_ns
    if stream == "hand" and hand_imu_mode == "disabled":
        for key in (
            "hand_orientation_deltas",
            "hand_orientations_absolute",
            "hand_orientation_neutral",
            "hand_orientation_sample_monotonic_ns",
            "hand_orientation_axis_calibrated",
        ):
            replay.pop(key, None)
    elif stream == "hand":
        replay["hand_orientation_is_forearm_relative"] = (
            hand_imu_mode == "relative"
        )

    if stream == "hand" and hand_imu_mode != "disabled" and hand_imu_neutral:
        deltas = replay.get("hand_orientation_deltas")
        if isinstance(deltas, dict):
            absolute: dict[str, list[float]] = {}
            neutral: dict[str, list[float]] = {}
            sample_times: dict[str, int] = {}
            for side in ("left", "right"):
                delta = deltas.get(side)
                neutral_quat = hand_imu_neutral.get(side)
                if not isinstance(delta, list) or not isinstance(neutral_quat, list):
                    continue
                if len(delta) != 4 or len(neutral_quat) != 4:
                    continue
                absolute[side] = _quat_multiply_wxyz(delta, neutral_quat)
                neutral[side] = [float(value) for value in neutral_quat]
                sample_times[side] = replay_time_ns
            if absolute:
                replay["hand_orientations_absolute"] = absolute
                replay["hand_orientation_neutral"] = neutral
                replay["hand_orientation_sample_monotonic_ns"] = sample_times
    return replay


def _send_loop(
    sock: socket.socket,
    events: list[tuple[float, str, dict[str, Any]]],
    args: argparse.Namespace,
    loop_index: int,
    sequence: int,
    hand_imu_neutral: dict[str, list[float]] | None,
) -> int:
    start = time.perf_counter()
    next_print = start + max(args.print_interval, 0.0)
    speed = max(args.speed, 1.0e-6)
    stream_counts = {"body": 0, "hand": 0}
    for local_index, (packet_time, stream_name, packet) in enumerate(events):
        target = start + packet_time / speed
        delay = target - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        replay = _replay_packet(
            packet,
            sequence,
            loop_index,
            stream_name,
            hand_imu_neutral,
            args.hand_imu_mode,
        )
        sock.sendto(
            json.dumps(replay, separators=(",", ":")).encode("utf-8"),
            (args.host, args.port),
        )
        stream_counts[stream_name] += 1
        sequence += 1
        now = time.perf_counter()
        if args.print_interval > 0.0 and now >= next_print:
            print(
                "[esrobo_arm_hand_replay] "
                f"loop={loop_index} event={local_index + 1}/{len(events)} t={packet_time:.3f}s "
                f"body={stream_counts['body']} hand={stream_counts['hand']}",
                flush=True,
            )
            next_print = now + args.print_interval
    return sequence


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        body_header, body_packets = _load_packets(
            args.body_record_path.expanduser(), "body"
        )
        hand_header, hand_packets = _load_packets(
            args.hand_record_path.expanduser(), "hand"
        )
        hand_imu_neutral = None
        if isinstance(hand_header, dict):
            calibration = hand_header.get("calibration")
            if isinstance(calibration, dict) and isinstance(
                calibration.get("imu_neutral"), dict
            ):
                hand_imu_neutral = calibration["imu_neutral"]
        body_start_offset = (
            args.start_offset
            if args.body_start_offset is None
            else args.body_start_offset
        )
        hand_start_offset = (
            args.start_offset
            if args.hand_start_offset is None
            else args.hand_start_offset
        )
        events = _select(
            body_packets,
            hand_packets,
            body_start_offset,
            hand_start_offset,
            args.duration,
        )
        expected_hash = (
            hand_header.get("body_recording_sha256") if hand_header else None
        )
        if expected_hash and hand_header.get("aligned_packet_count") != len(
            body_packets
        ):
            raise ValueError(
                "hand recording header packet count does not match the body recording"
            )
        if (
            expected_hash
            and _file_sha256(args.body_record_path.expanduser()) != expected_hash
        ):
            raise ValueError(
                "hand recording was generated for a different body recording (SHA-256 mismatch)"
            )
    except Exception as exc:
        print(f"[esrobo_arm_hand_replay] ERROR: {exc}", file=sys.stderr)
        return 2

    duration_s = events[-1][0]
    body_event_count = sum(stream == "body" for _, stream, _ in events)
    hand_event_count = sum(stream == "hand" for _, stream, _ in events)
    print(
        "[esrobo_arm_hand_replay] "
        f"events={len(events)} body={body_event_count} hand={hand_event_count} "
        f"duration={duration_s:.3f}s speed={args.speed:.3f} "
        f"destination={args.host}:{args.port} hand_imu_mode={args.hand_imu_mode} "
        f"synthetic_hand={bool(hand_header and hand_header.get('synthetic'))}",
        flush=True,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    loop_index = 0
    try:
        while True:
            sequence = _send_loop(
                sock, events, args, loop_index, sequence, hand_imu_neutral
            )
            loop_index += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[esrobo_arm_hand_replay] stopped.", flush=True)
    finally:
        sock.close()
    print(f"[esrobo_arm_hand_replay] done. sent_events={sequence}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
