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
from typing import Any


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
    return parser.parse_args(argv)


def load_recording(path: Path) -> tuple[dict[str, Any] | None, list[tuple[float, dict[str, Any]]]]:
    header: dict[str, Any] | None = None
    packets: list[tuple[float, dict[str, Any]]] = []
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
            packets.append((float(item["t_rel_s"]), packet))
    if not packets:
        raise ValueError(f"recording contains no hand packets: {path}")
    return header, packets


def select_packets(
    packets: list[tuple[float, dict[str, Any]]], start_offset: float, duration: float
) -> list[tuple[float, dict[str, Any]]]:
    end_time = start_offset + duration if duration > 0.0 else None
    selected = [
        (packet_time, packet)
        for packet_time, packet in packets
        if packet_time >= start_offset and (end_time is None or packet_time <= end_time)
    ]
    if not selected:
        raise ValueError("no packets selected by --start-offset/--duration")
    base_time = selected[0][0]
    return [(packet_time - base_time, packet) for packet_time, packet in selected]


def send_loop(
    sock: socket.socket,
    packets: list[tuple[float, dict[str, Any]]],
    args: argparse.Namespace,
    loop_index: int,
    sequence: int,
) -> int:
    start = time.perf_counter()
    next_print = start + max(args.print_interval, 0.0)
    speed = max(args.speed, 1.0e-6)
    for local_index, (packet_time, packet) in enumerate(packets):
        delay = start + packet_time / speed - time.perf_counter()
        if delay > 0.0:
            time.sleep(delay)
        replay = dict(packet)
        replay["replay_sequence"] = sequence
        replay["replay_loop"] = loop_index
        replay["replay_stream"] = "hand"
        replay["replay_timestamp"] = time.time()
        replay["replay_sender_monotonic_ns"] = time.monotonic_ns()
        sock.sendto(json.dumps(replay, separators=(",", ":")).encode("utf-8"), (args.host, args.port))
        sequence += 1
        now = time.perf_counter()
        if args.print_interval > 0.0 and now >= next_print:
            print(
                "[esrobo_hand_replay] "
                f"loop={loop_index} packet={local_index + 1}/{len(packets)} t={packet_time:.3f}s "
                f"hand={packet.get('synthetic_gesture', 'recorded')} "
                f"wrist={packet.get('synthetic_wrist_gesture', 'recorded')}",
                flush=True,
            )
            next_print = now + args.print_interval
    return sequence


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        header, packets = load_recording(args.hand_record_path.expanduser())
        packets = select_packets(packets, args.start_offset, args.duration)
    except Exception as exc:
        print(f"[esrobo_hand_replay] ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "[esrobo_hand_replay] "
        f"packets={len(packets)} duration={packets[-1][0]:.3f}s speed={args.speed:.3f} "
        f"destination={args.host}:{args.port} "
        f"imu_orientation={bool(header and header.get('imu_orientation_included'))}",
        flush=True,
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    loop_index = 0
    try:
        while True:
            sequence = send_loop(sock, packets, args, loop_index, sequence)
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
