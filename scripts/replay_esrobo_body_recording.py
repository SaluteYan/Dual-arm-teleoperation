#!/usr/bin/env python3
"""Replay recorded ESROBO/PICO body-tracking packets to IsaacLab over UDP."""

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
    parser.add_argument("--record-path", required=True, help="JSONL recording produced by xrobotoolkit_body_udp_bridge.py.")
    parser.add_argument("--host", default=os.environ.get("ESROBO_BODY_UDP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050")))
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier. 1.0 preserves timing.")
    parser.add_argument("--loop", action="store_true", help="Loop the recording until interrupted.")
    parser.add_argument("--start-offset", type=float, default=0.0, help="Skip packets before this recording time in seconds.")
    parser.add_argument("--duration", type=float, default=0.0, help="Replay at most this many seconds. 0 means no limit.")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=0.0,
        help="Override recorded timing with a fixed send rate. 0 uses recorded timing.",
    )
    parser.add_argument("--print-interval", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument(
        "--preserve-packet",
        action="store_true",
        help="Send packet JSON exactly as recorded, without adding replay metadata.",
    )
    return parser.parse_args(argv)


def _packet_time_from_item(item: dict[str, Any], fallback_index: int) -> float:
    for key in ("t_rel_s", "record_t_rel_s"):
        value = item.get(key)
        if value is not None:
            return float(value)
    record_ns = item.get("record_monotonic_ns")
    if record_ns is not None:
        return float(record_ns) / 1.0e9
    packet = item.get("packet", item)
    if isinstance(packet, dict):
        sender_ns = packet.get("sender_monotonic_ns")
        if sender_ns is not None:
            return float(sender_ns) / 1.0e9
        wall_time = packet.get("timestamp")
        if wall_time is not None:
            return float(wall_time)
    return float(fallback_index)


def _load_recording(path: Path) -> tuple[dict[str, Any] | None, list[tuple[float, dict[str, Any]]]]:
    header: dict[str, Any] | None = None
    packets: list[tuple[float, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_index}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "esrobo_body_recording_header":
                header = item
                continue
            packet = item.get("packet")
            if packet is None and ("frames" in item or "joint_names" in item):
                packet = item
            if not isinstance(packet, dict):
                continue
            packets.append((_packet_time_from_item(item, len(packets)), packet))

    if not packets:
        raise ValueError(f"recording contains no body packets: {path}")

    first_time = packets[0][0]
    normalized_packets = [(max(0.0, packet_time - first_time), packet) for packet_time, packet in packets]
    return header, normalized_packets


def _packet_frame_summary(packet: dict[str, Any]) -> tuple[int, str]:
    frames = packet.get("frames", {})
    if not isinstance(frames, dict):
        return 0, "none"
    return len(frames), ",".join(sorted(frames.keys())) or "none"


def _timed_packets(
    packets: list[tuple[float, dict[str, Any]]],
    start_offset: float,
    duration: float,
    rate_hz: float,
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
    if rate_hz > 0.0:
        step_s = 1.0 / rate_hz
        return [(index * step_s, packet) for index, (_packet_time, packet) in enumerate(selected)]
    return [(packet_time - base_time, packet) for packet_time, packet in selected]


def _packet_for_replay(packet: dict[str, Any], sequence: int, loop_index: int, preserve: bool) -> dict[str, Any]:
    if preserve:
        return packet
    replay_packet = dict(packet)
    replay_packet["replay_sequence"] = sequence
    replay_packet["replay_loop"] = loop_index
    replay_packet["replay_timestamp"] = time.time()
    replay_packet["replay_sender_monotonic_ns"] = time.monotonic_ns()
    return replay_packet


def _send_sequence(
    sock: socket.socket,
    packets: list[tuple[float, dict[str, Any]]],
    host: str,
    port: int,
    speed: float,
    loop_index: int,
    preserve_packet: bool,
    print_interval: float,
    sent_total: int,
) -> int:
    start_time = time.perf_counter()
    last_print = start_time
    speed = max(speed, 1.0e-6)
    for local_index, (packet_time, packet) in enumerate(packets):
        target_time = start_time + packet_time / speed
        sleep_s = target_time - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)

        replay_packet = _packet_for_replay(packet, sent_total, loop_index, preserve_packet)
        encoded = json.dumps(replay_packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(encoded, (host, port))
        sent_total += 1

        now = time.perf_counter()
        if now - last_print >= print_interval:
            frame_count, frame_names = _packet_frame_summary(packet)
            print(
                "[esrobo_body_replay] "
                f"loop={loop_index} sent={sent_total} local={local_index + 1}/{len(packets)} "
                f"t={packet_time:.3f}s frames={frame_count} [{frame_names}] bytes={len(encoded)}",
                flush=True,
            )
            last_print = now
    return sent_total


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    path = Path(args.record_path).expanduser()
    if not path.exists():
        print(f"[esrobo_body_replay] ERROR: recording does not exist: {path}", file=sys.stderr)
        return 2

    try:
        header, loaded_packets = _load_recording(path)
        packets = _timed_packets(loaded_packets, args.start_offset, args.duration, args.rate_hz)
    except Exception as exc:
        print(f"[esrobo_body_replay] ERROR: {exc}", file=sys.stderr)
        return 2

    duration_s = packets[-1][0] if packets else 0.0
    print(f"[esrobo_body_replay] Loaded {len(packets)} packets from {path}.", flush=True)
    if header is not None:
        print(
            "[esrobo_body_replay] Recording "
            f"version={header.get('version', 'unknown')} source={header.get('source', 'unknown')} "
            f"created={header.get('created_time_iso', 'unknown')}",
            flush=True,
        )
    print(
        f"[esrobo_body_replay] Sending to {args.host}:{args.port}; "
        f"duration={duration_s:.3f}s speed={args.speed:.3f} loop={args.loop}.",
        flush=True,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent_total = 0
    loop_index = 0
    try:
        while True:
            sent_total = _send_sequence(
                sock=sock,
                packets=packets,
                host=args.host,
                port=args.port,
                speed=args.speed,
                loop_index=loop_index,
                preserve_packet=args.preserve_packet,
                print_interval=max(args.print_interval, 0.1),
                sent_total=sent_total,
            )
            loop_index += 1
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[esrobo_body_replay] stopped.", flush=True)
    finally:
        sock.close()
    print(f"[esrobo_body_replay] done. sent={sent_total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
