#!/usr/bin/env python3
"""Stream XRoboToolkit PICO hand tracking to IsaacLab over UDP."""

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
    parser.add_argument("--port", type=int, default=15100, help="IsaacLab UDP port.")
    parser.add_argument("--rate-hz", type=float, default=120.0, help="Polling/sending rate limit.")
    parser.add_argument("--print-interval", type=float, default=1.0, help="Status print interval in seconds.")
    parser.add_argument("--only-on-change", action="store_true", help="Only send packets when XRoboToolkit timestamp changes.")
    parser.add_argument("--samples", type=int, default=0, help="Number of packets to send. 0 means forever.")
    return parser.parse_args(argv)


def _as_pose_rows(raw_state: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw_row in raw_state or []:
        row = [float(value) for value in raw_row]
        if len(row) >= 7 and all(math.isfinite(value) for value in row[:7]):
            rows.append(row[:7])
    return rows


def _hand_payload(active: bool, raw_state: Any) -> dict[str, Any]:
    joints = _as_pose_rows(raw_state) if active else []
    return {
        "active": bool(active and joints),
        "joint_count": len(joints),
        "joints": joints,
        "pose_order": "xyzw",
    }


def _rate(values: deque[float], now: float, window_s: float = 1.0) -> float:
    while values and now - values[0] > window_s:
        values.popleft()
    if len(values) < 2:
        return 0.0
    return (len(values) - 1) / max(values[-1] - values[0], 1.0e-9)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    rate_hz = max(args.rate_hz, 1.0)
    step_s = 1.0 / rate_hz

    try:
        import xrobotoolkit_sdk as xrt
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(f"[xrobotoolkit_hand_bridge] Missing Python module: {missing}", file=sys.stderr)
        print("[xrobotoolkit_hand_bridge] Activate env_xrobotoolkit and install xrobotoolkit_sdk first.", file=sys.stderr)
        return 2

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[xrobotoolkit_hand_bridge] Sending hand tracking UDP to {args.host}:{args.port}.")
    print("[xrobotoolkit_hand_bridge] Keep XRoboToolkit-PC-Service and the PICO app connected.")

    xrt.init()
    sent_times: deque[float] = deque()
    source_change_times: deque[float] = deque()
    last_source_timestamp_ns: int | None = None
    last_print = time.monotonic()
    sent_total = 0
    latest_packet_size = 0

    try:
        while args.samples <= 0 or sent_total < args.samples:
            loop_start = time.perf_counter()
            now = time.monotonic()
            source_timestamp_ns = int(xrt.get_time_stamp_ns())

            left_active = bool(xrt.get_left_hand_is_active())
            right_active = bool(xrt.get_right_hand_is_active())
            left_state = xrt.get_left_hand_tracking_state() if left_active else []
            right_state = xrt.get_right_hand_tracking_state() if right_active else []

            source_changed = source_timestamp_ns != last_source_timestamp_ns
            if source_changed:
                source_change_times.append(now)
                last_source_timestamp_ns = source_timestamp_ns

            if source_changed or not args.only_on_change:
                packet = {
                    "type": "xrobotoolkit_hand_tracking",
                    "sequence": sent_total,
                    "timestamp": time.time(),
                    "sender_monotonic_ns": time.monotonic_ns(),
                    "xrt_timestamp_ns": source_timestamp_ns,
                    "hands": {
                        "left": _hand_payload(left_active, left_state),
                        "right": _hand_payload(right_active, right_state),
                    },
                }
                encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                sock.sendto(encoded, (args.host, args.port))
                latest_packet_size = len(encoded)
                sent_total += 1
                sent_times.append(now)

            if now - last_print >= args.print_interval:
                print(
                    "[xrobotoolkit_hand_bridge] "
                    f"send_hz={_rate(sent_times, now):5.1f} "
                    f"xrt_update_hz={_rate(source_change_times, now):5.1f} "
                    f"left={'on' if left_active else 'off'}({len(_as_pose_rows(left_state)):02d}) "
                    f"right={'on' if right_active else 'off'}({len(_as_pose_rows(right_state)):02d}) "
                    f"timestamp_ns={source_timestamp_ns} bytes={latest_packet_size}"
                )
                last_print = now

            elapsed = time.perf_counter() - loop_start
            if elapsed < step_s:
                time.sleep(step_s - elapsed)
    except KeyboardInterrupt:
        print("\n[xrobotoolkit_hand_bridge] stopped.")
    finally:
        xrt.close()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
