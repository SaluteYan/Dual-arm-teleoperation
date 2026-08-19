#!/usr/bin/env python3
"""Print Pico XR data from XRoboToolkit PC Service without IsaacLab."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _as_list(values) -> list:
    return list(values) if values is not None else []


def _format_pose(pose) -> str:
    arr = [float(value) for value in _as_list(pose)]
    if len(arr) < 7:
        return f"invalid len={len(arr)}"
    pos = arr[:3]
    quat_xyzw = arr[3:7]
    return (
        f"pos=[{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] "
        f"quat_xyzw=[{quat_xyzw[0]: .3f}, {quat_xyzw[1]: .3f}, {quat_xyzw[2]: .3f}, {quat_xyzw[3]: .3f}]"
    )


def _shape(values) -> tuple[int, ...]:
    rows = _as_list(values)
    if not rows:
        return (0,)
    first = rows[0]
    if isinstance(first, (list, tuple)):
        return (len(rows), len(first))
    return (len(rows),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Print rate.")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples to print. 0 means forever.")
    parser.add_argument("--hands", action="store_true", help="Also print hand-tracking activity and array shapes.")
    parser.add_argument("--body", action="store_true", help="Also print body-tracking availability and array shape.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import xrobotoolkit_sdk as xrt
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(f"Missing Python module: {missing}", file=sys.stderr)
        print("Install XRoboToolkit-PC-Service-Pybind first, then run this script again.", file=sys.stderr)
        return 2

    xrt.init()
    interval = 1.0 / max(args.rate_hz, 1.0)
    count = 0
    print("Reading XRoboToolkit data. Move controllers and press grip/trigger. Ctrl+C to stop.")
    try:
        while args.samples <= 0 or count < args.samples:
            left_pose = xrt.get_left_controller_pose()
            right_pose = xrt.get_right_controller_pose()
            left_grip = xrt.get_left_grip()
            right_grip = xrt.get_right_grip()
            left_trigger = xrt.get_left_trigger()
            right_trigger = xrt.get_right_trigger()

            print(f"\n[{count:05d}] timestamp_ns={xrt.get_time_stamp_ns()}")
            print(f"  left_controller:  {_format_pose(left_pose)}")
            print(f"  right_controller: {_format_pose(right_pose)}")
            print(
                "  inputs: "
                f"left_grip={left_grip:.3f} right_grip={right_grip:.3f} "
                f"left_trigger={left_trigger:.3f} right_trigger={right_trigger:.3f}"
            )

            if args.hands:
                hand_specs = (
                    ("left", xrt.get_left_hand_is_active(), xrt.get_left_hand_tracking_state),
                    ("right", xrt.get_right_hand_is_active(), xrt.get_right_hand_tracking_state),
                )
                for side, is_active, getter in hand_specs:
                    if not is_active:
                        print(f"  {side}_hand: inactive")
                        continue
                    hand = _as_list(getter())
                    wrist = hand[1][:3] if len(hand) > 1 and isinstance(hand[1], (list, tuple)) else None
                    print(f"  {side}_hand: shape={_shape(hand)} wrist_pos={wrist}")

            if args.body:
                if not xrt.is_body_data_available():
                    print("  body_tracking: unavailable")
                else:
                    poses = xrt.get_body_joints_pose()
                    print(f"  body_tracking: poses_shape={_shape(poses)}")

            count += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        xrt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
