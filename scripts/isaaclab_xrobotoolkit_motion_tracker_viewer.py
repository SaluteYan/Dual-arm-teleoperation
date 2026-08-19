#!/usr/bin/env python3
"""Visualize XRoboToolkit motion tracker wrist poses in IsaacLab."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="0.0.0.0", help="UDP bind host.")
parser.add_argument("--port", type=int, default=15200, help="UDP bind port.")
parser.add_argument("--max-packet-bytes", type=int, default=65535, help="UDP receive buffer size.")
parser.add_argument("--max-stale-time", type=float, default=0.5, help="Hide trackers after this many seconds without packets.")
parser.add_argument("--scale", type=float, default=1.0, help="Display scale for tracker positions.")
parser.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 1.1), metavar=("X", "Y", "Z"))
parser.add_argument("--left-serial", default="", help="Serial number mounted on the left wrist.")
parser.add_argument("--right-serial", default="", help="Serial number mounted on the right wrist.")
parser.add_argument(
    "--origin-mode",
    choices=("calibrated", "raw"),
    default="calibrated",
    help="Subtract first observed tracker center before display, or show raw transformed positions.",
)
parser.add_argument(
    "--source-to-world-rotation",
    type=float,
    nargs=9,
    default=(0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    help="Row-major 3x3 rotation applied to PICO poses before display.",
)
parser.add_argument("--print-interval", type=float, default=1.0, help="Metrics print interval in seconds.")
parser.add_argument("--csv", default="", help="Optional CSV path for interval metrics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


HIDDEN_POSITION = np.asarray([0.0, 0.0, -100.0], dtype=np.float32)


class UdpTrackerReceiver:
    """Non-blocking receiver that keeps only the latest motion tracker packet."""

    def __init__(self, host: str, port: int, max_packet_bytes: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((host, port))
        self.max_packet_bytes = max_packet_bytes
        self.latest: dict[str, Any] | None = None
        self.last_receive_time: float | None = None
        self._last_sequence: int | None = None
        self._last_source_signature: tuple | None = None
        self.interval_packets = 0
        self.interval_source_changes = 0
        self.interval_dropped_sequence = 0
        self.interval_latency_ms: list[float] = []

    def close(self) -> None:
        self.socket.close()

    def receive_latest(self) -> bool:
        received_any = False
        while True:
            try:
                payload, _sender = self.socket.recvfrom(self.max_packet_bytes)
            except BlockingIOError:
                break
            message = json.loads(payload.decode("utf-8"))
            now_ns = time.monotonic_ns()

            sequence = message.get("sequence")
            if isinstance(sequence, int) and self._last_sequence is not None and sequence > self._last_sequence + 1:
                self.interval_dropped_sequence += sequence - self._last_sequence - 1
            if isinstance(sequence, int):
                self._last_sequence = sequence

            source_signature = _source_signature(message)
            if source_signature != self._last_source_signature:
                self.interval_source_changes += 1
                self._last_source_signature = source_signature

            sender_monotonic_ns = message.get("sender_monotonic_ns")
            if isinstance(sender_monotonic_ns, int):
                self.interval_latency_ms.append((now_ns - sender_monotonic_ns) / 1.0e6)

            self.latest = message
            self.last_receive_time = time.monotonic()
            self.interval_packets += 1
            received_any = True
        return received_any

    def is_fresh(self, max_stale_time: float) -> bool:
        if self.latest is None or self.last_receive_time is None:
            return False
        return time.monotonic() - self.last_receive_time <= max_stale_time

    def reset_interval(self) -> None:
        self.interval_packets = 0
        self.interval_source_changes = 0
        self.interval_dropped_sequence = 0
        self.interval_latency_ms.clear()


def _source_signature(message: dict[str, Any]) -> tuple:
    trackers = message.get("trackers", [])
    poses = tuple(
        (str(tracker.get("serial", "")), tuple(round(float(value), 5) for value in tracker.get("pose", [])[:7]))
        for tracker in trackers
        if isinstance(tracker, dict)
    )
    return int(message.get("xrt_motion_timestamp_ns", 0)), poses


def _define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/XRoboToolkitMotionTrackers",
        markers={
            "left_wrist": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.7, 1.0), roughness=0.65),
            ),
            "right_wrist": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.06), roughness=0.65),
            ),
            "tracker": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.95, 1.0), roughness=0.8),
            ),
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.16, 0.16, 0.16),
            ),
            "hidden": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat_xyzw(quat_xyzw)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def _transform_tracker_pose(pose_xyzw: list[float], rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_xyzw, dtype=np.float32)
    position = rotation @ pose[:3]
    orientation_matrix = rotation @ _quat_xyzw_to_matrix(pose[3:7]) @ rotation.T
    return position, _matrix_to_quat_wxyz(orientation_matrix)


def _valid_trackers(message: dict[str, Any], rotation: np.ndarray) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for tracker in message.get("trackers", []):
        if not isinstance(tracker, dict):
            continue
        pose = tracker.get("pose", [])
        if len(pose) < 7:
            continue
        try:
            position, quat_wxyz = _transform_tracker_pose([float(v) for v in pose[:7]], rotation)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quat_wxyz)):
            continue
        serial = str(tracker.get("serial", f"tracker_{tracker.get('index', len(result))}"))
        result[serial] = {
            "serial": serial,
            "position": position,
            "quat_wxyz": quat_wxyz,
            "raw": tracker,
        }
    return result


def _resolve_wrist_map(
    trackers: dict[str, dict[str, Any]],
    packet_map: dict[str, Any],
    left_serial: str,
    right_serial: str,
) -> dict[str, str | None]:
    left = left_serial or packet_map.get("left")
    right = right_serial or packet_map.get("right")
    if left in trackers or right in trackers:
        return {"left": left if left in trackers else None, "right": right if right in trackers else None}
    if len(trackers) == 2:
        ordered = sorted(trackers.values(), key=lambda item: item["position"][1])
        return {"right": ordered[0]["serial"], "left": ordered[1]["serial"]}
    return {"left": None, "right": None}


def _display_positions(
    trackers: dict[str, dict[str, Any]],
    origin_mode: str,
    origin: np.ndarray | None,
    offset: np.ndarray,
    scale: float,
) -> tuple[dict[str, dict[str, Any]], np.ndarray | None]:
    if not trackers:
        return {}, origin
    if origin_mode == "calibrated" and origin is None:
        origin = np.mean(np.stack([item["position"] for item in trackers.values()]), axis=0)

    displayed: dict[str, dict[str, Any]] = {}
    for serial, item in trackers.items():
        base = origin if origin_mode == "calibrated" and origin is not None else np.zeros(3, dtype=np.float32)
        displayed[serial] = {
            **item,
            "display_position": offset + scale * (item["position"] - base),
        }
    return displayed, origin


def _visualize(
    visualizer: VisualizationMarkers,
    trackers: dict[str, dict[str, Any]],
    wrist_map: dict[str, str | None],
    device: str,
) -> None:
    translations: list[torch.Tensor] = []
    orientations: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    marker_indices: list[int] = []
    unit_scale = torch.ones(3, device=device)

    for serial, item in trackers.items():
        position = torch.tensor(item["display_position"], device=device, dtype=torch.float32)
        quat = torch.tensor(item["quat_wxyz"], device=device, dtype=torch.float32)
        role = "tracker"
        if serial == wrist_map.get("left"):
            role = "left"
        elif serial == wrist_map.get("right"):
            role = "right"
        marker_index = {"left": 0, "right": 1, "tracker": 2}[role]
        translations.append(position)
        orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        scales.append(unit_scale)
        marker_indices.append(marker_index)
        translations.append(position)
        orientations.append(quat)
        scales.append(unit_scale)
        marker_indices.append(3)

    if not translations:
        translations.append(torch.tensor(HIDDEN_POSITION, device=device, dtype=torch.float32))
        orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        scales.append(torch.full((3,), 0.001, device=device))
        marker_indices.append(4)

    visualizer.visualize(
        translations=torch.stack(translations),
        orientations=torch.stack(orientations),
        scales=torch.stack(scales),
        marker_indices=torch.tensor(marker_indices, device=device, dtype=torch.int64),
    )


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile)) if values else float("nan")


def _format_pos(item: dict[str, Any] | None) -> str:
    if item is None:
        return "none"
    pos = item["display_position"]
    return f"[{pos[0]: .3f},{pos[1]: .3f},{pos[2]: .3f}]"


def _open_csv_writer(path: str) -> tuple[Any | None, csv.DictWriter | None]:
    if not path:
        return None, None
    csv_path = Path(path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file = csv_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "time",
        "recv_hz",
        "xrt_update_hz",
        "dropped_sequence",
        "latency_mean_ms",
        "latency_p95_ms",
        "tracker_count",
        "left_serial",
        "right_serial",
        "left_x",
        "left_y",
        "left_z",
        "right_x",
        "right_y",
        "right_z",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return file, writer


def main() -> None:
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.0, 1.6], [0.0, 0.0, 1.0])

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.82, 0.86))
    light_cfg.func("/World/Light", light_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    receiver = UdpTrackerReceiver(args_cli.host, args_cli.port, args_cli.max_packet_bytes)
    visualizer = _define_markers()
    rotation = np.asarray(args_cli.source_to_world_rotation, dtype=np.float32).reshape(3, 3)
    offset = np.asarray(args_cli.offset, dtype=np.float32)
    calibration_origin: np.ndarray | None = None
    csv_file, csv_writer = _open_csv_writer(args_cli.csv)

    sim.reset()
    print("[xrobotoolkit_tracker_viewer] Setup complete.")
    print(f"[xrobotoolkit_tracker_viewer] Listening on {args_cli.host}:{args_cli.port}.")
    print("[xrobotoolkit_tracker_viewer] Serial mapping can be set with --left-serial and --right-serial.")

    last_print = time.monotonic()
    last_interval_start = last_print
    waiting_printed = False

    try:
        while simulation_app.is_running():
            now = time.monotonic()
            receiver.receive_latest()
            fresh = receiver.is_fresh(args_cli.max_stale_time)

            displayed: dict[str, dict[str, Any]] = {}
            wrist_map = {"left": None, "right": None}
            if fresh and receiver.latest is not None:
                trackers = _valid_trackers(receiver.latest, rotation)
                displayed, calibration_origin = _display_positions(
                    trackers, args_cli.origin_mode, calibration_origin, offset, args_cli.scale
                )
                packet_map = receiver.latest.get("wrist_map", {}) if isinstance(receiver.latest.get("wrist_map", {}), dict) else {}
                wrist_map = _resolve_wrist_map(displayed, packet_map, args_cli.left_serial, args_cli.right_serial)

            _visualize(visualizer, displayed, wrist_map, args_cli.device)
            sim.step()

            if receiver.latest is None and not waiting_printed:
                print(
                    "[xrobotoolkit_tracker_viewer] Waiting for UDP packets. "
                    "Start scripts/xrobotoolkit_motion_tracker_udp_bridge.py in env_xrobotoolkit."
                )
                waiting_printed = True

            if now - last_print >= args_cli.print_interval:
                interval_s = max(now - last_interval_start, 1.0e-9)
                recv_hz = receiver.interval_packets / interval_s
                xrt_update_hz = receiver.interval_source_changes / interval_s
                latency_mean = _safe_mean(receiver.interval_latency_ms)
                latency_p95 = _safe_percentile(receiver.interval_latency_ms, 95.0)
                left_item = displayed.get(wrist_map["left"]) if wrist_map["left"] else None
                right_item = displayed.get(wrist_map["right"]) if wrist_map["right"] else None
                serials = ",".join(sorted(displayed.keys())) if displayed else "none"
                print(
                    "[xrobotoolkit_tracker_viewer] "
                    f"recv_hz={recv_hz:5.1f} xrt_update_hz={xrt_update_hz:5.1f} "
                    f"drop={receiver.interval_dropped_sequence:03d} "
                    f"lat_ms(avg/p95)={latency_mean:5.1f}/{latency_p95:5.1f} "
                    f"trackers={len(displayed)} serials=[{serials}] "
                    f"L={wrist_map['left'] or 'none'} pos={_format_pos(left_item)} "
                    f"R={wrist_map['right'] or 'none'} pos={_format_pos(right_item)}"
                )

                if csv_writer is not None:
                    left_pos = left_item["display_position"] if left_item is not None else [math.nan, math.nan, math.nan]
                    right_pos = right_item["display_position"] if right_item is not None else [math.nan, math.nan, math.nan]
                    csv_writer.writerow(
                        {
                            "time": time.time(),
                            "recv_hz": recv_hz,
                            "xrt_update_hz": xrt_update_hz,
                            "dropped_sequence": receiver.interval_dropped_sequence,
                            "latency_mean_ms": latency_mean,
                            "latency_p95_ms": latency_p95,
                            "tracker_count": len(displayed),
                            "left_serial": wrist_map["left"] or "",
                            "right_serial": wrist_map["right"] or "",
                            "left_x": left_pos[0],
                            "left_y": left_pos[1],
                            "left_z": left_pos[2],
                            "right_x": right_pos[0],
                            "right_y": right_pos[1],
                            "right_z": right_pos[2],
                        }
                    )
                    csv_file.flush()

                receiver.reset_interval()
                last_print = now
                last_interval_start = now
    finally:
        receiver.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[xrobotoolkit_tracker_viewer] stopped.")
    finally:
        simulation_app.close()
