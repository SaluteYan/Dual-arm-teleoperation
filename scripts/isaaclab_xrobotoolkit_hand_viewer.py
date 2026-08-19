#!/usr/bin/env python3
"""Visualize XRoboToolkit PICO hand tracking in IsaacLab and print stream metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host", default="0.0.0.0", help="UDP bind host.")
parser.add_argument("--port", type=int, default=15100, help="UDP bind port.")
parser.add_argument("--max-packet-bytes", type=int, default=65535, help="UDP receive buffer size.")
parser.add_argument("--max-stale-time", type=float, default=0.5, help="Hide hands after this many seconds without packets.")
parser.add_argument(
    "--display-mode",
    choices=("wrist-relative", "absolute"),
    default="wrist-relative",
    help="Show hand shape around fixed anchors or raw translated poses.",
)
parser.add_argument("--scale", type=float, default=1.0, help="Display scale for hand positions.")
parser.add_argument(
    "--offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 1.1),
    metavar=("X", "Y", "Z"),
    help="World offset used in absolute display mode.",
)
parser.add_argument(
    "--left-anchor",
    type=float,
    nargs=3,
    default=(0.0, 0.22, 1.1),
    metavar=("X", "Y", "Z"),
    help="Anchor for left wrist in wrist-relative display mode.",
)
parser.add_argument(
    "--right-anchor",
    type=float,
    nargs=3,
    default=(0.0, -0.22, 1.1),
    metavar=("X", "Y", "Z"),
    help="Anchor for right wrist in wrist-relative display mode.",
)
parser.add_argument(
    "--source-to-world-rotation",
    type=float,
    nargs=9,
    default=(0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
    help="Row-major 3x3 rotation applied to PICO positions before display.",
)
parser.add_argument("--stats-window", type=float, default=5.0, help="Window for static jitter metrics in seconds.")
parser.add_argument("--print-interval", type=float, default=1.0, help="Metrics print interval in seconds.")
parser.add_argument("--csv", default="", help="Optional CSV path for interval metrics.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext


PICO_HAND_JOINT_NAMES = [
    "Palm",
    "Wrist",
    "Thumb_metacarpal",
    "Thumb_proximal",
    "Thumb_distal",
    "Thumb_tip",
    "Index_metacarpal",
    "Index_proximal",
    "Index_intermediate",
    "Index_distal",
    "Index_tip",
    "Middle_metacarpal",
    "Middle_proximal",
    "Middle_intermediate",
    "Middle_distal",
    "Middle_tip",
    "Ring_metacarpal",
    "Ring_proximal",
    "Ring_intermediate",
    "Ring_distal",
    "Ring_tip",
    "Little_metacarpal",
    "Little_proximal",
    "Little_intermediate",
    "Little_distal",
    "Little_tip",
]

HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (1, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (1, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
    (6, 11),
    (11, 16),
    (16, 21),
]
FINGERTIP_INDICES = (5, 10, 15, 20, 25)
HIDDEN_POSITION = np.asarray([0.0, 0.0, -100.0], dtype=np.float32)


class UdpHandReceiver:
    """Non-blocking receiver that keeps only the latest hand-tracking packet."""

    def __init__(self, host: str, port: int, max_packet_bytes: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((host, port))
        self.max_packet_bytes = max_packet_bytes
        self.latest: dict[str, Any] | None = None
        self.last_sender: tuple[str, int] | None = None
        self.last_receive_time: float | None = None
        self.last_receive_monotonic_ns: int | None = None
        self.total_packets = 0
        self.total_dropped_sequence = 0
        self._last_sequence: int | None = None
        self.interval_packets = 0
        self.interval_source_changes = 0
        self.interval_dropped_sequence = 0
        self.interval_latency_ms: list[float] = []
        self._last_xrt_timestamp_ns: int | None = None

    def close(self) -> None:
        self.socket.close()

    def receive_latest(self) -> bool:
        received_any = False
        while True:
            try:
                payload, sender = self.socket.recvfrom(self.max_packet_bytes)
            except BlockingIOError:
                break
            message = json.loads(payload.decode("utf-8"))
            now = time.monotonic()
            now_ns = time.monotonic_ns()

            sequence = message.get("sequence")
            if isinstance(sequence, int) and self._last_sequence is not None and sequence > self._last_sequence + 1:
                dropped = sequence - self._last_sequence - 1
                self.total_dropped_sequence += dropped
                self.interval_dropped_sequence += dropped
            if isinstance(sequence, int):
                self._last_sequence = sequence

            xrt_timestamp_ns = message.get("xrt_timestamp_ns")
            if isinstance(xrt_timestamp_ns, int) and xrt_timestamp_ns != self._last_xrt_timestamp_ns:
                self.interval_source_changes += 1
                self._last_xrt_timestamp_ns = xrt_timestamp_ns

            sender_monotonic_ns = message.get("sender_monotonic_ns")
            if isinstance(sender_monotonic_ns, int):
                self.interval_latency_ms.append((now_ns - sender_monotonic_ns) / 1.0e6)

            self.latest = message
            self.last_sender = sender
            self.last_receive_time = now
            self.last_receive_monotonic_ns = now_ns
            self.total_packets += 1
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


class StaticHandStats:
    """Sliding-window static stability metrics for PICO hand poses."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self.samples: dict[str, deque[tuple[float, np.ndarray]]] = {
            "left": deque(),
            "right": deque(),
        }

    def add(self, side: str, positions: np.ndarray, now: float) -> None:
        samples = self.samples[side]
        samples.append((now, positions.copy()))
        self._trim(side, now)

    def _trim(self, side: str, now: float) -> None:
        samples = self.samples[side]
        while samples and now - samples[0][0] > self.window_s:
            samples.popleft()

    def summarize(self, side: str, now: float) -> dict[str, float]:
        self._trim(side, now)
        samples = self.samples[side]
        if len(samples) < 2:
            return {
                "samples": float(len(samples)),
                "wrist_rms_mm": float("nan"),
                "tip_rel_rms_mm": float("nan"),
                "bone_std_mean_mm": float("nan"),
                "bone_std_max_mm": float("nan"),
            }

        stack = np.stack([positions for _, positions in samples], axis=0)
        wrist = stack[:, 1, :]
        wrist_rms_mm = _mean_axis_std_norm_mm(wrist)

        tip_relative = stack[:, FINGERTIP_INDICES, :] - stack[:, 1:2, :]
        tip_rel_rms_mm = _mean_joint_std_norm_mm(tip_relative)

        bone_stds = []
        for start, end in HAND_BONES:
            lengths = np.linalg.norm(stack[:, end, :] - stack[:, start, :], axis=1)
            bone_stds.append(float(np.std(lengths)) * 1000.0)

        return {
            "samples": float(len(samples)),
            "wrist_rms_mm": wrist_rms_mm,
            "tip_rel_rms_mm": tip_rel_rms_mm,
            "bone_std_mean_mm": float(np.mean(bone_stds)) if bone_stds else float("nan"),
            "bone_std_max_mm": float(np.max(bone_stds)) if bone_stds else float("nan"),
        }


def _mean_axis_std_norm_mm(points_over_time: np.ndarray) -> float:
    std_xyz = np.std(points_over_time, axis=0)
    return float(np.linalg.norm(std_xyz) * 1000.0)


def _mean_joint_std_norm_mm(joints_over_time: np.ndarray) -> float:
    std_xyz = np.std(joints_over_time, axis=0)
    per_joint = np.linalg.norm(std_xyz, axis=1)
    return float(np.mean(per_joint) * 1000.0)


def _safe_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def _parse_hand_positions(message: dict[str, Any], side: str) -> np.ndarray | None:
    hand = message.get("hands", {}).get(side, {})
    if not hand.get("active", False):
        return None
    joints = hand.get("joints", [])
    if len(joints) < len(PICO_HAND_JOINT_NAMES):
        return None
    positions = np.asarray([row[:3] for row in joints[: len(PICO_HAND_JOINT_NAMES)]], dtype=np.float32)
    if positions.shape != (len(PICO_HAND_JOINT_NAMES), 3) or not np.all(np.isfinite(positions)):
        return None
    return positions


def _transform_positions(
    positions: np.ndarray,
    side: str,
    display_mode: str,
    rotation: np.ndarray,
    scale: float,
    offset: np.ndarray,
    left_anchor: np.ndarray,
    right_anchor: np.ndarray,
) -> np.ndarray:
    transformed = (rotation @ positions.T).T
    if display_mode == "wrist-relative":
        transformed = transformed - transformed[1:2, :]
        anchor = left_anchor if side == "left" else right_anchor
        return anchor + scale * transformed
    return offset + scale * transformed


def _define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/XRoboToolkitPicoHands",
        markers={
            "left_joint": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.65, 1.0), roughness=0.7),
            ),
            "right_joint": sim_utils.SphereCfg(
                radius=0.010,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.08), roughness=0.7),
            ),
            "left_wrist": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.95, 1.0), roughness=0.6),
            ),
            "right_wrist": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.0), roughness=0.6),
            ),
            "left_bone": sim_utils.CylinderCfg(
                radius=0.004,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.65, 1.0), roughness=0.85),
            ),
            "right_bone": sim_utils.CylinderCfg(
                radius=0.004,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.08), roughness=0.85),
            ),
            "hidden": sim_utils.SphereCfg(
                radius=0.001,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), roughness=1.0),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _line_markers(start_pos: torch.Tensor, end_pos: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direction = end_pos - start_pos
    lengths = torch.norm(direction, dim=-1)
    positions = 0.5 * (start_pos + end_pos)
    default_direction = torch.tensor([0.0, 0.0, 1.0], device=device).expand(start_pos.size(0), -1)
    safe_direction = torch.where(
        (lengths > 1.0e-6).unsqueeze(-1),
        direction,
        default_direction,
    )
    direction_norm = math_utils.normalize(safe_direction)
    rotation_axis = torch.linalg.cross(default_direction, direction_norm)
    rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
    rotation_axis = torch.where(
        (rotation_axis_norm > 1.0e-6).unsqueeze(-1),
        math_utils.normalize(rotation_axis),
        torch.tensor([1.0, 0.0, 0.0], device=device).expand(start_pos.size(0), -1),
    )
    cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    orientations = math_utils.quat_from_angle_axis(angle, rotation_axis)
    return positions, orientations, lengths


def _append_hidden(
    count: int,
    translations: list[torch.Tensor],
    orientations: list[torch.Tensor],
    scales: list[torch.Tensor],
    marker_indices: list[int],
    device: str,
) -> None:
    hidden = torch.tensor(HIDDEN_POSITION, device=device, dtype=torch.float32)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    scale = torch.full((3,), 0.001, device=device)
    for _ in range(count):
        translations.append(hidden)
        orientations.append(quat)
        scales.append(scale)
        marker_indices.append(6)


def _append_hand_markers(
    side: str,
    positions: np.ndarray | None,
    translations: list[torch.Tensor],
    orientations: list[torch.Tensor],
    scales: list[torch.Tensor],
    marker_indices: list[int],
    device: str,
) -> None:
    if positions is None:
        _append_hidden(len(PICO_HAND_JOINT_NAMES) + len(HAND_BONES), translations, orientations, scales, marker_indices, device)
        return

    point_marker_index = 0 if side == "left" else 1
    wrist_marker_index = 2 if side == "left" else 3
    bone_marker_index = 4 if side == "left" else 5
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    unit_scale = torch.ones(3, device=device)

    for joint_index, point in enumerate(positions):
        translations.append(torch.tensor(point, device=device, dtype=torch.float32))
        orientations.append(quat)
        scales.append(unit_scale)
        marker_indices.append(wrist_marker_index if joint_index == 1 else point_marker_index)

    bone_starts = torch.tensor([positions[start] for start, _ in HAND_BONES], device=device, dtype=torch.float32)
    bone_ends = torch.tensor([positions[end] for _, end in HAND_BONES], device=device, dtype=torch.float32)
    bone_positions, bone_orientations, bone_lengths = _line_markers(bone_starts, bone_ends, device)
    for bone_index in range(len(HAND_BONES)):
        translations.append(bone_positions[bone_index])
        orientations.append(bone_orientations[bone_index])
        scale = torch.ones(3, device=device)
        scale[-1] = max(float(bone_lengths[bone_index].item()), 1.0e-4)
        scales.append(scale)
        marker_indices.append(bone_marker_index)


def _visualize_hands(
    visualizer: VisualizationMarkers,
    left_positions: np.ndarray | None,
    right_positions: np.ndarray | None,
    device: str,
) -> None:
    translations: list[torch.Tensor] = []
    orientations: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    marker_indices: list[int] = []
    _append_hand_markers("left", left_positions, translations, orientations, scales, marker_indices, device)
    _append_hand_markers("right", right_positions, translations, orientations, scales, marker_indices, device)
    visualizer.visualize(
        translations=torch.stack(translations),
        orientations=torch.stack(orientations),
        scales=torch.stack(scales),
        marker_indices=torch.tensor(marker_indices, device=device, dtype=torch.int64),
    )


def _format_metric(value: float, width: int = 6, precision: int = 1) -> str:
    if not math.isfinite(value):
        return " " * (width - 3) + "nan"
    return f"{value:{width}.{precision}f}"


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
        "left_active",
        "right_active",
        "left_wrist_rms_mm",
        "right_wrist_rms_mm",
        "left_tip_rel_rms_mm",
        "right_tip_rel_rms_mm",
        "left_bone_std_mean_mm",
        "right_bone_std_mean_mm",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return file, writer


def main() -> None:
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.25, -1.8, 1.55], [0.0, 0.0, 1.0])

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.82, 0.86))
    light_cfg.func("/World/Light", light_cfg)
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    receiver = UdpHandReceiver(args_cli.host, args_cli.port, args_cli.max_packet_bytes)
    stats = StaticHandStats(args_cli.stats_window)
    visualizer = _define_markers()
    rotation = np.asarray(args_cli.source_to_world_rotation, dtype=np.float32).reshape(3, 3)
    offset = np.asarray(args_cli.offset, dtype=np.float32)
    left_anchor = np.asarray(args_cli.left_anchor, dtype=np.float32)
    right_anchor = np.asarray(args_cli.right_anchor, dtype=np.float32)
    csv_file, csv_writer = _open_csv_writer(args_cli.csv)

    sim.reset()
    print("[xrobotoolkit_hand_viewer] Setup complete.")
    print(f"[xrobotoolkit_hand_viewer] Listening on {args_cli.host}:{args_cli.port}.")
    print("[xrobotoolkit_hand_viewer] Hold hands still for 5s to interpret RMS values as static precision.")

    last_print = time.monotonic()
    last_print_time = last_print
    waiting_printed = False

    try:
        while simulation_app.is_running():
            now = time.monotonic()
            receiver.receive_latest()
            fresh = receiver.is_fresh(args_cli.max_stale_time)

            left_display = None
            right_display = None
            left_active = False
            right_active = False
            if fresh and receiver.latest is not None:
                left_raw = _parse_hand_positions(receiver.latest, "left")
                right_raw = _parse_hand_positions(receiver.latest, "right")
                left_active = left_raw is not None
                right_active = right_raw is not None

                if left_raw is not None:
                    stats.add("left", left_raw, now)
                    left_display = _transform_positions(
                        left_raw,
                        "left",
                        args_cli.display_mode,
                        rotation,
                        args_cli.scale,
                        offset,
                        left_anchor,
                        right_anchor,
                    )
                if right_raw is not None:
                    stats.add("right", right_raw, now)
                    right_display = _transform_positions(
                        right_raw,
                        "right",
                        args_cli.display_mode,
                        rotation,
                        args_cli.scale,
                        offset,
                        left_anchor,
                        right_anchor,
                    )

            _visualize_hands(visualizer, left_display, right_display, args_cli.device)
            sim.step()

            if receiver.latest is None and not waiting_printed:
                print(
                    "[xrobotoolkit_hand_viewer] Waiting for UDP packets. "
                    "Start scripts/xrobotoolkit_hand_udp_bridge.py in env_xrobotoolkit."
                )
                waiting_printed = True

            if now - last_print >= args_cli.print_interval:
                interval_s = max(now - last_print_time, 1.0e-9)
                recv_hz = receiver.interval_packets / interval_s
                xrt_update_hz = receiver.interval_source_changes / interval_s
                latency_mean = float(np.mean(receiver.interval_latency_ms)) if receiver.interval_latency_ms else float("nan")
                latency_p95 = _safe_percentile(receiver.interval_latency_ms, 95.0)
                left_summary = stats.summarize("left", now)
                right_summary = stats.summarize("right", now)

                print(
                    "[xrobotoolkit_hand_viewer] "
                    f"recv_hz={recv_hz:5.1f} xrt_update_hz={xrt_update_hz:5.1f} "
                    f"drop={receiver.interval_dropped_sequence:03d} "
                    f"lat_ms(avg/p95)={_format_metric(latency_mean)}/{_format_metric(latency_p95)} "
                    f"L={'on ' if left_active else 'off'} "
                    f"wrist_rms={_format_metric(left_summary['wrist_rms_mm'])}mm "
                    f"tip_rel_rms={_format_metric(left_summary['tip_rel_rms_mm'])}mm "
                    f"bone_std={_format_metric(left_summary['bone_std_mean_mm'])}mm | "
                    f"R={'on ' if right_active else 'off'} "
                    f"wrist_rms={_format_metric(right_summary['wrist_rms_mm'])}mm "
                    f"tip_rel_rms={_format_metric(right_summary['tip_rel_rms_mm'])}mm "
                    f"bone_std={_format_metric(right_summary['bone_std_mean_mm'])}mm"
                )

                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "time": time.time(),
                            "recv_hz": recv_hz,
                            "xrt_update_hz": xrt_update_hz,
                            "dropped_sequence": receiver.interval_dropped_sequence,
                            "latency_mean_ms": latency_mean,
                            "latency_p95_ms": latency_p95,
                            "left_active": int(left_active),
                            "right_active": int(right_active),
                            "left_wrist_rms_mm": left_summary["wrist_rms_mm"],
                            "right_wrist_rms_mm": right_summary["wrist_rms_mm"],
                            "left_tip_rel_rms_mm": left_summary["tip_rel_rms_mm"],
                            "right_tip_rel_rms_mm": right_summary["tip_rel_rms_mm"],
                            "left_bone_std_mean_mm": left_summary["bone_std_mean_mm"],
                            "right_bone_std_mean_mm": right_summary["bone_std_mean_mm"],
                        }
                    )
                    csv_file.flush()

                receiver.reset_interval()
                last_print = now
                last_print_time = now
    finally:
        receiver.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[xrobotoolkit_hand_viewer] stopped.")
    finally:
        simulation_app.close()
