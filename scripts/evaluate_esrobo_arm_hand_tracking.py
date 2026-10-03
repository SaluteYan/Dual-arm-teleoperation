"""Evaluate recorded arm/hand targets through the UDP device, Pink IK and PhysX.

Scheduling and device time use simulation seconds so results do not depend on
GPU throughput. The final tail repeats frozen inputs, or drops the IMU stream,
to measure settling. This is a simulation regression, not a live latency test.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--source-root",
    type=Path,
    default=ROOT,
    help="Optional source checkout for comparing a previous implementation.",
)
parser.add_argument(
    "--body-record-path",
    type=Path,
    default=ROOT / "recordings/pico_body_20260811_214430.jsonl",
)
parser.add_argument(
    "--hand-record-path",
    type=Path,
    default=ROOT / "recordings/senseglove_real_20260818_222013.jsonl",
)
parser.add_argument("--tail-mode", choices=("frozen", "hand_dropout"), default="frozen")
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--tail", type=float, default=4.0)
parser.add_argument("--enable_pinocchio", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration <= 0 or args.tail <= 1:
    parser.error("--duration must be positive and --tail must exceed one second")
sys.path[:0] = [str(args.source_root / "src"), str(args.source_root / "scripts")]
os.environ.setdefault("DUAL_ARM_TELEOP_ROOT", str(ROOT))
import pinocchio  # noqa: F401 -- load the conda Pinocchio before Isaac Sim

app = AppLauncher(args).app
import gymnasium as gym
import numpy as np
import dual_arm_teleop.isaaclab_ext.tasks  # noqa: F401 -- register project tasks
import dual_arm_teleop.isaaclab_ext.devices.udp_bimanual_body_device as device_module
from isaaclab_tasks.utils import parse_env_cfg
from replay_esrobo_arm_hand_recordings import _load_packets, _select


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now


def quat_error(a, b):
    dot = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    )
    return np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))


def main():
    cfg = parse_env_cfg(
        "DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0",
        device=args.device,
        num_envs=1,
    )
    cfg.terminations.time_out = None
    dev_cfg = cfg.teleop_devices.devices["bodytracking_udp"]
    dev_cfg.host = "127.0.0.1"
    dev_cfg.port = 0
    dev_cfg.enable_visualization = False
    dev_cfg.enable_hand_skeleton_visualization = False
    dev_cfg.use_hand_imu_orientation = True
    dev_cfg.print_calibration_events = False
    env = gym.make(
        "DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0", cfg=cfg
    ).unwrapped
    env.reset()
    dev = device_module.UdpBimanualBodyDevice(dev_cfg)
    dev.reset()
    import socket

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    destination = dev._socket.getsockname()
    clock = Clock()
    device_module.time = clock
    _, body = _load_packets(args.body_record_path.expanduser(), "body")
    _, hand = _load_packets(args.hand_record_path.expanduser(), "hand")
    events = _select(body, hand, 0, 0, args.duration)
    term = env.action_manager.get_term("upper_body_ik")
    robot = env.scene["robot"]
    body_ids = [
        robot.body_names.index(f"{side}Hand_link") for side in ("left", "right")
    ]
    frame_offsets = [7, 21] if dev._enable_elbow_ik_targets else [0, 7]
    wrist_ids = [
        robot.joint_names.index(f"{side}_nero_joint{i}")
        for side in ("left", "right")
        for i in (5, 6, 7)
    ]
    cursor = 0
    latest = {}
    rows = []
    dt = env.step_dt
    duration = events[-1][0]
    for step in range(math.ceil((duration + args.tail) / dt)):
        t = step * dt
        clock.now = 100.0 + t
        packets = []
        while cursor < len(events) and events[cursor][0] <= t + 1e-9:
            packet_time, stream, packet = events[cursor]
            latest[stream] = packet
            packets.append((packet_time, stream, packet))
            cursor += 1
        if t > duration:
            packets = [
                (t, stream, packet)
                for stream, packet in latest.items()
                if args.tail_mode != "hand_dropout" or stream == "body"
            ]
        for packet_time, stream, original in packets:
            packet = dict(original)
            sample_ns = int((100.0 + packet_time) * 1e9)
            packet["replay_sender_monotonic_ns"] = sample_ns
            if stream == "hand":
                packet["hand_orientation_is_forearm_relative"] = True
                packet["hand_orientation_sample_monotonic_ns"] = dict.fromkeys(
                    ("left", "right"), sample_ns
                )
            sender.sendto(json.dumps(packet).encode(), destination)
        action = dev.advance()
        rotations = []
        for side in ("left", "right"):
            points = dev._current_arm_points(side)
            mounting = getattr(dev, f"_initial_{side}_pose_matrix")[:3, :3]
            rotations.append(
                dev._attached_hand_skeleton_rotation(side, points, mounting)
            )
        env.step(action[None])
        target = np.stack(
            [action[offset : offset + 7].cpu().numpy() for offset in frame_offsets]
        )
        measured = robot.data.body_link_state_w[0, body_ids, :7].cpu().numpy().copy()
        cmd = (
            term._processed_actions[0, : len(term._isaaclab_controlled_joint_ids)]
            .cpu()
            .numpy()
            .copy()
        )
        rows.append(
            dict(
                t=t,
                target=target.tolist(),
                measured=measured.tolist(),
                wrist=robot.data.joint_pos[0, wrist_ids].cpu().numpy().tolist(),
                command=cmd.tolist(),
                visual_error=[
                    float(
                        quat_error(
                            device_module._matrix_to_quat_wxyz(rotations[i]),
                            target[i, 3:],
                        )
                    )
                    for i in range(2)
                ],
                hold=term._static_target_hold_active.cpu().numpy().tolist(),
            )
        )
        if step % 300 == 0:
            print(f"[tracking-eval] step={step} t={t:.2f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows))
    measured = np.asarray([row["measured"] for row in rows])
    target = np.asarray([row["target"] for row in rows])
    errors = quat_error(measured[..., 3:7], target[..., 3:7])
    pos_errors = np.linalg.norm(measured[..., :3] - target[..., :3], axis=-1) * 1000
    tail = [row for row in rows if row["t"] >= duration + 1]
    orientation_tail = np.asarray([
        row["target"] for row in rows if row["t"] >= duration + 0.25
    ])[..., 3:7]
    summary = dict(
        orientation_mean_deg=errors.mean(axis=0).tolist(),
        orientation_p95_deg=np.percentile(errors, 95, axis=0).tolist(),
        position_mean_mm=pos_errors.mean(axis=0).tolist(),
        visual_p95_deg=np.percentile(
            [r["visual_error"] for r in rows], 95, axis=0
        ).tolist(),
        final_orientation_deg=errors[-1].tolist(),
        tail_wrist_range_rad=np.ptp([r["wrist"] for r in tail], axis=0).tolist(),
        tail_command_range_rad=np.ptp([r["command"] for r in tail], axis=0).tolist(),
        tail_target_rotation_range_deg=quat_error(
            orientation_tail, orientation_tail[0]
        ).max(axis=0).tolist(),
        final_hold=rows[-1]["hold"],
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print("[tracking-eval summary]", json.dumps(summary), flush=True)
    sender.close()
    dev._socket.close()
    env.close()


try:
    main()
finally:
    app.close()
