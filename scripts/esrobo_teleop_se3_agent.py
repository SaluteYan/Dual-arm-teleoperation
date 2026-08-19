# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--debug_xr_data",
    action="store_true",
    default=False,
    help="Print raw device data and retargeted teleop actions periodically.",
)
parser.add_argument(
    "--debug_xr_interval",
    type=int,
    default=30,
    help="Number of simulation frames between XR debug prints.",
)
parser.add_argument(
    "--debug_perf",
    action="store_true",
    default=False,
    help="Print approximate teleoperation loop frequency periodically.",
)
parser.add_argument(
    "--debug_eef_error",
    action="store_true",
    default=False,
    help="Print arm-segment posture, joint motion, and auxiliary position tracking diagnostics.",
)
parser.add_argument(
    "--xr_start_active",
    action="store_true",
    default=False,
    help=(
        "Start applying XR teleoperation commands immediately instead of waiting for the headset client's Play"
        " command. Use only after aligning the XR hands with the robot."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if any(
    name in args_cli.teleop_device.lower()
    for name in ("handtracking", "motioncontroller", "openxr")
):
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
import dual_arm_teleop.isaaclab_ext.tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def _get_render_antialiasing_mode() -> str | None:
    """Return the requested RTX realtime anti-aliasing mode."""
    raw_mode = os.environ.get("ESROBO_RENDER_ANTIALIASING_MODE", "DLSS").strip()
    if not raw_mode:
        return None

    normalized_modes = {
        "off": "Off",
        "none": "Off",
        "fxaa": "FXAA",
        "dlss": "DLSS",
        "taa": "TAA",
        "dlaa": "DLAA",
    }
    mode = normalized_modes.get(raw_mode.lower())
    if mode is None:
        print(
            f"[teleop] Unknown ESROBO_RENDER_ANTIALIASING_MODE={raw_mode!r}; using DLSS.",
            flush=True,
        )
        mode = "DLSS"
    return mode


def _initialize_eef_error_debug(env) -> dict[str, object] | None:
    """Resolve robot/action handles used by the optional arm-posture diagnostics."""
    try:
        robot = env.scene["robot"]
        body_names = robot.body_names
        left_body_idx = body_names.index("leftHand_link")
        right_body_idx = body_names.index("rightHand_link")
        left_shoulder_body_idx = body_names.index("left_nero_link2")
        right_shoulder_body_idx = body_names.index("right_nero_link2")
        left_elbow_body_idx = body_names.index("left_nero_link3") if "left_nero_link3" in body_names else None
        right_elbow_body_idx = body_names.index("right_nero_link3") if "right_nero_link3" in body_names else None
        locked_joint_ids = [
            robot.joint_names.index(joint_name)
            for joint_name in ("waist_joint1", "waist_joint2", "waist_joint3", "head_joint1", "head_joint2")
            if joint_name in robot.joint_names
        ]
    except Exception as exc:
        print(f"[teleop eef] Could not enable EEF error debug: {exc}", flush=True)
        return None

    action_term = None
    controlled_joint_ids = None
    hand_joint_dim = 0
    left_elbow_action_start = None
    left_hand_action_start = 0
    right_elbow_action_start = None
    right_hand_action_start = 7
    try:
        action_term = env.action_manager.get_term("upper_body_ik")
        controlled_joint_ids = list(getattr(action_term, "_controlled_joint_ids", []))
        hand_joint_dim = int(getattr(action_term, "hand_joint_dim", 0))
        frame_task_names = [
            getattr(task, "frame", "")
            for task in action_term.cfg.controller.variable_input_tasks
            if hasattr(task, "frame")
        ]
        if "left_nero_link3" in frame_task_names:
            left_elbow_action_start = frame_task_names.index("left_nero_link3") * 7
        if "leftHand_link" in frame_task_names:
            left_hand_action_start = frame_task_names.index("leftHand_link") * 7
        if "right_nero_link3" in frame_task_names:
            right_elbow_action_start = frame_task_names.index("right_nero_link3") * 7
        if "rightHand_link" in frame_task_names:
            right_hand_action_start = frame_task_names.index("rightHand_link") * 7
    except Exception as exc:
        print(f"[teleop eef] Joint target error debug unavailable: {exc}", flush=True)

    body_device_cfg = env.cfg.teleop_devices.devices.get("bodytracking_udp")
    target_left_shoulder = torch.tensor(body_device_cfg.robot_left_shoulder_position, device=robot.device)
    target_right_shoulder = torch.tensor(body_device_cfg.robot_right_shoulder_position, device=robot.device)

    print("[teleop posture] Arm-segment and hand-joint tracking diagnostics enabled.", flush=True)
    return {
        "robot": robot,
        "left_body_idx": left_body_idx,
        "right_body_idx": right_body_idx,
        "left_shoulder_body_idx": left_shoulder_body_idx,
        "right_shoulder_body_idx": right_shoulder_body_idx,
        "left_elbow_body_idx": left_elbow_body_idx,
        "right_elbow_body_idx": right_elbow_body_idx,
        "target_left_shoulder": target_left_shoulder,
        "target_right_shoulder": target_right_shoulder,
        "action_term": action_term,
        "controlled_joint_ids": controlled_joint_ids,
        "hand_joint_dim": hand_joint_dim,
        "left_elbow_action_start": left_elbow_action_start,
        "left_hand_action_start": left_hand_action_start,
        "right_elbow_action_start": right_elbow_action_start,
        "right_hand_action_start": right_hand_action_start,
        "locked_joint_ids": locked_joint_ids,
        "previous_arm_joint_targets": None,
        "previous_hand_joint_targets": None,
    }


def _vector_angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the unsigned angle between two 3D vectors in degrees."""
    denominator = torch.linalg.norm(first) * torch.linalg.norm(second)
    if denominator <= 1.0e-8:
        return torch.zeros((), device=first.device)
    cosine = torch.clamp(torch.dot(first, second) / denominator, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _print_eef_error_debug(env, debug_state: dict[str, object], actions: torch.Tensor, frame: int) -> None:
    """Print arm posture tracking, joint motion, and auxiliary position errors."""
    robot = debug_state["robot"]
    left_body_idx = int(debug_state["left_body_idx"])
    right_body_idx = int(debug_state["right_body_idx"])

    current_left = robot.data.body_link_state_w[0, left_body_idx, :3] - env.scene.env_origins[0]
    current_right = robot.data.body_link_state_w[0, right_body_idx, :3] - env.scene.env_origins[0]
    left_hand_action_start = int(debug_state.get("left_hand_action_start", 0))
    right_hand_action_start = int(debug_state.get("right_hand_action_start", 7))
    target_left = actions[0, left_hand_action_start : left_hand_action_start + 3]
    target_right = actions[0, right_hand_action_start : right_hand_action_start + 3]
    left_error = target_left - current_left
    right_error = target_right - current_right

    posture_text = ""
    elbow_text = ""
    left_elbow_body_idx = debug_state.get("left_elbow_body_idx")
    right_elbow_body_idx = debug_state.get("right_elbow_body_idx")
    left_elbow_action_start = debug_state.get("left_elbow_action_start")
    right_elbow_action_start = debug_state.get("right_elbow_action_start")
    if (
        left_elbow_body_idx is not None
        and right_elbow_body_idx is not None
        and left_elbow_action_start is not None
        and right_elbow_action_start is not None
    ):
        current_left_elbow = robot.data.body_link_state_w[0, int(left_elbow_body_idx), :3] - env.scene.env_origins[0]
        current_right_elbow = robot.data.body_link_state_w[0, int(right_elbow_body_idx), :3] - env.scene.env_origins[0]
        target_left_elbow = actions[0, int(left_elbow_action_start) : int(left_elbow_action_start) + 3]
        target_right_elbow = actions[0, int(right_elbow_action_start) : int(right_elbow_action_start) + 3]
        left_elbow_error = target_left_elbow - current_left_elbow
        right_elbow_error = target_right_elbow - current_right_elbow
        current_left_shoulder = (
            robot.data.body_link_state_w[0, int(debug_state["left_shoulder_body_idx"]), :3]
            - env.scene.env_origins[0]
        )
        current_right_shoulder = (
            robot.data.body_link_state_w[0, int(debug_state["right_shoulder_body_idx"]), :3]
            - env.scene.env_origins[0]
        )
        target_left_shoulder = debug_state["target_left_shoulder"]
        target_right_shoulder = debug_state["target_right_shoulder"]
        left_upper_target = target_left_elbow - target_left_shoulder
        right_upper_target = target_right_elbow - target_right_shoulder
        left_forearm_target = target_left - target_left_elbow
        right_forearm_target = target_right - target_right_elbow
        left_upper_current = current_left_elbow - current_left_shoulder
        right_upper_current = current_right_elbow - current_right_shoulder
        left_forearm_current = current_left - current_left_elbow
        right_forearm_current = current_right - current_right_elbow
        left_upper_angle_error = _vector_angle_degrees(left_upper_current, left_upper_target)
        right_upper_angle_error = _vector_angle_degrees(right_upper_current, right_upper_target)
        left_forearm_angle_error = _vector_angle_degrees(left_forearm_current, left_forearm_target)
        right_forearm_angle_error = _vector_angle_degrees(right_forearm_current, right_forearm_target)
        left_bend_target = _vector_angle_degrees(left_upper_target, left_forearm_target)
        right_bend_target = _vector_angle_degrees(right_upper_target, right_forearm_target)
        left_bend_current = _vector_angle_degrees(left_upper_current, left_forearm_current)
        right_bend_current = _vector_angle_degrees(right_upper_current, right_forearm_current)
        posture_text = (
            f" L_upper_ang_err={left_upper_angle_error.detach().cpu().item():.2f}deg"
            f" R_upper_ang_err={right_upper_angle_error.detach().cpu().item():.2f}deg"
            f" L_forearm_ang_err={left_forearm_angle_error.detach().cpu().item():.2f}deg"
            f" R_forearm_ang_err={right_forearm_angle_error.detach().cpu().item():.2f}deg"
            f" L_bend_ang_err={torch.abs(left_bend_current - left_bend_target).detach().cpu().item():.2f}deg"
            f" R_bend_ang_err={torch.abs(right_bend_current - right_bend_target).detach().cpu().item():.2f}deg"
        )
        elbow_text = (
            f" L_elbow_err={torch.linalg.norm(left_elbow_error).detach().cpu().item():.4f}m"
            f" R_elbow_err={torch.linalg.norm(right_elbow_error).detach().cpu().item():.4f}m"
            f" L_elbow_target={np.round(target_left_elbow.detach().cpu().numpy(), 4).tolist()}"
            f" R_elbow_target={np.round(target_right_elbow.detach().cpu().numpy(), 4).tolist()}"
        )

    joint_text = ""
    hand_text = ""
    action_term = debug_state.get("action_term")
    controlled_joint_ids = debug_state.get("controlled_joint_ids")
    if action_term is not None and controlled_joint_ids:
        processed_actions = getattr(action_term, "processed_actions", None)
        if processed_actions is not None:
            joint_ids = list(controlled_joint_ids)
            hand_joint_dim = int(debug_state.get("hand_joint_dim", 0))
            arm_joint_dim = max(len(joint_ids) - hand_joint_dim, 0)
            if arm_joint_dim > 0:
                current_arm_joints = robot.data.joint_pos[0, joint_ids[:arm_joint_dim]]
                target_arm_joints = processed_actions[0, :arm_joint_dim]
                arm_joint_error = target_arm_joints - current_arm_joints
                arm_joint_velocity = robot.data.joint_vel[0, joint_ids[:arm_joint_dim]]
                previous_arm_joint_targets = debug_state.get("previous_arm_joint_targets")
                target_step_max = 0.0
                if previous_arm_joint_targets is not None:
                    target_step_max = torch.max(
                        torch.abs(target_arm_joints - previous_arm_joint_targets)
                    ).detach().cpu().item()
                debug_state["previous_arm_joint_targets"] = target_arm_joints.detach().clone()
                joint_text = (
                    f" arm_joint_err_max={torch.max(torch.abs(arm_joint_error)).detach().cpu().item():.4f}rad"
                    f" arm_joint_err_mean={torch.mean(torch.abs(arm_joint_error)).detach().cpu().item():.4f}rad"
                    f" arm_joint_vel_max={torch.max(torch.abs(arm_joint_velocity)).detach().cpu().item():.4f}rad/s"
                    f" arm_joint_target_step_max={target_step_max:.4f}rad"
                )
                velocity_targets = getattr(action_term, "_last_arm_velocity_targets", None)
                if velocity_targets is not None:
                    joint_text += (
                        " arm_velocity_ff_max="
                        f"{torch.max(torch.abs(velocity_targets[0])).detach().cpu().item():.3f}rad/s"
                    )
                velocity_activation = getattr(action_term, "_velocity_feedforward_activation", None)
                if velocity_activation is not None:
                    joint_text += (
                        " arm_velocity_ff_activation="
                        f"{torch.max(velocity_activation[0]).detach().cpu().item():.2f}"
                    )
            if hand_joint_dim > 0:
                hand_ids = joint_ids[arm_joint_dim : arm_joint_dim + hand_joint_dim]
                current_hand_joints = robot.data.joint_pos[0, hand_ids]
                target_hand_joints = processed_actions[0, arm_joint_dim : arm_joint_dim + hand_joint_dim]
                hand_joint_error = target_hand_joints - current_hand_joints
                hand_joint_velocity = robot.data.joint_vel[0, hand_ids]
                side_dim = hand_joint_dim // 2
                left_hand_joint_error = torch.abs(hand_joint_error[:side_dim])
                right_hand_joint_error = torch.abs(hand_joint_error[side_dim:])
                left_velocity = torch.abs(hand_joint_velocity[:side_dim])
                right_velocity = torch.abs(hand_joint_velocity[side_dim:])
                previous_hand_joint_targets = debug_state.get("previous_hand_joint_targets")
                hand_target_step_max = 0.0
                if previous_hand_joint_targets is not None:
                    hand_target_step_max = torch.max(
                        torch.abs(target_hand_joints - previous_hand_joint_targets)
                    ).detach().cpu().item()
                debug_state["previous_hand_joint_targets"] = target_hand_joints.detach().clone()
                hand_text = (
                    f" L_hand_err_max={torch.max(left_hand_joint_error).detach().cpu().item():.4f}rad"
                    f" R_hand_err_max={torch.max(right_hand_joint_error).detach().cpu().item():.4f}rad"
                    f" L_hand_err_mean={torch.mean(left_hand_joint_error).detach().cpu().item():.4f}rad"
                    f" R_hand_err_mean={torch.mean(right_hand_joint_error).detach().cpu().item():.4f}rad"
                    f" L_hand_vel_max={torch.max(left_velocity).detach().cpu().item():.4f}rad/s"
                    f" R_hand_vel_max={torch.max(right_velocity).detach().cpu().item():.4f}rad/s"
                    f" L_hand_target_max={torch.max(target_hand_joints[:side_dim]).detach().cpu().item():.4f}rad"
                    f" R_hand_target_max={torch.max(target_hand_joints[side_dim:]).detach().cpu().item():.4f}rad"
                    f" hand_target_step_max={hand_target_step_max:.4f}rad"
                )
    locked_joint_text = ""
    locked_joint_ids = debug_state.get("locked_joint_ids")
    if locked_joint_ids:
        locked_ids = list(locked_joint_ids)
        locked_joint_pos = robot.data.joint_pos[0, locked_ids]
        locked_joint_vel = robot.data.joint_vel[0, locked_ids]
        locked_joint_default = robot.data.default_joint_pos[0, locked_ids]
        locked_joint_error = locked_joint_pos - locked_joint_default
        locked_joint_text = (
            f" locked_joint_abs_max={torch.max(torch.abs(locked_joint_error)).detach().cpu().item():.5f}rad"
            f" locked_joint_vel_max={torch.max(torch.abs(locked_joint_vel)).detach().cpu().item():.5f}rad/s"
        )

    print(
        "[teleop posture] "
        f"frame={frame} "
        f"L_target={np.round(target_left.detach().cpu().numpy(), 4).tolist()} "
        f"L_current={np.round(current_left.detach().cpu().numpy(), 4).tolist()} "
        f"L_err={torch.linalg.norm(left_error).detach().cpu().item():.4f}m "
        f"R_target={np.round(target_right.detach().cpu().numpy(), 4).tolist()} "
        f"R_current={np.round(current_right.detach().cpu().numpy(), 4).tolist()} "
        f"R_err={torch.linalg.norm(right_error).detach().cpu().item():.4f}m"
        f"{posture_text}"
        f"{elbow_text}"
        f"{joint_text}",
        f"{hand_text}",
        f"{locked_joint_text}",
        flush=True,
    )


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    antialiasing_mode = _get_render_antialiasing_mode()
    if antialiasing_mode is not None:
        env_cfg.sim.render.antialiasing_mode = antialiasing_mode
        print(f"[teleop] RTX realtime anti-aliasing mode: {antialiasing_mode}.", flush=True)
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleop_device_name = args_cli.teleop_device.lower()
    wait_for_xr_start_command = (
        args_cli.xr and not args_cli.xr_start_active and "bodytracking_udp" not in teleop_device_name
    )
    teleoperation_active = not wait_for_xr_start_command

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        try:
            current_teleop_interface = teleop_interface
        except NameError:
            current_teleop_interface = None
        if current_teleop_interface is not None and hasattr(current_teleop_interface, "reset_retargeters"):
            current_teleop_interface.reset_retargeters()
            print("[teleop] Teleop calibration reset. Hold the zero pose for the active input device.")
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For hand tracking devices, add additional callbacks
    if args_cli.xr:
        if teleoperation_active:
            if "bodytracking_udp" in teleop_device_name:
                print(
                    "[teleop] Body-tracking UDP starts ACTIVE. "
                    "The input device will lock its initial pose after its calibration countdown."
                )
            else:
                print(
                    "[teleop] XR teleoperation starts ACTIVE because --xr_start_active was set. "
                    "Make sure the hands are aligned before moving."
                )
        else:
            print("[teleop] XR teleoperation is waiting for the headset client's Play/start teleop command.")
    else:
        teleoperation_active = True
        if "bodytracking_udp" in teleop_device_name:
            print(
                "[teleop] Body-tracking UDP runs without IsaacLab XR rendering. "
                "The input device will lock its initial pose after its calibration countdown."
            )

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")
    debug_xr_interval = max(args_cli.debug_xr_interval, 1)
    debug_frame = 0
    waiting_notice_frame = 0
    perf_frame_count = 0
    perf_last_time = time.perf_counter()
    eef_debug_state = None
    if args_cli.debug_xr_data:
        if hasattr(teleop_interface, "set_debug_print"):
            teleop_interface.set_debug_print(True, debug_xr_interval)
            print(f"[teleop debug] Device raw-data debug printing enabled every {debug_xr_interval} frames.")
        else:
            print("[teleop debug] Teleop interface does not support XR raw-data debug printing.")

    # reset environment
    env.reset()
    teleop_interface.reset()
    if args_cli.debug_eef_error:
        eef_debug_state = _initialize_eef_error_debug(env)

    print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()
                if args_cli.debug_xr_data and debug_frame % debug_xr_interval == 0:
                    action_np = action.detach().cpu().numpy()
                    print(
                        "[teleop debug] "
                        f"frame={debug_frame} active={teleoperation_active} "
                        f"action_shape={tuple(action.shape)} "
                        f"action={np.array2string(action_np, precision=4, suppress_small=True)}"
                    )
                if args_cli.debug_perf:
                    perf_frame_count += 1
                    now = time.perf_counter()
                    if perf_frame_count >= debug_xr_interval:
                        elapsed = now - perf_last_time
                        loop_hz = perf_frame_count / elapsed if elapsed > 0 else 0.0
                        step_dt = float(getattr(env, "step_dt", env_cfg.sim.dt * env_cfg.decimation))
                        real_time_factor = loop_hz * step_dt
                        print(
                            f"[teleop perf] loop_hz={loop_hz:.1f} sim_rtf={real_time_factor:.2f} "
                            f"step_dt={step_dt:.4f}s active={teleoperation_active}"
                        )
                        perf_frame_count = 0
                        perf_last_time = now
                debug_frame += 1

                # Only apply teleop commands when active
                if teleoperation_active:
                    # process actions
                    actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                    if (
                        args_cli.debug_eef_error
                        and eef_debug_state is not None
                        and debug_frame % debug_xr_interval == 0
                    ):
                        _print_eef_error_debug(env, eef_debug_state, actions, debug_frame)
                else:
                    if args_cli.xr and waiting_notice_frame % 300 == 0:
                        print("[teleop] XR input is visible, but robot control is paused. Press Play in the headset client.")
                    waiting_notice_frame += 1
                    env.sim.render()

                if should_reset_recording_instance:
                    env.reset()
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
