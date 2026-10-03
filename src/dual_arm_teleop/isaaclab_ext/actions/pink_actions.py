"""ESROBO-specific Pink IK action term."""

from __future__ import annotations

import os

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

from dual_arm_teleop.isaaclab_ext.assets.esrobo import (
    ESROBO_HAND_MIMIC_JOINTS,
    ESROBO_LEFT_ARM_JOINTS,
    ESROBO_RIGHT_ARM_JOINTS,
)
from dual_arm_teleop.isaaclab_ext.controllers import ESROBOPinkIKController
from dual_arm_teleop.pose_stability import PoseTargetStability


class ESROBOPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink IK action that instantiates the project-local ESROBO controller."""

    def _initialize_joint_info(self) -> None:
        """Resolve arm and hand joints in the explicit configuration order used by UDP actions."""
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names,
            preserve_order=True,
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(
            self.cfg.hand_joint_names,
            preserve_order=True,
        )
        self._mimic_joint_names = list(ESROBO_HAND_MIMIC_JOINTS)
        self._mimic_joint_ids, resolved_mimic_names = self._asset.find_joints(
            self._mimic_joint_names,
            preserve_order=True,
        )
        if resolved_mimic_names != self._mimic_joint_names:
            raise RuntimeError(
                "ESROBO mimic joints did not resolve in the configured order: "
                f"expected={self._mimic_joint_names}, resolved={resolved_mimic_names}"
            )
        hand_target_index = {
            name: index for index, name in enumerate(self._hand_joint_names)
        }
        self._mimic_source_hand_indices = [
            hand_target_index[ESROBO_HAND_MIMIC_JOINTS[name][0]]
            for name in self._mimic_joint_names
        ]
        controlled_names = set(self._isaaclab_controlled_joint_names)
        locked_proximal_names = [
            name
            for name in ESROBO_LEFT_ARM_JOINTS + ESROBO_RIGHT_ARM_JOINTS
            if name not in controlled_names
        ]
        if locked_proximal_names:
            self._locked_proximal_joint_ids, self._locked_proximal_joint_names = self._asset.find_joints(
                locked_proximal_names,
                preserve_order=True,
            )
        else:
            self._locked_proximal_joint_ids = []
            self._locked_proximal_joint_names = []
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._mimic_source_hand_indices_tensor = torch.tensor(
            self._mimic_source_hand_indices,
            dtype=torch.long,
            device=self.device,
        )
        self._mimic_multipliers = torch.tensor(
            [ESROBO_HAND_MIMIC_JOINTS[name][1] for name in self._mimic_joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        self._mimic_offsets = torch.tensor(
            [ESROBO_HAND_MIMIC_JOINTS[name][2] for name in self._mimic_joint_names],
            dtype=torch.float32,
            device=self.device,
        )
        print(
            "[esrobo_pink] Explicit hand mimic control active: "
            f"{len(self._mimic_joint_names)} follower joints.",
            flush=True,
        )
        self._debug_hand_mimic = os.environ.get(
            "ESROBO_DEBUG_HAND_MIMIC", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._debug_hand_mimic_interval = max(
            int(os.environ.get("ESROBO_DEBUG_HAND_MIMIC_INTERVAL", "60")),
            1,
        )
        self._debug_hand_mimic_frame = 0
        self._initial_mimic_joint_positions: torch.Tensor | None = None
        self._locked_proximal_joint_positions: torch.Tensor | None = None
        if self._locked_proximal_joint_names:
            print(
                "[esrobo_pink] Wrist-only IK active. "
                f"IK joints={self._isaaclab_controlled_joint_names}; "
                f"locked proximal joints={self._locked_proximal_joint_names}.",
                flush=True,
            )
        self._velocity_feedforward_gain = max(
            float(os.environ.get("ESROBO_ARM_VELOCITY_FEEDFORWARD_GAIN", "0.6")),
            0.0,
        )
        self._velocity_feedforward_max_rad_s = max(
            float(os.environ.get("ESROBO_ARM_VELOCITY_FEEDFORWARD_MAX_RAD_S", "8.0")),
            0.0,
        )
        self._velocity_feedforward_deadband_rad = max(
            float(os.environ.get("ESROBO_ARM_VELOCITY_FEEDFORWARD_DEADBAND_RAD", "0.0015")),
            0.0,
        )
        self._velocity_feedforward_start_speed_m_s = max(
            float(os.environ.get("ESROBO_ARM_VELOCITY_FEEDFORWARD_START_SPEED_M_S", "0.12")),
            0.0,
        )
        self._velocity_feedforward_full_speed_m_s = max(
            float(os.environ.get("ESROBO_ARM_VELOCITY_FEEDFORWARD_FULL_SPEED_M_S", "0.75")),
            self._velocity_feedforward_start_speed_m_s + 1.0e-6,
        )
        frame_names = [
            str(getattr(task, "frame", "")).lower()
            for task in self.cfg.controller.variable_input_tasks
            if getattr(task, "frame", None) is not None
        ]
        self._frame_indices_by_side = {
            side: [index for index, frame_name in enumerate(frame_names) if side in frame_name]
            for side in ("left", "right")
        }
        self._joint_indices_by_side = {
            side: [
                index
                for index, joint_name in enumerate(self._isaaclab_controlled_joint_names)
                if side in joint_name.lower()
            ]
            for side in ("left", "right")
        }
        self._previous_frame_positions = torch.zeros(
            (self.num_envs, self._num_frame_tasks, self.position_dim),
            device=self.device,
        )
        self._previous_frame_targets = torch.zeros(
            (self.num_envs, self._num_frame_tasks, self.pose_dim),
            device=self.device,
        )
        self._has_previous_frame_positions = False
        self._static_target_tolerance = max(
            float(os.environ.get("ESROBO_IK_STATIC_TARGET_TOLERANCE", "0.003")),
            0.0,
        )
        self._static_target_angular_tolerance = max(
            float(os.environ.get("ESROBO_IK_STATIC_TARGET_ANGULAR_TOLERANCE_RAD", "0.0174532925")),
            0.0,
        )
        self._static_target_hold_frames = max(
            int(os.environ.get("ESROBO_IK_STATIC_TARGET_HOLD_FRAMES", "12")),
            1,
        )
        self._static_target_position_error_tolerance_m = max(
            float(os.environ.get("ESROBO_IK_STATIC_TARGET_POSITION_ERROR_M", "0.01")),
            0.0,
        )
        self._static_target_orientation_error_tolerance_rad = max(
            float(os.environ.get("ESROBO_IK_STATIC_TARGET_ORIENTATION_ERROR_RAD", "0.05")),
            0.0,
        )
        self._static_target_hold_active = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self._settled_solution_frame_count = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        self._static_joint_tolerance_rad = max(
            float(os.environ.get("ESROBO_IK_STATIC_JOINT_TOLERANCE_RAD", "0.001")), 0.0,
        )
        self._cached_ik_joint_positions: torch.Tensor | None = None
        frame_names = [
            str(getattr(task, "frame", ""))
            for task in self.cfg.controller.variable_input_tasks
            if getattr(task, "frame", None) is not None
        ]
        self._controlled_frame_body_ids = [self._asset.body_names.index(frame_name) for frame_name in frame_names]
        frame_tasks = [
            task for task in self.cfg.controller.variable_input_tasks
            if getattr(task, "frame", None) is not None
        ]
        self._orientation_frame_indices = [
            index for index, task in enumerate(frame_tasks)
            if task.cost is not None and bool(torch.any(torch.as_tensor(task.cost).reshape(-1)[-3:] > 0))
        ]
        self._pose_stability = PoseTargetStability(
            [self._frame_indices_by_side[side] for side in ("left", "right")],
            self._orientation_frame_indices, self._static_target_tolerance,
            self._static_target_angular_tolerance, self._static_target_hold_frames,
        )
        self._velocity_feedforward_activation = torch.zeros(
            (self.num_envs, len(self._isaaclab_controlled_joint_ids)),
            device=self.device,
        )
        self._last_arm_velocity_targets = torch.zeros(
            (self.num_envs, len(self._isaaclab_controlled_joint_ids)),
            device=self.device,
        )
        self._debug_wrist_only_ik = os.environ.get("ESROBO_DEBUG_WRIST_ONLY_IK", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._wrist_only_debug_interval = max(
            int(os.environ.get("ESROBO_DEBUG_WRIST_ONLY_IK_INTERVAL", "60")),
            1,
        )
        self._wrist_only_debug_frame = 0
        self._initial_wrist_joint_positions: torch.Tensor | None = None

    def _initialize_ik_controllers(self) -> None:
        """Initialize project-local Pink IK controllers for all environments."""
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."

        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            self._ik_controllers.append(
                ESROBOPinkIKController(
                    cfg=self.cfg.controller.copy(),
                    robot_cfg=self._env.scene.cfg.robot,
                    device=self.device,
                    controlled_joint_indices=self._isaaclab_controlled_joint_ids,
                )
            )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Track task-space target speed before forwarding actions to Pink."""
        frame_targets = actions[:, : self._num_frame_tasks * self.pose_dim].reshape(
            self.num_envs,
            self._num_frame_tasks,
            self.pose_dim,
        )
        frame_positions = frame_targets[:, :, : self.position_dim]
        self._velocity_feedforward_activation.zero_()
        self._stable_arm_targets = self._pose_stability.update(frame_targets)
        self._static_target_hold_active &= self._stable_arm_targets

        if self._has_previous_frame_positions:
            target_speed = torch.linalg.norm(
                frame_positions - self._previous_frame_positions,
                dim=-1,
            ) / max(float(self._env.step_dt), 1.0e-6)
            speed_range = (
                self._velocity_feedforward_full_speed_m_s - self._velocity_feedforward_start_speed_m_s
            )
            for side in ("left", "right"):
                frame_indices = self._frame_indices_by_side[side]
                joint_indices = self._joint_indices_by_side[side]
                if not frame_indices or not joint_indices:
                    continue
                side_speed = torch.max(target_speed[:, frame_indices], dim=1).values
                side_activation = torch.clamp(
                    (side_speed - self._velocity_feedforward_start_speed_m_s) / speed_range,
                    0.0,
                    1.0,
                )
                self._velocity_feedforward_activation[:, joint_indices] = side_activation[:, None]

        self._previous_frame_positions.copy_(frame_positions)
        self._previous_frame_targets.copy_(frame_targets)
        self._has_previous_frame_positions = True
        super().process_actions(actions)

    def apply_actions(self) -> None:
        """Apply Pink position targets with bounded velocity feed-forward for fast motion."""
        current_frame_positions = (
            self._asset.data.body_link_state_w[:, self._controlled_frame_body_ids, :3]
            - self._env.scene.env_origins[:, None, :]
        )
        target_frame_positions = self._raw_actions[:, : self._num_frame_tasks * self.pose_dim].reshape(
            self.num_envs,
            self._num_frame_tasks,
            self.pose_dim,
        )[:, :, : self.position_dim]
        target_frame_quaternions = self._raw_actions[:, : self._num_frame_tasks * self.pose_dim].reshape(
            self.num_envs,
            self._num_frame_tasks,
            self.pose_dim,
        )[:, :, self.position_dim :]
        frame_position_errors = torch.linalg.norm(target_frame_positions - current_frame_positions, dim=-1)
        current_frame_quaternions = self._asset.data.body_link_state_w[
            :, self._controlled_frame_body_ids, 3:7
        ]
        current_frame_quaternions = torch.nn.functional.normalize(current_frame_quaternions, dim=-1)
        target_frame_quaternions = torch.nn.functional.normalize(target_frame_quaternions, dim=-1)
        quaternion_dots = torch.abs(
            torch.sum(current_frame_quaternions * target_frame_quaternions, dim=-1)
        )
        frame_orientation_errors = 2.0 * torch.acos(torch.clamp(quaternion_dots, 0.0, 1.0))
        # Elbow orientation has zero IK cost: it must never prevent wrist hold.
        for side_index, side in enumerate(("left", "right")):
            frames = self._frame_indices_by_side[side]
            if not frames:
                continue
            reached = frame_position_errors[:, frames].amax(dim=1) <= self._static_target_position_error_tolerance_m
            orientation_frames = [i for i in frames if i in self._orientation_frame_indices]
            if orientation_frames:
                reached &= frame_orientation_errors[:, orientation_frames].amax(dim=1) <= self._static_target_orientation_error_tolerance_rad
            # Some poses cannot satisfy all arm-vector, position and orientation
            # constraints. A converged solution must also be allowed to stop.
            settled = self._settled_solution_frame_count[:, side_index] >= self._static_target_hold_frames
            self._static_target_hold_active[:, side_index] |= self._stable_arm_targets[:, side_index] & (reached | settled)

        joint_hold_mask = torch.zeros_like(self._last_arm_velocity_targets, dtype=torch.bool)
        for side_index, side in enumerate(("left", "right")):
            joint_hold_mask[:, self._joint_indices_by_side[side]] = self._static_target_hold_active[:, side_index, None]
        if self._cached_ik_joint_positions is None or not bool(torch.all(joint_hold_mask)):
            candidate = self._compute_ik_solutions()
            if self._cached_ik_joint_positions is None:
                self._cached_ik_joint_positions = candidate.detach().clone()
            else:
                for side_index, side in enumerate(("left", "right")):
                    indices = self._joint_indices_by_side[side]
                    if not indices:
                        continue
                    solution_change = torch.abs(candidate[:, indices] - self._cached_ik_joint_positions[:, indices]).amax(dim=1)
                    settled = self._stable_arm_targets[:, side_index] & (solution_change <= self._static_joint_tolerance_rad)
                    self._settled_solution_frame_count[:, side_index] = torch.where(
                        settled, self._settled_solution_frame_count[:, side_index] + 1, 0,
                    )
                self._cached_ik_joint_positions = torch.where(
                    joint_hold_mask, self._cached_ik_joint_positions, candidate.detach(),
                )
        # Preserve the solved command on hold, rather than repeatedly replacing
        # it with measured joints. PhysX can finish tracking that fixed command.
        ik_joint_positions = self._cached_ik_joint_positions.clone()
        for env_index, controller in enumerate(self._ik_controllers):
            if bool(torch.any(joint_hold_mask[env_index])):
                controller.hold_command_joints(
                    joint_hold_mask[env_index].cpu().numpy(),
                    ik_joint_positions[env_index].cpu().numpy(),
                )
        current_joint_positions = self._asset.data.joint_pos[:, self._isaaclab_controlled_joint_ids]
        joint_delta = ik_joint_positions - current_joint_positions

        velocity_targets = joint_delta / max(float(self._sim_dt), 1.0e-6)
        velocity_targets *= self._velocity_feedforward_gain
        velocity_targets *= self._velocity_feedforward_activation
        velocity_targets.masked_fill_(joint_hold_mask, 0.0)
        if self._velocity_feedforward_deadband_rad > 0.0:
            velocity_targets = torch.where(
                torch.abs(joint_delta) >= self._velocity_feedforward_deadband_rad,
                velocity_targets,
                torch.zeros_like(velocity_targets),
            )
        if self._velocity_feedforward_max_rad_s > 0.0:
            velocity_targets = torch.clamp(
                velocity_targets,
                -self._velocity_feedforward_max_rad_s,
                self._velocity_feedforward_max_rad_s,
            )
        else:
            velocity_targets.zero_()
        self._last_arm_velocity_targets = velocity_targets

        self._processed_actions = torch.cat((ik_joint_positions, self._target_hand_joint_positions), dim=1)

        if self.cfg.enable_gravity_compensation:
            self._apply_gravity_compensation()

        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)
        mimic_targets = (
            self._target_hand_joint_positions[:, self._mimic_source_hand_indices_tensor]
            * self._mimic_multipliers
            + self._mimic_offsets
        )
        mimic_limits = self._asset.data.soft_joint_pos_limits[:, self._mimic_joint_ids]
        mimic_targets = torch.maximum(
            torch.minimum(mimic_targets, mimic_limits[..., 1]),
            mimic_limits[..., 0],
        )
        self._asset.set_joint_position_target(mimic_targets, self._mimic_joint_ids)
        if self._debug_hand_mimic:
            measured_mimic = self._asset.data.joint_pos[:, self._mimic_joint_ids]
            if self._initial_mimic_joint_positions is None:
                self._initial_mimic_joint_positions = measured_mimic.detach().clone()
            self._debug_hand_mimic_frame += 1
            if self._debug_hand_mimic_frame % self._debug_hand_mimic_interval == 0:
                measured_active = self._asset.data.joint_pos[
                    :, self._hand_joint_ids
                ][:, self._mimic_source_hand_indices_tensor]
                expected_measured = (
                    measured_active * self._mimic_multipliers + self._mimic_offsets
                )
                expected_measured = torch.maximum(
                    torch.minimum(expected_measured, mimic_limits[..., 1]),
                    mimic_limits[..., 0],
                )
                max_mimic_error = torch.max(
                    torch.abs(measured_mimic - expected_measured)
                ).item()
                max_mimic_motion = torch.max(
                    torch.abs(measured_mimic - self._initial_mimic_joint_positions)
                ).item()
                print(
                    "[esrobo_pink] Hand mimic measured: "
                    f"max_motion={max_mimic_motion:.5f}rad "
                    f"max_relation_error={max_mimic_error:.5f}rad.",
                    flush=True,
                )
        self._asset.set_joint_velocity_target(
            self._last_arm_velocity_targets,
            self._isaaclab_controlled_joint_ids,
        )
        if self._locked_proximal_joint_ids:
            if self._locked_proximal_joint_positions is None:
                self._locked_proximal_joint_positions = self._asset.data.joint_pos[
                    :, self._locked_proximal_joint_ids
                ].detach().clone()
                self._initial_wrist_joint_positions = self._asset.data.joint_pos[
                    :, self._isaaclab_controlled_joint_ids
                ].detach().clone()
            self._asset.set_joint_position_target(
                self._locked_proximal_joint_positions,
                self._locked_proximal_joint_ids,
            )
            self._asset.set_joint_velocity_target(
                torch.zeros_like(self._locked_proximal_joint_positions),
                self._locked_proximal_joint_ids,
            )
            if self._debug_wrist_only_ik:
                self._wrist_only_debug_frame += 1
                if self._wrist_only_debug_frame % self._wrist_only_debug_interval == 0:
                    proximal_drift = torch.max(
                        torch.abs(
                            self._asset.data.joint_pos[:, self._locked_proximal_joint_ids]
                            - self._locked_proximal_joint_positions
                        )
                    ).item()
                    wrist_motion = torch.max(
                        torch.abs(
                            self._asset.data.joint_pos[:, self._isaaclab_controlled_joint_ids]
                            - self._initial_wrist_joint_positions
                        )
                    ).item()
                    print(
                        "[esrobo_pink] Wrist-only measured motion: "
                        f"proximal_max_drift={proximal_drift:.6f}rad "
                        f"wrist_max_delta={wrist_motion:.6f}rad.",
                        flush=True,
                    )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            controller_indices = range(len(self._ik_controllers))
        elif isinstance(env_ids, torch.Tensor):
            controller_indices = env_ids.detach().cpu().tolist()
        else:
            controller_indices = env_ids
        for controller_index in controller_indices:
            self._ik_controllers[int(controller_index)].reset_command_state()
        self._has_previous_frame_positions = False
        self._pose_stability.reset()
        self._settled_solution_frame_count.zero_()
        self._static_target_hold_active.zero_()
        self._cached_ik_joint_positions = None
        self._velocity_feedforward_activation.zero_()
        self._last_arm_velocity_targets.zero_()
        self._locked_proximal_joint_positions = None
        self._initial_wrist_joint_positions = None
        self._wrist_only_debug_frame = 0
