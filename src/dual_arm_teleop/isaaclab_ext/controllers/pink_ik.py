"""ESROBO-specific Pink IK controller extensions."""

from __future__ import annotations

import numpy as np
import torch
from pink import solve_ik

from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.utils import configclass

from .arm_vector_task import ArmSegmentVectorTask


@configclass
class ESROBOPinkIKControllerCfg(PinkIKControllerCfg):
    """Pink IK configuration with runtime stabilizers used by ESROBO teleoperation."""

    task_error_tolerance: float | None = None
    """If set, skip IK and hold current joints when every task error norm is below this value."""

    enforce_velocity_limits: bool = True
    """If True, let Pink enforce URDF velocity limits during IK."""

    max_joint_position_delta: float | None = None
    """If set, clamp each IK output joint target to current joint position +/- this value in radians."""

    max_command_lead: float | None = None
    """Maximum IK command-state lead over the measured joint position in radians."""

    max_command_step: float | None = None
    """Maximum change of the persistent IK command state per solve in radians."""

    clamp_joint_positions_to_limits: bool = False
    """If True, clamp IK output joint targets to the URDF position limits for the controlled joints."""


class ESROBOPinkIKController(PinkIKController):
    """Pink IK controller with ESROBO teleoperation stabilizers."""

    cfg: ESROBOPinkIKControllerCfg

    def __init__(
        self,
        cfg: ESROBOPinkIKControllerCfg,
        robot_cfg: ArticulationCfg,
        device: str,
        controlled_joint_indices: list[int],
    ):
        super().__init__(
            cfg=cfg,
            robot_cfg=robot_cfg,
            device=device,
            controlled_joint_indices=controlled_joint_indices,
        )
        self._setup_joint_limit_tensors()
        self._commanded_controlled_joint_positions: np.ndarray | None = None

    def _setup_joint_limit_tensors(self) -> None:
        """Cache controlled joint position limits in Isaac Lab controlled-joint order."""
        controlled_joint_indices_pinocchio_order = self.pink_configuration._controlled_joint_indices
        lower_limits_pinocchio_order = self.pink_configuration.full_model.lowerPositionLimit[
            controlled_joint_indices_pinocchio_order
        ]
        upper_limits_pinocchio_order = self.pink_configuration.full_model.upperPositionLimit[
            controlled_joint_indices_pinocchio_order
        ]

        lower_limits_isaac_lab_order = lower_limits_pinocchio_order[self.pink_to_isaac_lab_controlled_ordering]
        upper_limits_isaac_lab_order = upper_limits_pinocchio_order[self.pink_to_isaac_lab_controlled_ordering]
        self._controlled_joint_lower_limits = torch.tensor(
            lower_limits_isaac_lab_order, device=self.device, dtype=torch.float32
        )
        self._controlled_joint_upper_limits = torch.tensor(
            upper_limits_isaac_lab_order, device=self.device, dtype=torch.float32
        )
        self._controlled_joint_lower_limits_numpy = lower_limits_isaac_lab_order
        self._controlled_joint_upper_limits_numpy = upper_limits_isaac_lab_order

    def reset_command_state(self) -> None:
        """Reset the persistent IK command state from measured joints on the next solve."""
        self._commanded_controlled_joint_positions = None

    def hold_command_joints(self, mask: np.ndarray, positions: np.ndarray) -> None:
        """Keep warm-start state aligned with held joints while the other arm moves."""
        if self._commanded_controlled_joint_positions is not None:
            self._commanded_controlled_joint_positions[mask] = positions[mask]

    def _clamp_command_lead(self, measured_joint_positions: np.ndarray) -> None:
        """Keep the persistent command state within a bounded lead of measured joints."""
        if self._commanded_controlled_joint_positions is None:
            return
        max_command_lead = float(self.cfg.max_command_lead or 0.0)
        if max_command_lead > 0.0:
            self._commanded_controlled_joint_positions = np.clip(
                self._commanded_controlled_joint_positions,
                measured_joint_positions - max_command_lead,
                measured_joint_positions + max_command_lead,
            )

    def _task_errors_within_tolerance(self) -> bool:
        """Return whether all configured task errors are inside the optional deadband."""
        if self.cfg.task_error_tolerance is None:
            return False

        tolerance = float(self.cfg.task_error_tolerance)
        for task in self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks:
            error_norm = np.linalg.norm(task.compute_error(self.pink_configuration))
            if error_norm > tolerance:
                return False
        return True

    def compute(self, curr_joint_pos: np.ndarray, dt: float) -> torch.Tensor:
        """Compute target joint positions with ESROBO stabilizing clamps."""
        curr_controlled_joint_pos = [curr_joint_pos[i] for i in self.controlled_joint_indices]
        curr_controlled_joint_pos_tensor = torch.tensor(
            curr_controlled_joint_pos, device=self.device, dtype=torch.float32
        )

        measured_controlled_joint_positions = np.asarray(curr_controlled_joint_pos)
        if self._commanded_controlled_joint_positions is None:
            self._commanded_controlled_joint_positions = measured_controlled_joint_positions.copy()
        else:
            if float(self.cfg.max_command_lead or 0.0) <= 0.0:
                self._commanded_controlled_joint_positions = measured_controlled_joint_positions.copy()
            else:
                self._clamp_command_lead(measured_controlled_joint_positions)

        command_state_full_joint_positions = curr_joint_pos.copy()
        command_state_full_joint_positions[self.controlled_joint_indices] = (
            self._commanded_controlled_joint_positions
        )
        joint_positions_pink = command_state_full_joint_positions[self.isaac_lab_to_pink_ordering]
        self.pink_configuration.update(joint_positions_pink)
        self._update_arm_vector_targets()

        if self._task_errors_within_tolerance():
            target_joint_pos = torch.tensor(
                self._commanded_controlled_joint_positions,
                device=self.device,
                dtype=torch.float32,
            )
            return self._clamp_physical_target(target_joint_pos, curr_controlled_joint_pos_tensor)

        try:
            ik_limits = None if self.cfg.enforce_velocity_limits else []
            velocity = solve_ik(
                self.pink_configuration,
                self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks,
                dt,
                solver="daqp",
                limits=ik_limits,
                safety_break=self.cfg.fail_on_joint_limit_violation,
            )
            joint_angle_changes = velocity * dt
        except (AssertionError, Exception) as exc:
            if self.cfg.show_ik_warnings:
                print(
                    "Warning: IK quadratic solver could not find a solution! Did not update the target joint"
                    f" positions.\nError: {exc}"
                )

            if self.cfg.xr_enabled:
                from isaaclab.ui.xr_widgets import XRVisualization

                XRVisualization.push_event("ik_error", {"error": exc})
            return curr_controlled_joint_pos_tensor

        joint_angle_changes_isaac_lab = joint_angle_changes[self.pink_to_isaac_lab_controlled_ordering]
        if self.cfg.max_command_step is not None:
            max_command_step = max(float(self.cfg.max_command_step), 0.0)
            if max_command_step > 0.0:
                joint_angle_changes_isaac_lab = np.clip(
                    joint_angle_changes_isaac_lab,
                    -max_command_step,
                    max_command_step,
                )
        self._commanded_controlled_joint_positions = (
            self._commanded_controlled_joint_positions + joint_angle_changes_isaac_lab
        )
        self._clamp_command_lead(measured_controlled_joint_positions)
        if self.cfg.clamp_joint_positions_to_limits:
            self._commanded_controlled_joint_positions = np.clip(
                self._commanded_controlled_joint_positions,
                self._controlled_joint_lower_limits_numpy,
                self._controlled_joint_upper_limits_numpy,
            )

        target_joint_pos = torch.tensor(
            self._commanded_controlled_joint_positions,
            device=self.device,
            dtype=torch.float32,
        )

        return self._clamp_physical_target(target_joint_pos, curr_controlled_joint_pos_tensor)

    def _clamp_physical_target(
        self,
        target_joint_pos: torch.Tensor,
        curr_controlled_joint_pos_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Bound the command sent to PhysX while preserving the persistent IK command state."""

        if self.cfg.max_joint_position_delta is not None:
            max_delta = float(self.cfg.max_joint_position_delta)
            target_joint_pos = torch.clamp(
                target_joint_pos,
                curr_controlled_joint_pos_tensor - max_delta,
                curr_controlled_joint_pos_tensor + max_delta,
            )

        if self.cfg.clamp_joint_positions_to_limits:
            target_joint_pos = torch.clamp(
                target_joint_pos,
                min=self._controlled_joint_lower_limits,
                max=self._controlled_joint_upper_limits,
            )

        return target_joint_pos

    def _update_arm_vector_targets(self) -> None:
        """Refresh segment-vector targets from the frame targets set by the action term."""
        frame_positions: dict[str, np.ndarray] = {}
        all_tasks = self.cfg.variable_input_tasks + self.cfg.fixed_input_tasks
        for task in all_tasks:
            frame = getattr(task, "frame", None)
            transform = getattr(task, "transform_target_to_base", None)
            if frame is not None and transform is not None:
                frame_positions[str(frame)] = np.asarray(
                    transform.translation, dtype=np.float64
                ).copy()
        for task in all_tasks:
            if isinstance(task, ArmSegmentVectorTask):
                task.update_target_from_frame_positions(
                    frame_positions, self.pink_configuration
                )
