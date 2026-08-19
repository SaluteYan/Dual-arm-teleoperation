# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR retargeter for ESROBO fixed-base bimanual teleoperation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


ESROBO_DEFAULT_HAND_JOINT_LIMITS = {
    "left_thumb_cmc_roll": (0.0, 1.03),
    "left_thumb_cmc_yaw": (0.0, 1.40),
    "left_thumb_cmc_pitch": (0.0, 0.52),
    "left_index_mcp_roll": (0.0, 0.19),
    "left_index_mcp_pitch": (0.0, 1.36),
    "left_middle_mcp_pitch": (0.0, 1.36),
    "left_ring_mcp_roll": (0.0, 0.20),
    "left_ring_mcp_pitch": (0.0, 1.36),
    "left_pinky_mcp_roll": (0.0, 0.30),
    "left_pinky_mcp_pitch": (0.0, 1.36),
    "right_thumb_cmc_roll": (0.0, 1.03),
    "right_thumb_cmc_yaw": (0.0, 1.40),
    "right_thumb_cmc_pitch": (0.0, 0.52),
    "right_index_mcp_roll": (0.0, 0.19),
    "right_index_mcp_pitch": (0.0, 1.36),
    "right_middle_mcp_pitch": (0.0, 1.36),
    "right_ring_mcp_roll": (0.0, 0.20),
    "right_ring_mcp_pitch": (0.0, 1.36),
    "right_pinky_mcp_roll": (0.0, 0.30),
    "right_pinky_mcp_pitch": (0.0, 1.36),
}

OPENXR_HAND_BONES = [
    ("wrist", "palm"),
    ("wrist", "thumb_metacarpal"),
    ("thumb_metacarpal", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("palm", "index_metacarpal"),
    ("index_metacarpal", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index_tip"),
    ("palm", "middle_metacarpal"),
    ("middle_metacarpal", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle_tip"),
    ("palm", "ring_metacarpal"),
    ("ring_metacarpal", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_distal"),
    ("ring_distal", "ring_tip"),
    ("palm", "little_metacarpal"),
    ("little_metacarpal", "little_proximal"),
    ("little_proximal", "little_intermediate"),
    ("little_intermediate", "little_distal"),
    ("little_distal", "little_tip"),
]
"""OpenXR hand skeleton edges used for debug visualization."""


class ESROBOUpperBodyRetargeter(RetargeterBase):
    """Retargets OpenXR hand tracking to ESROBO Pink IK actions.

    Output layout:
        left hand pose 7 + right hand pose 7 + left active hand joints 10 + right active hand joints 10.
    """

    def __init__(self, cfg: ESROBOUpperBodyRetargeterCfg):
        super().__init__(cfg)
        if cfg.hand_joint_names is None:
            raise ValueError("hand_joint_names must be provided for ESROBOUpperBodyRetargeter")

        self._hand_joint_names = cfg.hand_joint_names
        self._joint_limits = dict(ESROBO_DEFAULT_HAND_JOINT_LIMITS)
        self._joint_limits.update(cfg.hand_joint_limits)
        self._position_scale = cfg.position_scale
        self._position_offset = np.array(cfg.position_offset, dtype=np.float32)
        self._left_wrist_rotation_offset = torch.tensor(cfg.left_wrist_rotation_offset, dtype=torch.float32)
        self._right_wrist_rotation_offset = torch.tensor(cfg.right_wrist_rotation_offset, dtype=torch.float32)
        self._ignore_default_wrist_pose = cfg.ignore_default_wrist_pose
        self._calibrate_on_first_valid_wrist_pose = cfg.calibrate_on_first_valid_wrist_pose
        self._calibrated_position_scale = cfg.calibrated_position_scale
        self._print_calibration_events = cfg.print_calibration_events
        self._visualize_head_pose = cfg.visualize_head_pose
        self._visualize_upper_body_links = cfg.visualize_upper_body_links
        self._estimated_shoulder_width = cfg.estimated_shoulder_width
        self._estimated_head_to_shoulder_offset = torch.tensor(
            cfg.estimated_head_to_shoulder_offset, dtype=torch.float32
        )
        self._default_openxr_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._initial_left_pose = np.array(cfg.initial_left_wrist_pose, dtype=np.float32)
        self._initial_right_pose = np.array(cfg.initial_right_wrist_pose, dtype=np.float32)
        self._initial_left_pose_matrix = self._pose_array_to_matrix(self._initial_left_pose)
        self._initial_right_pose_matrix = self._pose_array_to_matrix(self._initial_right_pose)
        self._left_calibration_pose_matrix: torch.Tensor | None = None
        self._right_calibration_pose_matrix: torch.Tensor | None = None
        self._previous_left_pose = self._initial_left_pose.copy()
        self._previous_right_pose = self._initial_right_pose.copy()
        self._previous_hand_joints = np.zeros(len(self._hand_joint_names), dtype=np.float32)

        self._enable_visualization = cfg.enable_visualization
        self._num_open_xr_hand_joints = cfg.num_open_xr_hand_joints
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/esrobo_ar_human_pose",
                markers={
                    "left_joint": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 1.0)),
                    ),
                    "right_joint": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.1)),
                    ),
                    "head_joint": sim_utils.SphereCfg(
                        radius=0.035,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.35)),
                    ),
                    "body_joint": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.9, 1.0)),
                    ),
                    "left_bone": sim_utils.CylinderCfg(
                        radius=0.003,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 1.0), roughness=1.0),
                    ),
                    "right_bone": sim_utils.CylinderCfg(
                        radius=0.003,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.1), roughness=1.0),
                    ),
                    "body_bone": sim_utils.CylinderCfg(
                        radius=0.005,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.9, 1.0), roughness=1.0),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)

    def retarget(self, data: dict) -> torch.Tensor:
        left_hand = data.get(DeviceBase.TrackingTarget.HAND_LEFT, {})
        right_hand = data.get(DeviceBase.TrackingTarget.HAND_RIGHT, {})

        left_pose = self._retarget_wrist_pose(left_hand, is_left=True)
        right_pose = self._retarget_wrist_pose(right_hand, is_left=False)
        hand_joints = self._retarget_hand_joints(left_hand, right_hand)

        self._previous_left_pose = left_pose
        self._previous_right_pose = right_pose
        self._previous_hand_joints = hand_joints

        if self._enable_visualization:
            self._visualize_openxr_human_pose(data)

        command = np.concatenate([left_pose, right_pose, hand_joints]).astype(np.float32)
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        requirements = [RetargeterBase.Requirement.HAND_TRACKING]
        if self._enable_visualization and self._visualize_head_pose:
            requirements.append(RetargeterBase.Requirement.HEAD_TRACKING)
        return requirements

    def reset(self) -> None:
        self._left_calibration_pose_matrix = None
        self._right_calibration_pose_matrix = None
        self._previous_left_pose = self._initial_left_pose.copy()
        self._previous_right_pose = self._initial_right_pose.copy()
        self._previous_hand_joints = np.zeros(len(self._hand_joint_names), dtype=np.float32)
        if self._print_calibration_events:
            print("[ESROBO retargeter] Calibration reset. Waiting for the first valid left/right wrist poses.")

    def _retarget_wrist_pose(self, hand_data: dict[str, np.ndarray], is_left: bool) -> np.ndarray:
        wrist = hand_data.get("wrist")
        if wrist is None or (self._ignore_default_wrist_pose and self._is_default_openxr_pose(wrist)):
            return self._previous_left_pose.copy() if is_left else self._previous_right_pose.copy()

        wrist_pose = self._openxr_wrist_to_robot_wrist_pose(wrist, is_left)
        if self._calibrate_on_first_valid_wrist_pose:
            wrist_pose = self._apply_initial_pose_calibration(wrist_pose, is_left)

        return self._pose_matrix_to_array(wrist_pose)

    def _openxr_wrist_to_robot_wrist_pose(self, wrist: np.ndarray, is_left: bool) -> torch.Tensor:
        wrist_pos = torch.tensor(wrist[:3] * self._position_scale + self._position_offset, dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)
        rotation_offset = self._left_wrist_rotation_offset if is_left else self._right_wrist_rotation_offset

        openxr_pose = math_utils.make_pose(wrist_pos, math_utils.matrix_from_quat(wrist_quat))
        offset_pose = math_utils.make_pose(torch.zeros(3), math_utils.matrix_from_quat(rotation_offset))
        return math_utils.pose_in_A_to_pose_in_B(offset_pose, openxr_pose)

    def _apply_initial_pose_calibration(self, wrist_pose: torch.Tensor, is_left: bool) -> torch.Tensor:
        if is_left:
            calibration_pose = self._left_calibration_pose_matrix
            initial_pose = self._initial_left_pose_matrix
        else:
            calibration_pose = self._right_calibration_pose_matrix
            initial_pose = self._initial_right_pose_matrix

        if calibration_pose is None:
            if is_left:
                self._left_calibration_pose_matrix = wrist_pose.clone()
            else:
                self._right_calibration_pose_matrix = wrist_pose.clone()
            if self._print_calibration_events:
                side = "left" if is_left else "right"
                pos, _ = math_utils.unmake_pose(wrist_pose)
                print(f"[ESROBO retargeter] Calibrated {side} wrist at {pos.numpy().round(4).tolist()}.")
            return initial_pose.clone()

        target_pose = initial_pose.clone()
        target_pose[:3, 3] = initial_pose[:3, 3] + (
            wrist_pose[:3, 3] - calibration_pose[:3, 3]
        ) * self._calibrated_position_scale

        rotation_delta = wrist_pose[:3, :3] @ calibration_pose[:3, :3].transpose(-1, -2)
        target_pose[:3, :3] = rotation_delta @ initial_pose[:3, :3]
        return target_pose

    def _pose_array_to_matrix(self, pose: np.ndarray) -> torch.Tensor:
        position = torch.tensor(pose[:3], dtype=torch.float32)
        quaternion = torch.tensor(pose[3:], dtype=torch.float32)
        return math_utils.make_pose(position, math_utils.matrix_from_quat(quaternion))

    def _pose_matrix_to_array(self, pose: torch.Tensor) -> np.ndarray:
        pos, rot_mat = math_utils.unmake_pose(pose)
        quat = math_utils.quat_from_matrix(rot_mat)
        return np.concatenate([pos.numpy(), quat.numpy()]).astype(np.float32)

    def _is_default_openxr_pose(self, pose: np.ndarray) -> bool:
        return np.allclose(pose, self._default_openxr_pose, atol=1.0e-6)

    def _retarget_hand_joints(
        self, left_hand: dict[str, np.ndarray], right_hand: dict[str, np.ndarray]
    ) -> np.ndarray:
        joint_targets = {}
        joint_targets.update(self._single_hand_joint_targets(left_hand, "left"))
        joint_targets.update(self._single_hand_joint_targets(right_hand, "right"))

        values = np.zeros(len(self._hand_joint_names), dtype=np.float32)
        for index, joint_name in enumerate(self._hand_joint_names):
            if joint_name in joint_targets:
                values[index] = joint_targets[joint_name]
            elif joint_name in self._joint_limits:
                values[index] = self._joint_limits[joint_name][0]
            else:
                values[index] = self._previous_hand_joints[index]
        return values

    def _single_hand_joint_targets(self, hand_data: dict[str, np.ndarray], side: str) -> dict[str, float]:
        if not hand_data:
            return {}

        thumb_curl = self._thumb_curl(hand_data)
        index_curl = self._finger_curl(hand_data, "index")
        middle_curl = self._finger_curl(hand_data, "middle")
        ring_curl = self._finger_curl(hand_data, "ring")
        pinky_curl = self._finger_curl(hand_data, "little")

        return {
            f"{side}_thumb_cmc_roll": self._limit_lerp(f"{side}_thumb_cmc_roll", 0.35 * thumb_curl),
            f"{side}_thumb_cmc_yaw": self._limit_lerp(f"{side}_thumb_cmc_yaw", 0.50 * thumb_curl),
            f"{side}_thumb_cmc_pitch": self._limit_lerp(f"{side}_thumb_cmc_pitch", thumb_curl),
            f"{side}_index_mcp_roll": self._limit_lerp(f"{side}_index_mcp_roll", 0.0),
            f"{side}_index_mcp_pitch": self._limit_lerp(f"{side}_index_mcp_pitch", index_curl),
            f"{side}_middle_mcp_pitch": self._limit_lerp(f"{side}_middle_mcp_pitch", middle_curl),
            f"{side}_ring_mcp_roll": self._limit_lerp(f"{side}_ring_mcp_roll", 0.0),
            f"{side}_ring_mcp_pitch": self._limit_lerp(f"{side}_ring_mcp_pitch", ring_curl),
            f"{side}_pinky_mcp_roll": self._limit_lerp(f"{side}_pinky_mcp_roll", 0.0),
            f"{side}_pinky_mcp_pitch": self._limit_lerp(f"{side}_pinky_mcp_pitch", pinky_curl),
        }

    def _finger_curl(self, hand_data: dict[str, np.ndarray], finger: str) -> float:
        metacarpal = hand_data.get(f"{finger}_metacarpal")
        proximal = hand_data.get(f"{finger}_proximal")
        intermediate = hand_data.get(f"{finger}_intermediate")
        distal = hand_data.get(f"{finger}_distal")
        tip = hand_data.get(f"{finger}_tip")

        curl_a = self._joint_curl(metacarpal, proximal, intermediate)
        curl_b = self._joint_curl(proximal, intermediate, distal)
        curl_c = self._joint_curl(intermediate, distal, tip)
        return float(np.clip(0.55 * curl_a + 0.30 * curl_b + 0.15 * curl_c, 0.0, 1.0))

    def _thumb_curl(self, hand_data: dict[str, np.ndarray]) -> float:
        metacarpal = hand_data.get("thumb_metacarpal")
        proximal = hand_data.get("thumb_proximal")
        distal = hand_data.get("thumb_distal")
        tip = hand_data.get("thumb_tip")

        curl_a = self._joint_curl(metacarpal, proximal, distal)
        curl_b = self._joint_curl(proximal, distal, tip)
        return float(np.clip(0.65 * curl_a + 0.35 * curl_b, 0.0, 1.0))

    def _joint_curl(self, first: np.ndarray | None, middle: np.ndarray | None, last: np.ndarray | None) -> float:
        if first is None or middle is None or last is None:
            return 0.0

        first_vec = first[:3] - middle[:3]
        last_vec = last[:3] - middle[:3]
        first_norm = np.linalg.norm(first_vec)
        last_norm = np.linalg.norm(last_vec)
        if first_norm < 1e-6 or last_norm < 1e-6:
            return 0.0

        cos_angle = np.clip(np.dot(first_vec, last_vec) / (first_norm * last_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return float(np.clip((np.pi - angle) / (0.5 * np.pi), 0.0, 1.0))

    def _limit_lerp(self, joint_name: str, normalized_value: float) -> float:
        lower, upper = self._joint_limits[joint_name]
        alpha = float(np.clip(normalized_value, 0.0, 1.0))
        return float(lower + alpha * (upper - lower))

    def _visualize_openxr_human_pose(self, data: dict) -> None:
        left_hand = data.get(DeviceBase.TrackingTarget.HAND_LEFT, {})
        right_hand = data.get(DeviceBase.TrackingTarget.HAND_RIGHT, {})
        head_pose = data.get(DeviceBase.TrackingTarget.HEAD)

        translations = []
        orientations = []
        scales = []
        marker_indices = []
        line_start_positions = []
        line_end_positions = []
        line_marker_indices = []

        self._append_hand_visualization(
            left_hand,
            joint_marker_index=0,
            bone_marker_index=4,
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
            line_start_positions=line_start_positions,
            line_end_positions=line_end_positions,
            line_marker_indices=line_marker_indices,
        )
        self._append_hand_visualization(
            right_hand,
            joint_marker_index=1,
            bone_marker_index=5,
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=marker_indices,
            line_start_positions=line_start_positions,
            line_end_positions=line_end_positions,
            line_marker_indices=line_marker_indices,
        )
        if self._visualize_head_pose and head_pose is not None and not self._is_default_openxr_pose(head_pose):
            head_position = torch.tensor(head_pose[:3], device=self._sim_device, dtype=torch.float32)
            self._append_point(head_position, 2, translations, orientations, scales, marker_indices)
            if self._visualize_upper_body_links:
                self._append_upper_body_visualization(
                    head_position,
                    left_hand.get("wrist"),
                    right_hand.get("wrist"),
                    translations,
                    orientations,
                    scales,
                    marker_indices,
                    line_start_positions,
                    line_end_positions,
                    line_marker_indices,
                )

        if line_start_positions:
            lines_pos, lines_quat, lines_length = self._get_connecting_lines(
                torch.stack(line_start_positions), torch.stack(line_end_positions)
            )
            for line_index in range(lines_pos.shape[0]):
                translations.append(lines_pos[line_index])
                orientations.append(lines_quat[line_index])
                line_scale = torch.ones(3, device=self._sim_device)
                line_scale[-1] = lines_length[line_index]
                scales.append(line_scale)
                marker_indices.append(line_marker_indices[line_index])

        if not translations:
            return

        self._markers.visualize(
            translations=torch.stack(translations),
            orientations=torch.stack(orientations),
            scales=torch.stack(scales),
            marker_indices=torch.tensor(marker_indices, device=self._sim_device, dtype=torch.int64),
        )

    def _append_hand_visualization(
        self,
        hand_data: dict[str, np.ndarray],
        joint_marker_index: int,
        bone_marker_index: int,
        translations: list[torch.Tensor],
        orientations: list[torch.Tensor],
        scales: list[torch.Tensor],
        marker_indices: list[int],
        line_start_positions: list[torch.Tensor],
        line_end_positions: list[torch.Tensor],
        line_marker_indices: list[int],
    ) -> None:
        valid_positions = {}
        for joint_name in HAND_JOINT_NAMES:
            pose = hand_data.get(joint_name)
            if pose is None or self._is_default_openxr_pose(pose):
                continue
            position = torch.tensor(pose[:3], device=self._sim_device, dtype=torch.float32)
            valid_positions[joint_name] = position
            self._append_point(position, joint_marker_index, translations, orientations, scales, marker_indices)

        for start_name, end_name in OPENXR_HAND_BONES:
            if start_name not in valid_positions or end_name not in valid_positions:
                continue
            line_start_positions.append(valid_positions[start_name])
            line_end_positions.append(valid_positions[end_name])
            line_marker_indices.append(bone_marker_index)

    def _append_upper_body_visualization(
        self,
        head_position: torch.Tensor,
        left_wrist_pose: np.ndarray | None,
        right_wrist_pose: np.ndarray | None,
        translations: list[torch.Tensor],
        orientations: list[torch.Tensor],
        scales: list[torch.Tensor],
        marker_indices: list[int],
        line_start_positions: list[torch.Tensor],
        line_end_positions: list[torch.Tensor],
        line_marker_indices: list[int],
    ) -> None:
        shoulder_center = head_position + self._estimated_head_to_shoulder_offset.to(device=self._sim_device)
        left_shoulder = shoulder_center + torch.tensor(
            [0.0, self._estimated_shoulder_width * 0.5, 0.0], device=self._sim_device
        )
        right_shoulder = shoulder_center + torch.tensor(
            [0.0, -self._estimated_shoulder_width * 0.5, 0.0], device=self._sim_device
        )
        for point in (left_shoulder, right_shoulder):
            self._append_point(point, 3, translations, orientations, scales, marker_indices)

        line_start_positions.append(left_shoulder)
        line_end_positions.append(right_shoulder)
        line_marker_indices.append(6)
        line_start_positions.append(head_position)
        line_end_positions.append(shoulder_center)
        line_marker_indices.append(6)

        for wrist_pose, shoulder in ((left_wrist_pose, left_shoulder), (right_wrist_pose, right_shoulder)):
            if wrist_pose is None or self._is_default_openxr_pose(wrist_pose):
                continue
            wrist_position = torch.tensor(wrist_pose[:3], device=self._sim_device, dtype=torch.float32)
            elbow = 0.5 * (shoulder + wrist_position)
            elbow = elbow + torch.tensor([0.0, 0.0, -0.08], device=self._sim_device)
            self._append_point(elbow, 3, translations, orientations, scales, marker_indices)
            line_start_positions.extend((shoulder, elbow))
            line_end_positions.extend((elbow, wrist_position))
            line_marker_indices.extend((6, 6))

    def _append_point(
        self,
        position: torch.Tensor,
        marker_index: int,
        translations: list[torch.Tensor],
        orientations: list[torch.Tensor],
        scales: list[torch.Tensor],
        marker_indices: list[int],
    ) -> None:
        translations.append(position)
        orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._sim_device))
        scales.append(torch.ones(3, device=self._sim_device))
        marker_indices.append(marker_index)

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self._sim_device).expand(start_pos.size(0), -1)
        direction_norm = math_utils.normalize(direction)
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            math_utils.normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self._sim_device).expand(start_pos.size(0), -1),
        )
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        orientations = math_utils.quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths


@dataclass
class ESROBOUpperBodyRetargeterCfg(RetargeterCfg):
    """Configuration for ESROBO OpenXR upper-body retargeting."""

    enable_visualization: bool = False
    num_open_xr_hand_joints: int = 2 * len(HAND_JOINT_NAMES)
    hand_joint_names: list[str] | None = None
    hand_joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    position_scale: float = 1.0
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_left_wrist_pose: tuple[float, float, float, float, float, float, float] = (
        -0.14664,
        0.24538,
        0.48300,
        0.0,
        0.70711,
        -0.70710,
        0.0,
    )
    initial_right_wrist_pose: tuple[float, float, float, float, float, float, float] = (
        -0.14680,
        -0.23480,
        0.48300,
        0.0,
        -0.70710,
        -0.70711,
        0.0,
    )
    ignore_default_wrist_pose: bool = True
    calibrate_on_first_valid_wrist_pose: bool = True
    calibrated_position_scale: float = 1.0
    print_calibration_events: bool = True
    visualize_head_pose: bool = True
    visualize_upper_body_links: bool = True
    estimated_shoulder_width: float = 0.36
    estimated_head_to_shoulder_offset: tuple[float, float, float] = (0.0, 0.0, -0.28)
    left_wrist_rotation_offset: tuple[float, float, float, float] = (0.7071, 0.0, 0.7071, 0.0)
    right_wrist_rotation_offset: tuple[float, float, float, float] = (0.0, -0.7071, 0.0, 0.7071)
    retargeter_type: type[RetargeterBase] = ESROBOUpperBodyRetargeter


__all__ = [
    "ESROBO_DEFAULT_HAND_JOINT_LIMITS",
    "ESROBOUpperBodyRetargeter",
    "ESROBOUpperBodyRetargeterCfg",
]
