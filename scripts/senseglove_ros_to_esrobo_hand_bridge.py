#!/usr/bin/env python3
"""Bridge official SenseGlove ROS 2 state topics to ESROBO IsaacLab hand joints."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FLEX_SUFFIXES = ("mcp", "pip", "dip")
DEFAULT_LEFT_SERIAL = "01001"
DEFAULT_RIGHT_SERIAL = "01002"
DEFAULT_CALIBRATION_PATH = Path("config/senseglove_esrobo_calibration.json")
IMU_ROBOT_LOCAL_AXIS_SIGNS = {
    # Axis order: lateral bend, vertical bend, palm roll.
    # Apply the verified robot-local direction signs after each glove's device
    # axes have been converted to the shared semantic order below.
    "left": np.asarray([1.0, -1.0, 1.0], dtype=np.float64),
    "right": np.asarray([-1.0, -1.0, 1.0], dtype=np.float64),
}
IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS = {
    # Measured left Nova 2 source axes: Y=lateral bend, Z=vertical bend,
    # X=palm roll. Convert them to the shared semantic X/Y/Z order.
    "left": np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    ),
    "right": np.eye(3, dtype=np.float64),
}
DEFAULT_ESROBO_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "urdf"
    / "esrobo_waist_with_head"
    / "urdf"
    / "esrobo_waist_with_head.urdf"
)

ESROBO_HAND_JOINT_ORDER = (
    "left_thumb_cmc_roll",
    "left_thumb_cmc_yaw",
    "left_thumb_cmc_pitch",
    "left_index_mcp_roll",
    "left_index_mcp_pitch",
    "left_middle_mcp_pitch",
    "left_ring_mcp_roll",
    "left_ring_mcp_pitch",
    "left_pinky_mcp_roll",
    "left_pinky_mcp_pitch",
    "right_thumb_cmc_roll",
    "right_thumb_cmc_yaw",
    "right_thumb_cmc_pitch",
    "right_index_mcp_roll",
    "right_index_mcp_pitch",
    "right_middle_mcp_pitch",
    "right_ring_mcp_roll",
    "right_ring_mcp_pitch",
    "right_pinky_mcp_roll",
    "right_pinky_mcp_pitch",
)

ESROBO_HAND_JOINT_LIMITS = {
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


@dataclass
class SideState:
    topic: str
    msg: Any | None = None
    stamp_monotonic: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    imu_quat_wxyz: list[float] | None = None
    imu_raw_norm: float | None = None
    imu_w_repaired: bool = False
    imu_repaired_raw_xyzw: list[float] | None = None
    hand_positions_mm: list[list[float]] = field(default_factory=list)
    callback_sequence: int = 0


@dataclass(frozen=True)
class UrdfJointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit: tuple[float, float]
    mimic: tuple[str, float, float] | None


@dataclass(frozen=True)
class HandVectorSpec:
    origin_label: str
    task_label: str
    human_origin_index: int
    human_task_index: int
    weight: float
    preserve_relative_length: bool = False


class ESROBOHandKinematics:
    """Small URDF FK helper for one ESROBO hand subtree."""

    def __init__(self, urdf_path: Path, side: str, active_joint_names: list[str]):
        self.side = side
        self.root_link = f"{side}Hand_link"
        self.active_joint_names = list(active_joint_names)
        self.joints_by_parent = self._load_joints(urdf_path)
        self.joints_by_name = {
            joint.name: joint
            for joints in self.joints_by_parent.values()
            for joint in joints
        }
        missing = [
            name for name in self.active_joint_names if name not in self.joints_by_name
        ]
        if missing:
            raise ValueError(
                f"active ESROBO hand joints missing from {urdf_path}: {missing}"
            )
        self.joint_limits = np.asarray(
            [self.joints_by_name[name].limit for name in self.active_joint_names],
            dtype=np.float64,
        )

    @staticmethod
    def _load_joints(urdf_path: Path) -> dict[str, list[UrdfJointInfo]]:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joints_by_parent: dict[str, list[UrdfJointInfo]] = {}
        for joint_xml in root.findall("joint"):
            parent_xml = joint_xml.find("parent")
            child_xml = joint_xml.find("child")
            if parent_xml is None or child_xml is None:
                continue
            name = str(joint_xml.attrib.get("name", ""))
            joint_type = str(joint_xml.attrib.get("type", "fixed"))
            parent = str(parent_xml.attrib["link"])
            child = str(child_xml.attrib["link"])
            origin_xml = joint_xml.find("origin")
            axis_xml = joint_xml.find("axis")
            limit_xml = joint_xml.find("limit")
            mimic_xml = joint_xml.find("mimic")
            origin_xyz = _parse_vec3_attr(origin_xml, "xyz", (0.0, 0.0, 0.0))
            origin_rpy = _parse_vec3_attr(origin_xml, "rpy", (0.0, 0.0, 0.0))
            axis = _parse_vec3_attr(axis_xml, "xyz", (0.0, 0.0, 1.0))
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1.0e-9:
                axis = axis / axis_norm
            if limit_xml is not None:
                lower = float(limit_xml.attrib.get("lower", "-1000000"))
                upper = float(limit_xml.attrib.get("upper", "1000000"))
            else:
                lower, upper = -1000000.0, 1000000.0
            mimic = None
            if mimic_xml is not None:
                mimic = (
                    str(mimic_xml.attrib["joint"]),
                    float(mimic_xml.attrib.get("multiplier", "1.0")),
                    float(mimic_xml.attrib.get("offset", "0.0")),
                )
            joint = UrdfJointInfo(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limit=(lower, upper),
                mimic=mimic,
            )
            joints_by_parent.setdefault(parent, []).append(joint)
        return joints_by_parent

    def forward_points(self, active_qpos: np.ndarray) -> dict[str, np.ndarray]:
        active_values = {
            name: float(active_qpos[index])
            for index, name in enumerate(self.active_joint_names)
        }
        link_poses: dict[str, np.ndarray] = {self.root_link: np.eye(4)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            parent_pose = link_poses[parent]
            for joint in self.joints_by_parent.get(parent, []):
                if not joint.name.startswith(f"{self.side}_"):
                    continue
                value = self._joint_value(joint, active_values)
                child_pose = parent_pose @ _origin_matrix(
                    joint.origin_xyz, joint.origin_rpy
                )
                if joint.joint_type in ("revolute", "continuous"):
                    child_pose = child_pose @ _axis_angle_matrix(joint.axis, value)
                link_poses[joint.child] = child_pose
                stack.append(joint.child)

        points = {
            "wrist": link_poses[self.root_link][:3, 3].copy(),
        }
        for name, pose in link_poses.items():
            if name.startswith(f"{self.side}_"):
                points[name] = pose[:3, 3].copy()

        for finger, length in _robot_tip_lengths().items():
            distal = f"{self.side}_{finger}_distal"
            if distal in link_poses:
                points[f"{self.side}_{finger}_tip"] = (
                    link_poses[distal] @ np.asarray([0.0, 0.0, length, 1.0])
                )[:3]
        return points

    @staticmethod
    def _joint_value(joint: UrdfJointInfo, active_values: dict[str, float]) -> float:
        if joint.name in active_values:
            return active_values[joint.name]
        if joint.mimic is not None:
            source, multiplier, offset = joint.mimic
            return active_values.get(source, 0.0) * multiplier + offset
        return 0.0


class ESROBOHandDexVectorRetargeter:
    """Dex-retargeting-style vector optimizer using numpy-only damped least squares."""

    def __init__(
        self,
        urdf_path: Path,
        side: str,
        open_hand_positions_mm: list[list[float]],
        args: argparse.Namespace,
        curl_calibration: dict[str, dict[str, float]] | None = None,
    ):
        self.side = side
        self.active_joint_names = [
            name for name in ESROBO_HAND_JOINT_ORDER if name.startswith(f"{side}_")
        ]
        self.kinematics = ESROBOHandKinematics(
            urdf_path, side, self.active_joint_names
        )
        self.lower = self.kinematics.joint_limits[:, 0]
        self.upper = self.kinematics.joint_limits[:, 1]
        self.qpos = np.zeros(len(self.active_joint_names), dtype=np.float64)
        self.iterations = max(int(args.dex_iterations), 1)
        self.damping = max(float(args.dex_damping), 1.0e-6)
        self.max_joint_step = max(float(args.dex_max_joint_step), 1.0e-4)
        self.normal_delta = max(float(args.dex_normal_delta), 0.0)
        self.vector_eps = max(float(args.dex_jacobian_eps), 1.0e-6)
        self.direction_weight = max(float(args.dex_direction_weight), 0.0)
        self.huber_delta = max(float(args.dex_huber_delta), 1.0e-6)
        self.joint_limit_margin = max(float(args.dex_joint_limit_margin), 0.0)
        self.joint_limit_weight = max(float(args.dex_joint_limit_weight), 0.0)
        self.curl_weight = max(float(getattr(args, "dex_curl_weight", 0.015)), 0.0)
        self.curl_calibration = curl_calibration or {}
        self.vector_set = getattr(args, "dex_vector_set", "fingertip")
        self.specs = self._build_specs(side, self.vector_set)
        self.robot_reference_vectors = self._robot_vectors(self.qpos)
        self.reference_lengths = np.linalg.norm(
            self.robot_reference_vectors, axis=1
        ).clip(min=1.0e-5)
        open_points = senseglove_points_with_wrist_m(open_hand_positions_mm, side)
        self.human_reference_lengths = self._human_vector_lengths(open_points)
        self.human_to_robot_rotation = _fit_rotation(
            self._human_unit_vectors(open_points), self._robot_unit_vectors()
        )

    @staticmethod
    def _build_specs(side: str, vector_set: str = "fingertip") -> list[HandVectorSpec]:
        def label(name: str) -> str:
            return f"{side}_{name}"

        finger_indices = {
            "thumb": (1, 2, 3, 4),
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }
        robot_points = {
            "thumb": (
                label("thumb_metacarpals_base1"),
                label("thumb_metacarpals"),
                label("thumb_proximal"),
                label("thumb_tip"),
            ),
            "index": (
                label("index_metacarpals"),
                label("index_proximal"),
                label("index_middle"),
                label("index_tip"),
            ),
            "middle": (
                label("middle_proximal"),
                label("middle_middle"),
                label("middle_distal"),
                label("middle_tip"),
            ),
            "ring": (
                label("ring_metacarpals"),
                label("ring_proximal"),
                label("ring_middle"),
                label("ring_tip"),
            ),
            "pinky": (
                label("pinky_metacarpals"),
                label("pinky_proximal"),
                label("pinky_middle"),
                label("pinky_tip"),
            ),
        }
        # dex-retargeting's standard teleoperation configs optimize vectors from
        # the wrist to each fingertip. This is the appropriate objective for an
        # underactuated hand whose PIP/DIP joints mimic one MCP flexion joint.
        # Matching every human phalanx over-constrains that coupling and causes
        # the least-squares solution to settle at roughly half the desired curl.
        if vector_set == "fingertip":
            return [
                HandVectorSpec("wrist", robot_points[finger][-1], 0, finger_indices[finger][-1], 1.0)
                for finger in FINGERS
            ]
        if vector_set != "dense":
            raise ValueError(f"unsupported dex vector set: {vector_set}")

        specs: list[HandVectorSpec] = []
        for finger in FINGERS:
            h = finger_indices[finger]
            r = robot_points[finger]
            specs.append(HandVectorSpec("wrist", r[-1], 0, h[-1], 1.8))
            specs.append(HandVectorSpec("wrist", r[0], 0, h[0], 0.7))
            specs.append(HandVectorSpec(r[0], r[1], h[0], h[1], 1.0))
            specs.append(HandVectorSpec(r[1], r[2], h[1], h[2], 1.0))
            specs.append(HandVectorSpec(r[2], r[3], h[2], h[3], 1.0))
        lateral_pairs = (
            ("index", "middle"),
            ("middle", "ring"),
            ("ring", "pinky"),
            ("thumb", "index"),
        )
        for first, second in lateral_pairs:
            first_h = finger_indices[first]
            second_h = finger_indices[second]
            first_r = robot_points[first]
            second_r = robot_points[second]
            specs.append(HandVectorSpec(first_r[0], second_r[0], first_h[0], second_h[0], 0.45))
            specs.append(HandVectorSpec(first_r[-1], second_r[-1], first_h[-1], second_h[-1], 0.55, True))
        for finger in ("middle", "ring", "pinky"):
            h = finger_indices[finger]
            r = robot_points[finger]
            specs.append(
                HandVectorSpec(
                    robot_points["thumb"][-1], r[-1],
                    finger_indices["thumb"][-1], h[-1], 0.70, True,
                )
            )
        return specs

    def retarget(
        self, hand_positions_mm: list[list[float]]
    ) -> tuple[list[float], dict[str, float]]:
        human_points = senseglove_points_with_wrist_m(hand_positions_mm, self.side)
        target_vectors = self._target_vectors(human_points)
        curl_targets = self._curl_targets(hand_positions_mm)
        previous_qpos = self.qpos.copy()
        qpos = previous_qpos.copy()
        for _ in range(self.iterations):
            residual = self._weighted_residual(qpos, target_vectors)
            jacobian = self._numeric_jacobian(qpos, target_vectors, residual)
            lhs = jacobian.T @ jacobian + (self.damping * self.damping) * np.eye(
                qpos.size
            )
            rhs = -jacobian.T @ residual
            if self.normal_delta > 0.0:
                lhs += self.normal_delta * np.eye(qpos.size)
                rhs += -self.normal_delta * (qpos - previous_qpos)
            self._add_soft_joint_limit_objective(lhs, rhs, qpos)
            self._add_curl_objective(lhs, rhs, qpos, curl_targets)
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
            qpos = np.clip(qpos + delta, self.lower, self.upper)
        self.qpos = qpos
        diagnostics = self._diagnostics(target_vectors)
        return qpos.astype(float).tolist(), diagnostics

    def _curl_targets(
        self, hand_positions_mm: list[list[float]]
    ) -> dict[int, float]:
        if self.curl_weight <= 0.0:
            return {}
        open_features = self.curl_calibration.get("open")
        closed_features = self.curl_calibration.get("closed")
        if not isinstance(open_features, dict) or not isinstance(closed_features, dict):
            return {}
        current = extract_hand_vector_features(hand_positions_mm)
        targets: dict[int, float] = {}
        for finger in FINGERS:
            feature = f"vector_{finger}_curl"
            if feature not in current or feature not in open_features or feature not in closed_features:
                continue
            span = float(closed_features[feature]) - float(open_features[feature])
            if abs(span) < 1.0e-5:
                continue
            progress = np.clip(
                (float(current[feature]) - float(open_features[feature])) / span,
                0.0,
                1.0,
            )
            suffix = "thumb_cmc_pitch" if finger == "thumb" else f"{finger}_mcp_pitch"
            joint_name = f"{self.side}_{suffix}"
            joint_index = self.active_joint_names.index(joint_name)
            targets[joint_index] = float(
                self.lower[joint_index]
                + progress * (self.upper[joint_index] - self.lower[joint_index])
            )
        return targets

    def _add_curl_objective(
        self,
        lhs: np.ndarray,
        rhs: np.ndarray,
        qpos: np.ndarray,
        curl_targets: dict[int, float],
    ) -> None:
        for joint_index, target in curl_targets.items():
            lhs[joint_index, joint_index] += self.curl_weight
            rhs[joint_index] += self.curl_weight * (target - qpos[joint_index])

    def _robot_vectors(self, qpos: np.ndarray) -> np.ndarray:
        points = self.kinematics.forward_points(qpos)
        vectors = [
            points[spec.task_label] - points[spec.origin_label]
            for spec in self.specs
        ]
        return np.asarray(vectors, dtype=np.float64)

    def _human_unit_vectors(self, points: np.ndarray) -> np.ndarray:
        vectors = []
        for spec in self.specs:
            vector = points[spec.human_task_index] - points[spec.human_origin_index]
            vectors.append(_unit_vector_np(vector))
        return np.asarray(vectors, dtype=np.float64)

    def _human_vector_lengths(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                np.linalg.norm(
                    points[spec.human_task_index] - points[spec.human_origin_index]
                )
                for spec in self.specs
            ],
            dtype=np.float64,
        ).clip(min=1.0e-5)

    def _robot_unit_vectors(self) -> np.ndarray:
        return np.asarray(
            [_unit_vector_np(vector) for vector in self.robot_reference_vectors],
            dtype=np.float64,
        )

    def _target_vectors(self, points: np.ndarray) -> np.ndarray:
        vectors = self._human_unit_vectors(points)
        vectors = (self.human_to_robot_rotation @ vectors.T).T
        current_lengths = self._human_vector_lengths(points)
        length_scales = np.ones(len(self.specs), dtype=np.float64)
        for index, spec in enumerate(self.specs):
            if spec.preserve_relative_length:
                length_scales[index] = np.clip(
                    current_lengths[index] / self.human_reference_lengths[index],
                    0.05,
                    1.5,
                )
        return vectors * (self.reference_lengths * length_scales)[:, None]

    def _weights(self) -> np.ndarray:
        return np.asarray([spec.weight for spec in self.specs], dtype=np.float64)

    def _weighted_residual(
        self, qpos: np.ndarray, target_vectors: np.ndarray
    ) -> np.ndarray:
        robot_vectors = self._robot_vectors(qpos)
        position_error = robot_vectors - target_vectors
        target_units = np.asarray(
            [_unit_vector_np(vector) for vector in target_vectors], dtype=np.float64
        )
        robot_units = np.asarray(
            [_unit_vector_np(vector) for vector in robot_vectors], dtype=np.float64
        )
        direction_error = (
            robot_units - target_units
        ) * self.reference_lengths[:, None]
        vector_error_norm = np.linalg.norm(position_error, axis=1)
        robust_scale = np.ones_like(vector_error_norm)
        outside = vector_error_norm > self.huber_delta
        robust_scale[outside] = np.sqrt(
            self.huber_delta / vector_error_norm[outside]
        )
        weights = (self._weights() * robust_scale)[:, None]
        return np.concatenate(
            [
                (position_error * weights).reshape(-1),
                (direction_error * weights * self.direction_weight).reshape(-1),
            ]
        )

    def _numeric_jacobian(
        self, qpos: np.ndarray, target_vectors: np.ndarray, base_residual: np.ndarray
    ) -> np.ndarray:
        rows = base_residual.size
        jacobian = np.zeros((rows, qpos.size), dtype=np.float64)
        for joint_index in range(qpos.size):
            stepped = qpos.copy()
            stepped[joint_index] = min(
                max(stepped[joint_index] + self.vector_eps, self.lower[joint_index]),
                self.upper[joint_index],
            )
            actual_eps = stepped[joint_index] - qpos[joint_index]
            if abs(actual_eps) < 1.0e-9:
                continue
            stepped_residual = self._weighted_residual(stepped, target_vectors)
            jacobian[:, joint_index] = (stepped_residual - base_residual) / actual_eps
        return jacobian

    def _add_soft_joint_limit_objective(
        self, lhs: np.ndarray, rhs: np.ndarray, qpos: np.ndarray
    ) -> None:
        if self.joint_limit_weight <= 0.0 or self.joint_limit_margin <= 0.0:
            return
        ranges = np.maximum(self.upper - self.lower, 1.0e-6)
        margin = np.minimum(self.joint_limit_margin, 0.45) * ranges
        lower_soft = self.lower + margin
        upper_soft = self.upper - margin
        targets = np.clip(qpos, lower_soft, upper_soft)
        active = (qpos < lower_soft) | (qpos > upper_soft)
        for index in np.flatnonzero(active):
            lhs[index, index] += self.joint_limit_weight
            rhs[index] += self.joint_limit_weight * (targets[index] - qpos[index])

    def _diagnostics(self, target_vectors: np.ndarray) -> dict[str, float]:
        robot_vectors = self._robot_vectors(self.qpos)
        errors = np.linalg.norm(robot_vectors - target_vectors, axis=1)
        result = {
            "dex_vector_mean_error_m": float(np.mean(errors)),
            "dex_vector_max_error_m": float(np.max(errors)),
        }
        for finger in FINGERS:
            names = [
                index
                for index, spec in enumerate(self.specs)
                if f"_{finger}_" in spec.task_label or f"_{finger}_" in spec.origin_label
            ]
            if names:
                result[f"{finger}_dex_error_m"] = float(np.mean(errors[names]))
        return result


class SenseGloveUdpBridge:
    """ROS subscriber state and SenseGlove-to-ESROBO hand retargeting."""

    def __init__(
        self, node: Any, msg_type: Any, args: argparse.Namespace, qos_profile: Any
    ):
        self.args = args
        self.left = SideState(args.left_topic)
        self.right = SideState(args.right_topic)
        self._sent_count = 0
        self._last_targets = [0.0] * len(ESROBO_HAND_JOINT_ORDER)
        self._dex_retargeters: dict[str, ESROBOHandDexVectorRetargeter] | None = None

        node.create_subscription(
            msg_type, args.left_topic, self._make_callback(self.left), qos_profile
        )
        node.create_subscription(
            msg_type, args.right_topic, self._make_callback(self.right), qos_profile
        )

    def _make_callback(self, state: SideState):
        def callback(msg: Any) -> None:
            state.msg = msg
            state.stamp_monotonic = time.monotonic()
            state.features = extract_features(msg)
            imu_diagnostics: dict[str, Any] = {}
            state.imu_quat_wxyz = extract_imu_quat_wxyz(
                msg,
                self.args.imu_correction,
                self.args.imu_w_repair,
                imu_diagnostics,
                previous_raw_xyzw=state.imu_repaired_raw_xyzw,
            )
            state.imu_raw_norm = imu_diagnostics.get("raw_norm")
            state.imu_w_repaired = bool(imu_diagnostics.get("w_repaired", False))
            state.imu_repaired_raw_xyzw = imu_diagnostics.get("repaired_raw_xyzw")
            state.hand_positions_mm = extract_hand_positions_mm(msg)
            state.features.update(extract_hand_vector_features(state.hand_positions_mm))
            state.callback_sequence += 1

        return callback

    def configure_calibration(self, calibration: dict[str, Any]) -> None:
        if self.args.retargeting_mode != "dex_vector":
            return
        open_points = calibration.get("hand_position_open_mm")
        if not isinstance(open_points, dict):
            raise RuntimeError(
                "dex_vector retargeting requires hand_position_open_mm in the "
                "current calibration capture"
            )
        urdf_path = self.args.dex_urdf_path.expanduser().resolve()
        self._dex_retargeters = {
            side: ESROBOHandDexVectorRetargeter(
                urdf_path,
                side,
                open_points[source_side_for_target(side, self.args.swap_left_right_targets)],
                self.args,
                {
                    "open": calibration.get("open", {}).get(
                        source_side_for_target(side, self.args.swap_left_right_targets), {}
                    ),
                    "closed": calibration.get("closed", {}).get(
                        source_side_for_target(side, self.args.swap_left_right_targets), {}
                    ),
                },
            )
            for side in ("left", "right")
        }
        print(
            "[senseglove_bridge] Dex-style hand retargeting active: "
            f"urdf={urdf_path}, vectors={len(self._dex_retargeters['left'].specs)}/hand, "
            f"vector_set={self.args.dex_vector_set}, "
            f"iterations={self.args.dex_iterations}, damping={self.args.dex_damping}, "
            f"direction_weight={self.args.dex_direction_weight}, "
            f"curl_weight={self.args.dex_curl_weight}, "
            f"huber={self.args.dex_huber_delta}m, "
            f"limit_margin/weight={self.args.dex_joint_limit_margin}/"
            f"{self.args.dex_joint_limit_weight}.",
            flush=True,
        )

    def has_both_hands(self, max_age_s: float) -> bool:
        return self._is_fresh(self.left, max_age_s) and self._is_fresh(
            self.right, max_age_s
        )

    def _is_fresh(self, state: SideState, max_age_s: float) -> bool:
        if state.msg is None or state.stamp_monotonic is None:
            return False
        if time.monotonic() - state.stamp_monotonic > max_age_s:
            return False
        return (
            self.args.retargeting_mode != "vector" or len(state.hand_positions_mm) == 20
        )

    def topic_status(self) -> str:
        return (
            f"left={self._age_text(self.left)}/points={len(self.left.hand_positions_mm)}"
            f"/imu={self._imu_status(self.left)}({self.left.topic}) "
            f"right={self._age_text(self.right)}/points={len(self.right.hand_positions_mm)}"
            f"/imu={self._imu_status(self.right)}({self.right.topic})"
        )

    def _age_text(self, state: SideState) -> str:
        if state.stamp_monotonic is None:
            return "none"
        return f"{time.monotonic() - state.stamp_monotonic:.3f}s"

    @staticmethod
    def _imu_status(state: SideState) -> str:
        if state.imu_raw_norm is None:
            return "none"
        suffix = "/w-repaired" if state.imu_w_repaired else ""
        return f"norm={state.imu_raw_norm:.3f}{suffix}"

    def build_targets(
        self, calibration: dict[str, Any]
    ) -> tuple[list[float], dict[str, dict[str, float]]]:
        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        if self.args.retargeting_mode == "dex_vector":
            if self._dex_retargeters is None:
                self.configure_calibration(calibration)
            assert self._dex_retargeters is not None
            left_targets, left_curls = self._dex_retargeters["left"].retarget(
                source_left.hand_positions_mm
            )
            right_targets, right_curls = self._dex_retargeters["right"].retarget(
                source_right.hand_positions_mm
            )
            targets = left_targets + right_targets
            if self.args.smoothing_alpha < 1.0:
                alpha = max(0.0, self.args.smoothing_alpha)
                targets = [
                    alpha * current + (1.0 - alpha) * previous
                    for current, previous in zip(targets, self._last_targets)
                ]
            self._last_targets = targets
            return targets, {"left": left_curls, "right": right_curls}

        left_targets, left_curls = retarget_side(
            source_left.features,
            calibration["open"][
                "right" if self.args.swap_left_right_targets else "left"
            ],
            calibration["closed"][
                "right" if self.args.swap_left_right_targets else "left"
            ],
            "left",
            self.args,
        )
        right_targets, right_curls = retarget_side(
            source_right.features,
            calibration["open"][
                "left" if self.args.swap_left_right_targets else "right"
            ],
            calibration["closed"][
                "left" if self.args.swap_left_right_targets else "right"
            ],
            "right",
            self.args,
        )
        targets = left_targets + right_targets
        if self.args.smoothing_alpha < 1.0:
            alpha = max(0.0, self.args.smoothing_alpha)
            targets = [
                alpha * current + (1.0 - alpha) * previous
                for current, previous in zip(targets, self._last_targets)
            ]
        self._last_targets = targets
        return targets, {"left": left_curls, "right": right_curls}

    def build_orientation_deltas(
        self, calibration: dict[str, Any]
    ) -> dict[str, list[float]]:
        if self.args.disable_imu_orientation:
            return {}

        imu_neutral = calibration.get("imu_neutral")
        if not isinstance(imu_neutral, dict):
            return {}

        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        left_key = "right" if self.args.swap_left_right_targets else "left"
        right_key = "left" if self.args.swap_left_right_targets else "right"

        axis_calibration = calibration.get("imu_axis_calibration")
        delta_function = (
            orientation_delta_local_wxyz
            if isinstance(axis_calibration, dict)
            else orientation_delta_wxyz
        )
        left_delta = delta_function(
            source_left.imu_quat_wxyz, imu_neutral.get(left_key)
        )
        right_delta = delta_function(
            source_right.imu_quat_wxyz, imu_neutral.get(right_key)
        )
        if left_delta is None or right_delta is None:
            return {}
        if isinstance(axis_calibration, dict):
            left_delta = apply_imu_axis_calibration(
                left_delta, axis_calibration.get(left_key), target_side="left"
            )
            right_delta = apply_imu_axis_calibration(
                right_delta, axis_calibration.get(right_key), target_side="right"
            )
        return {"left": left_delta, "right": right_delta}

    def build_orientation_samples(
        self, calibration: dict[str, Any]
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, int]]:
        """Return robot-side absolute IMU samples, neutral samples, and sample times."""
        if self.args.disable_imu_orientation:
            return {}, {}, {}
        imu_neutral = calibration.get("imu_neutral")
        if not isinstance(imu_neutral, dict):
            return {}, {}, {}
        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        left_key = "right" if self.args.swap_left_right_targets else "left"
        right_key = "left" if self.args.swap_left_right_targets else "right"
        absolute: dict[str, list[float]] = {}
        neutral: dict[str, list[float]] = {}
        sample_times: dict[str, int] = {}
        for side, state, key in (
            ("left", source_left, left_key),
            ("right", source_right, right_key),
        ):
            if state.imu_quat_wxyz is None or state.stamp_monotonic is None:
                continue
            neutral_quat = imu_neutral.get(key)
            if not isinstance(neutral_quat, list) or len(neutral_quat) != 4:
                continue
            absolute[side] = list(state.imu_quat_wxyz)
            neutral[side] = normalize_quat_wxyz(neutral_quat)
            sample_times[side] = int(state.stamp_monotonic * 1.0e9)
        return absolute, neutral, sample_times

    def send(
        self,
        sock: socket.socket,
        calibration: dict[str, Any],
    ) -> tuple[
        list[float],
        dict[str, dict[str, float]],
        dict[str, list[float]],
        dict[str, Any],
    ]:
        targets, curls = self.build_targets(calibration)
        orientation_deltas = self.build_orientation_deltas(calibration)
        absolute_orientations, neutral_orientations, orientation_sample_times = (
            self.build_orientation_samples(calibration)
        )
        packet = {
            "timestamp": time.time(),
            "sender_monotonic_ns": time.monotonic_ns(),
            "source": "senseglove_ros",
            "type": "esrobo_hand_joints",
            "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
            "hand_joints": targets,
            "retargeting": {
                "method": (
                    "dex_style_fk_vector_optimization_to_esrobo_10_active_joints_per_hand"
                    if self.args.retargeting_mode == "dex_vector"
                    else "calibrated_hand_vectors_to_esrobo_10_active_joints_per_hand"
                    if self.args.retargeting_mode == "vector"
                    else "calibrated_finger_curl_to_esrobo_active_hand_joints"
                ),
                "mode": self.args.retargeting_mode,
                "robot_left_source": (
                    "right_glove" if self.args.swap_left_right_targets else "left_glove"
                ),
                "robot_right_source": (
                    "left_glove" if self.args.swap_left_right_targets else "right_glove"
                ),
            },
            "curls": curls,
        }
        source_left = self.right if self.args.swap_left_right_targets else self.left
        source_right = self.left if self.args.swap_left_right_targets else self.right
        if (
            len(source_left.hand_positions_mm) == 20
            and len(source_right.hand_positions_mm) == 20
        ):
            packet["hand_skeleton_positions"] = {
                "left": canonicalize_hand_points_for_robot(
                    source_left.hand_positions_mm, "left"
                ),
                "right": canonicalize_hand_points_for_robot(
                    source_right.hand_positions_mm, "right"
                ),
            }
            packet["hand_skeleton_layout"] = (
                "finger_major_5x4_thumb_to_pinky_proximal_to_distal"
            )
            packet["hand_skeleton_unit"] = "mm"
            packet["hand_skeleton_frame"] = "robot_hand_local"
            packet["hand_skeleton_source"] = (
                "senseglove_ros/SenseGloveState.hand_position"
            )
        if orientation_deltas:
            packet["hand_orientation_deltas"] = orientation_deltas
            packet["hand_orientation_delta_order"] = "wxyz"
            has_axis_calibration = isinstance(
                calibration.get("imu_axis_calibration"), dict
            )
            packet["hand_orientation_axis_calibrated"] = has_axis_calibration
            packet["hand_orientation_source_frame"] = (
                "calibrated_robot_hand_local_axes"
                if has_axis_calibration
                else self.args.imu_correction
            )
            packet["hand_orientation_retargeting"] = {
                "method": (
                    "motion_calibrated_senseglove_imu_delta"
                    if has_axis_calibration
                    else "calibrated_senseglove_imu_delta"
                ),
                "robot_left_source": (
                    "right_glove" if self.args.swap_left_right_targets else "left_glove"
                ),
                "robot_right_source": (
                    "left_glove" if self.args.swap_left_right_targets else "right_glove"
                ),
            }
            packet["hand_orientations_absolute"] = absolute_orientations
            packet["hand_orientation_neutral"] = neutral_orientations
            packet["hand_orientation_sample_monotonic_ns"] = orientation_sample_times
        encoded = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        sock.sendto(encoded, (self.args.host, self.args.port))
        self._sent_count += 1
        return targets, curls, orientation_deltas, packet

    def raw_recording_snapshot(self) -> dict[str, Any]:
        """Capture both physical glove messages that produced the current UDP target."""
        return {
            "left": side_recording_snapshot(self.left),
            "right": side_recording_snapshot(self.right),
        }


class HandRecordingWriter:
    """Write replay-compatible hand packets plus same-frame raw SenseGlove data."""

    def __init__(
        self,
        path: Path,
        args: argparse.Namespace,
        calibration: dict[str, Any],
    ):
        self.path = path.expanduser()
        if self.path.exists() and not args.record_overwrite:
            raise FileExistsError(
                f"recording already exists: {self.path}; use --record-overwrite to replace it"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", buffering=1)
        self.start_delay_s = args.record_start_delay_s
        self.duration_s = args.record_duration_s
        self.armed_monotonic: float | None = None
        self.started_monotonic: float | None = None
        self.sequence = 0
        self.complete = False
        self._last_countdown: int | None = None
        header = {
            "type": "esrobo_hand_recording_header",
            "format_version": 2,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": "real_senseglove_ros",
            "left_serial": args.left_serial,
            "right_serial": args.right_serial,
            "left_topic": args.left_topic,
            "right_topic": args.right_topic,
            "nominal_rate_hz": args.rate_hz,
            "record_start_delay_s": self.start_delay_s,
            "record_duration_s": self.duration_s,
            "imu_orientation_included": not args.disable_imu_orientation,
            "imu_correction": args.imu_correction,
            "imu_w_repair": args.imu_w_repair,
            "retargeting_mode": args.retargeting_mode,
            "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
            "raw_fields": {
                "joint_position": "SenseGloveState.position as published",
                "joint_absolute_velocity": "SenseGloveState.absolute_velocity as published",
                "hand_position": "millimeters, physical glove wrist-local frame",
                "finger_tip_position": "millimeters, physical glove wrist-local frame",
                "imu_orientation_raw": "xyzw, exactly as published",
                "imu_orientation_corrected": "wxyz, repaired/normalized and coordinate-corrected",
            },
            "calibration": calibration,
        }
        self.stream.write(json.dumps(header, separators=(",", ":")) + "\n")

    def update(
        self,
        now: float,
        packet: dict[str, Any],
        raw_snapshot: dict[str, Any],
    ) -> str:
        if self.complete:
            return "complete"
        if self.armed_monotonic is None:
            self.armed_monotonic = now
            print(
                "[senseglove_bridge] RECORDING ARMED: keep both hands still for "
                f"{self.start_delay_s:.1f}s; teleoperation is already active.",
                flush=True,
            )

        delay_elapsed = now - self.armed_monotonic
        if delay_elapsed < self.start_delay_s:
            remaining = max(self.start_delay_s - delay_elapsed, 0.0)
            countdown = max(1, math.ceil(remaining))
            if countdown != self._last_countdown:
                print(
                    f"[senseglove_bridge] Recording starts in {countdown}s; hold still.",
                    flush=True,
                )
                self._last_countdown = countdown
            return "countdown"

        if self.started_monotonic is None:
            self.started_monotonic = now
            print(
                f"[senseglove_bridge] RECORDING ACTIVE: move now; saving to {self.path}.",
                flush=True,
            )

        t_rel_s = now - self.started_monotonic
        if self.duration_s > 0.0 and t_rel_s > self.duration_s:
            self.complete = True
            self.stream.flush()
            print(
                "[senseglove_bridge] RECORDING COMPLETE: "
                f"saved={self.path} frames={self.sequence} duration={self.duration_s:.3f}s.",
                flush=True,
            )
            return "complete"

        item = {
            "type": "esrobo_hand_packet",
            "record_sequence": self.sequence,
            "t_rel_s": t_rel_s,
            "record_phase": "motion",
            "packet": packet,
            "senseglove_raw": raw_snapshot,
        }
        self.stream.write(json.dumps(item, separators=(",", ":")) + "\n")
        self.sequence += 1
        return "recording"

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            self.stream.close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    env_swap = _env_bool(
        "ESROBO_HAND_SWAP_LEFT_RIGHT_TARGETS",
        False,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("ESROBO_BODY_UDP_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050"))
    )
    parser.add_argument(
        "--left-serial",
        default=os.environ.get("SENSEGLOVE_LEFT_SERIAL", DEFAULT_LEFT_SERIAL),
    )
    parser.add_argument(
        "--right-serial",
        default=os.environ.get("SENSEGLOVE_RIGHT_SERIAL", DEFAULT_RIGHT_SERIAL),
    )
    parser.add_argument("--left-topic", default=None)
    parser.add_argument("--right-topic", default=None)
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=float(os.environ.get("SENSEGLOVE_BRIDGE_RATE_HZ", "60")),
    )
    parser.add_argument("--qos-depth", type=int, default=5)
    parser.add_argument("--max-state-age", type=float, default=0.5)
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--print-raw", action="store_true")
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Write replay-compatible JSONL with raw finger, skeleton, and IMU samples.",
    )
    parser.add_argument(
        "--record-start-delay-s",
        type=float,
        default=3.0,
        help="Keep streaming but wait this long after fresh data before recording.",
    )
    parser.add_argument(
        "--record-duration-s",
        type=float,
        default=0.0,
        help="Stop recording after this duration; 0 records until Ctrl-C.",
    )
    parser.add_argument(
        "--record-overwrite",
        action="store_true",
        help="Replace an existing --record-path instead of refusing to start.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help=(
            "Save this run's mandatory calibration result to this path. "
            "Existing content is replaced and is never reused to skip calibration."
        ),
    )
    parser.add_argument("--calibration-duration", type=float, default=10.0)
    parser.add_argument("--calibration-sample-start", type=float, default=6.0)
    parser.add_argument(
        "--calibration-method", choices=("last", "average"), default="last"
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Start each calibration phase without waiting for ENTER.",
    )
    parser.add_argument(
        "--disable-imu-orientation",
        action="store_true",
        help="Send only finger joints; do not send calibrated SenseGlove IMU hand orientation.",
    )
    parser.add_argument(
        "--imu-correction",
        choices=("unity_to_ros", "raw"),
        default=os.environ.get("SENSEGLOVE_IMU_CORRECTION", "unity_to_ros"),
        help="Coordinate correction for SenseGlove imu_orientation before calibration.",
    )
    parser.add_argument(
        "--imu-w-repair",
        choices=("auto", "off", "reconstruct"),
        default=os.environ.get("SENSEGLOVE_IMU_W_REPAIR", "auto"),
        help=(
            "Repair the observed Nova 2/SGCore fault where quaternion w duplicates z. "
            "Auto repairs only duplicated, non-unit samples."
        ),
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=1.0,
        help="1.0 disables smoothing; lower values smooth more.",
    )
    parser.add_argument(
        "--retargeting-mode",
        choices=("vector", "curl", "dex_vector"),
        default=os.environ.get("ESROBO_HAND_RETARGETING_MODE", "vector"),
        help=(
            "Use calibrated feature-vector mapping, legacy averaged joint-curl "
            "mapping, or dex_vector FK vector optimization."
        ),
    )
    parser.add_argument(
        "--dex-urdf-path",
        type=Path,
        default=Path(os.environ.get("ESROBO_DEX_HAND_URDF_PATH", str(DEFAULT_ESROBO_URDF_PATH))),
        help="URDF used by dex_vector hand FK retargeting.",
    )
    parser.add_argument(
        "--dex-iterations",
        type=int,
        default=int(os.environ.get("ESROBO_DEX_HAND_ITERATIONS", "1")),
        help="Damped least-squares iterations per hand per frame for dex_vector.",
    )
    parser.add_argument(
        "--dex-damping",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_DAMPING", "0.025")),
        help="Damping term for dex_vector least-squares solves.",
    )
    parser.add_argument(
        "--dex-max-joint-step",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_MAX_JOINT_STEP", "0.16")),
        help="Per-iteration joint update clamp in radians for dex_vector.",
    )
    parser.add_argument(
        "--dex-normal-delta",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_NORMAL_DELTA", "0.003")),
        help="Regularization toward the previous frame for dex_vector.",
    )
    parser.add_argument(
        "--dex-jacobian-eps",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_JACOBIAN_EPS", "0.0001")),
        help="Finite-difference step in radians for dex_vector numeric Jacobians.",
    )
    parser.add_argument(
        "--dex-direction-weight",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_DIRECTION_WEIGHT", "0.35")),
        help="Relative weight of unit-vector direction residuals in dex_vector.",
    )
    parser.add_argument(
        "--dex-vector-set",
        choices=("fingertip", "dense"),
        default=os.environ.get("ESROBO_DEX_HAND_VECTOR_SET", "fingertip"),
        help=(
            "Use dex-retargeting-style wrist-to-fingertip vectors, or the legacy "
            "dense per-phalanx objective."
        ),
    )
    parser.add_argument(
        "--dex-curl-weight",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_CURL_WEIGHT", "0.015")),
        help=(
            "Auxiliary weight that maps calibrated human open-to-fist progress "
            "onto each underactuated robot MCP flexion joint."
        ),
    )
    parser.add_argument(
        "--dex-huber-delta",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_HUBER_DELTA", "0.025")),
        help="Per-vector Huber transition in meters for noisy glove points.",
    )
    parser.add_argument(
        "--dex-joint-limit-margin",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_JOINT_LIMIT_MARGIN", "0.05")),
        help="Soft joint-limit margin as a fraction of each active joint range.",
    )
    parser.add_argument(
        "--dex-joint-limit-weight",
        type=float,
        default=float(os.environ.get("ESROBO_DEX_HAND_JOINT_LIMIT_WEIGHT", "0.001")),
        help="Soft joint-limit objective weight for dex_vector.",
    )
    parser.add_argument(
        "--spread-scale",
        type=float,
        default=None,
        help="Scale finger-vector spread into ESROBO roll joints; defaults to 1 for vector mode and 0 for curl mode.",
    )
    parser.add_argument("--vector-spread-range-deg", type=float, default=18.0)
    parser.add_argument("--thumb-roll-gain", type=float, default=0.35)
    parser.add_argument("--thumb-yaw-gain", type=float, default=0.50)
    parser.add_argument(
        "--swap-left-right-targets",
        dest="swap_left_right_targets",
        action="store_true",
        default=env_swap,
    )
    parser.add_argument(
        "--no-swap-left-right-targets",
        dest="swap_left_right_targets",
        action="store_false",
    )
    args = parser.parse_args(argv)
    args.left_topic = (
        args.left_topic or f"/senseglove/glove{args.left_serial}/lh/senseglove_states"
    )
    args.right_topic = (
        args.right_topic or f"/senseglove/glove{args.right_serial}/rh/senseglove_states"
    )
    args.rate_hz = max(args.rate_hz, 1.0)
    args.calibration_duration = max(args.calibration_duration, 0.1)
    args.calibration_sample_start = min(
        max(args.calibration_sample_start, 0.0), args.calibration_duration
    )
    args.smoothing_alpha = min(max(args.smoothing_alpha, 0.0), 1.0)
    args.spread_scale = (
        (1.0 if args.retargeting_mode == "vector" else 0.0)
        if args.spread_scale is None
        else args.spread_scale
    )
    args.vector_spread_range_deg = max(args.vector_spread_range_deg, 1.0)
    args.record_start_delay_s = max(args.record_start_delay_s, 0.0)
    args.record_duration_s = max(args.record_duration_s, 0.0)
    args.dex_iterations = max(args.dex_iterations, 1)
    args.dex_damping = max(args.dex_damping, 1.0e-6)
    args.dex_max_joint_step = max(args.dex_max_joint_step, 1.0e-4)
    args.dex_normal_delta = max(args.dex_normal_delta, 0.0)
    args.dex_jacobian_eps = max(args.dex_jacobian_eps, 1.0e-6)
    if args.retargeting_mode == "dex_vector" and not args.dex_urdf_path.exists():
        parser.error(f"--dex-urdf-path does not exist: {args.dex_urdf_path}")
    return args


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def extract_features(msg: Any) -> dict[str, float]:
    names = list(getattr(msg, "joint_names", []))
    positions = list(getattr(msg, "position", []))
    joints: dict[str, float] = {}
    for name, position in zip(names, positions):
        try:
            value = float(position)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            joints[_normalize_senseglove_joint_name(name)] = value

    features: dict[str, float] = {}
    for finger in FINGERS:
        flex_values = [
            joints[f"{finger}_{suffix}"]
            for suffix in FLEX_SUFFIXES
            if f"{finger}_{suffix}" in joints
        ]
        features[finger] = _mean(flex_values) if flex_values else 0.0
        brake_name = f"{finger}_brake"
        features[brake_name] = joints.get(brake_name, 0.0)
    return features


def side_recording_snapshot(state: SideState) -> dict[str, Any]:
    """Serialize the original ROS message and the corrected values used by retargeting."""
    msg = state.msg
    if msg is None:
        return {}
    orientation = getattr(msg, "imu_orientation", None)
    raw_imu_xyzw = (
        [
            _finite_float(getattr(orientation, axis, None))
            for axis in ("x", "y", "z", "w")
        ]
        if orientation is not None
        else None
    )
    return {
        "topic": state.topic,
        "callback_sequence": state.callback_sequence,
        "received_monotonic_ns": (
            int(state.stamp_monotonic * 1.0e9)
            if state.stamp_monotonic is not None
            else None
        ),
        "ros_stamp": _ros_stamp_dict(msg),
        "joint_names": [str(value) for value in getattr(msg, "joint_names", [])],
        "position": _finite_float_list(getattr(msg, "position", [])),
        "absolute_velocity": _finite_float_list(getattr(msg, "absolute_velocity", [])),
        "hand_position_mm_raw": _point_list(getattr(msg, "hand_position", [])),
        "finger_tip_position_mm_raw": _point_list(
            getattr(msg, "finger_tip_position", [])
        ),
        "hand_position_mm_used": [list(point) for point in state.hand_positions_mm],
        "imu_orientation_raw_xyzw": raw_imu_xyzw,
        "imu_raw_norm": state.imu_raw_norm,
        "imu_w_repaired": state.imu_w_repaired,
        "imu_orientation_corrected_wxyz": (
            list(state.imu_quat_wxyz) if state.imu_quat_wxyz is not None else None
        ),
        "retargeting_features": dict(state.features),
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_float_list(values: Any) -> list[float | None]:
    return [_finite_float(value) for value in list(values)]


def _point_list(points: Any) -> list[list[float | None]]:
    return [
        [_finite_float(getattr(point, axis, None)) for axis in ("x", "y", "z")]
        for point in list(points)
    ]


def _ros_stamp_dict(msg: Any) -> dict[str, int] | None:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    try:
        return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
    except (AttributeError, TypeError, ValueError):
        return None


def extract_imu_quat_wxyz(
    msg: Any,
    correction: str,
    imu_w_repair: str = "auto",
    diagnostics: dict[str, Any] | None = None,
    previous_raw_xyzw: list[float] | None = None,
) -> list[float] | None:
    orientation = getattr(msg, "imu_orientation", None)
    if orientation is None:
        return None
    raw_xyzw = [
        getattr(orientation, "x", 0.0),
        getattr(orientation, "y", 0.0),
        getattr(orientation, "z", 0.0),
        getattr(orientation, "w", 1.0),
    ]
    raw_xyzw, raw_norm, w_repaired = repair_senseglove_imu_w(
        raw_xyzw, imu_w_repair, previous_raw_xyzw
    )
    if diagnostics is not None:
        diagnostics["raw_norm"] = raw_norm
        diagnostics["w_repaired"] = w_repaired
        diagnostics["repaired_raw_xyzw"] = list(raw_xyzw)
    corrected_xyzw = correct_imu_quat_xyzw(raw_xyzw, correction)
    if corrected_xyzw is None:
        return None
    x, y, z, w = corrected_xyzw
    return normalize_quat_wxyz([w, x, y, z])


def repair_senseglove_imu_w(
    raw_xyzw: list[float], mode: str, previous_xyzw: list[float] | None = None
) -> tuple[list[float], float | None, bool]:
    """Repair Nova 2 samples where SGCore returns the z component as w."""
    try:
        values = [float(value) for value in raw_xyzw]
    except (TypeError, ValueError):
        return raw_xyzw, None, False
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return values, None, False

    x, y, z, w = values
    raw_norm = math.sqrt(sum(value * value for value in values))
    xyz_norm_sq = x * x + y * y + z * z
    duplicated_w = math.isclose(w, z, rel_tol=1.0e-5, abs_tol=1.0e-5)
    non_unit = abs(raw_norm - 1.0) > 0.02
    should_repair = mode == "reconstruct" or (
        mode == "auto" and duplicated_w and non_unit
    )
    if not should_repair or xyz_norm_sq > 1.02:
        return values, raw_norm, False

    repaired_w_abs = math.sqrt(max(0.0, 1.0 - min(xyz_norm_sq, 1.0)))
    candidates = ([x, y, z, repaired_w_abs], [x, y, z, -repaired_w_abs])
    previous = normalize_quat_xyzw(previous_xyzw) if previous_xyzw is not None else None
    if previous is None:
        repaired = list(candidates[0] if w >= 0.0 else candidates[1])
    else:
        repaired = list(
            max(candidates, key=lambda candidate: abs(sum(a * b for a, b in zip(previous, candidate))))
        )
        if sum(a * b for a, b in zip(previous, repaired)) < 0.0:
            repaired = [-value for value in repaired]
    return repaired, raw_norm, True


def extract_hand_positions_mm(msg: Any) -> list[list[float]]:
    """Extract SGCore's 5x4 hand joints, which are millimeters relative to the wrist."""
    positions: list[list[float]] = []
    for point in list(getattr(msg, "hand_position", [])):
        xyz = [float(getattr(point, axis, float("nan"))) for axis in ("x", "y", "z")]
        if not all(math.isfinite(value) for value in xyz):
            return []
        positions.append(xyz)
    if len(positions) != 20:
        return []

    # Prefer the separately published SGCore fingertips when they are geometrically
    # plausible. This also repairs recordings made with the older 19-point guard,
    # where the fourth pinky point was left at zero.
    fingertips = list(getattr(msg, "finger_tip_position", []))
    if len(fingertips) == 5:
        for finger_index, point in enumerate(fingertips):
            xyz = [
                float(getattr(point, axis, float("nan"))) for axis in ("x", "y", "z")
            ]
            distal = positions[finger_index * 4 + 2]
            distal_to_tip = math.dist(distal, xyz)
            if (
                all(math.isfinite(value) for value in xyz)
                and 1.0 <= distal_to_tip <= 120.0
            ):
                positions[finger_index * 4 + 3] = xyz
    return positions


def extract_hand_vector_features(
    hand_positions_mm: list[list[float]],
) -> dict[str, float]:
    """Describe a hand using palm-relative segment directions, independent of hand size."""
    if len(hand_positions_mm) != 20:
        return {}
    chains = {
        finger: [
            [float(value) for value in point]
            for point in hand_positions_mm[index * 4 : index * 4 + 4]
        ]
        for index, finger in enumerate(FINGERS)
    }
    lateral = _vector_normalize(
        _vector_subtract(chains["index"][0], chains["pinky"][0])
    )
    forward = _vector_normalize(chains["middle"][0])
    if lateral is None or forward is None:
        return {}
    normal = _vector_normalize(_vector_cross(lateral, forward))
    if normal is None:
        return {}
    forward = _vector_normalize(_vector_cross(normal, lateral))
    if forward is None:
        return {}

    features: dict[str, float] = {}
    for finger, points in chains.items():
        segments = [
            _vector_normalize(_vector_subtract(points[index + 1], points[index]))
            for index in range(3)
        ]
        if any(segment is None for segment in segments):
            return {}
        first, second, third = segments
        base_flex = math.atan2(_vector_dot(first, normal), _vector_dot(first, forward))
        pip_flex = _unsigned_vector_angle(first, second)
        dip_flex = _unsigned_vector_angle(second, third)
        features[f"vector_{finger}_curl"] = base_flex + pip_flex + dip_flex
        features[f"vector_{finger}_spread"] = math.atan2(
            _vector_dot(first, lateral),
            _vector_dot(first, forward),
        )
        features[f"vector_{finger}_elevation"] = math.atan2(
            _vector_dot(first, normal),
            math.hypot(_vector_dot(first, forward), _vector_dot(first, lateral)),
        )
    return features


def canonicalize_hand_points_for_robot(
    hand_positions_mm: list[list[float]], side: str
) -> list[list[float]]:
    """Express SGCore points in the mirrored ESROBO hand-link axis convention."""
    if len(hand_positions_mm) != 20 or side not in ("left", "right"):
        return []
    index_base = [float(value) for value in hand_positions_mm[4]]
    middle_base = [float(value) for value in hand_positions_mm[8]]
    pinky_base = [float(value) for value in hand_positions_mm[16]]
    lateral = _vector_normalize(_vector_subtract(index_base, pinky_base))
    forward = _vector_normalize(middle_base)
    if lateral is None or forward is None:
        return []
    normal = _vector_normalize(_vector_cross(lateral, forward))
    if normal is None:
        return []
    forward = _vector_normalize(_vector_cross(normal, lateral))
    if forward is None:
        return []
    robot_y_sign = -1.0 if side == "left" else 1.0
    return [
        [
            _vector_dot(point, normal),
            robot_y_sign * _vector_dot(point, lateral),
            _vector_dot(point, forward),
        ]
        for point in hand_positions_mm
    ]


def senseglove_points_with_wrist_m(
    hand_positions_mm: list[list[float]], side: str
) -> np.ndarray:
    points = canonicalize_hand_points_for_robot(hand_positions_mm, side)
    if len(points) != 20:
        raise ValueError("dex_vector requires 20 valid SenseGlove hand points")
    return np.asarray([[0.0, 0.0, 0.0]] + points, dtype=np.float64) * 0.001


def _parse_vec3_attr(
    element: ET.Element | None,
    attr_name: str,
    default: tuple[float, float, float],
) -> np.ndarray:
    if element is None or attr_name not in element.attrib:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in element.attrib[attr_name].split()]
    if len(values) != 3:
        raise ValueError(f"expected 3 values for URDF attribute {attr_name}")
    return np.asarray(values, dtype=np.float64)


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = xyz
    return matrix


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64
    )
    rot_y = np.asarray(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
    )
    rot_z = np.asarray(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return rot_z @ rot_y @ rot_x


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = _unit_vector_np(axis)
    x, y, z = [float(value) for value in unit_axis]
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    one_c = 1.0 - c
    rotation = np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return matrix


def _unit_vector_np(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _fit_rotation(source_vectors: np.ndarray, target_vectors: np.ndarray) -> np.ndarray:
    valid = (
        np.linalg.norm(source_vectors, axis=1) > 1.0e-8
    ) & (np.linalg.norm(target_vectors, axis=1) > 1.0e-8)
    if int(np.count_nonzero(valid)) < 3:
        return np.eye(3, dtype=np.float64)
    source = source_vectors[valid]
    target = target_vectors[valid]
    covariance = source.T @ target
    u_mat, _singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = vt_mat.T @ u_mat.T
    if np.linalg.det(rotation) < 0.0:
        vt_mat[-1, :] *= -1.0
        rotation = vt_mat.T @ u_mat.T
    return rotation.astype(np.float64)


def _robot_tip_lengths() -> dict[str, float]:
    return {
        "thumb": 0.024,
        "index": 0.022,
        "middle": 0.023,
        "ring": 0.022,
        "pinky": 0.020,
    }


def _vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(3)]


def _vector_dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _vector_cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _vector_normalize(vector: list[float]) -> list[float] | None:
    norm = math.sqrt(_vector_dot(vector, vector))
    if norm < 1.0e-6 or not math.isfinite(norm):
        return None
    return [value / norm for value in vector]


def _unsigned_vector_angle(first: list[float], second: list[float]) -> float:
    return math.acos(min(max(_vector_dot(first, second), -1.0), 1.0))


def correct_imu_quat_xyzw(raw_xyzw: list[float], correction: str) -> list[float] | None:
    q_in = normalize_quat_xyzw(raw_xyzw)
    if q_in is None:
        return None
    if correction == "raw":
        return q_in

    # Matches the official senseglove_interaction/common/imu_tf_broadcaster.py conversion.
    x, y, z, w = q_in
    unity_to_ros_q = normalize_quat_xyzw([z, -y, x, w])
    if unity_to_ros_q is None:
        return None
    q_corr = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
    return normalize_quat_xyzw(quat_multiply_xyzw(q_corr, unity_to_ros_q))


def _normalize_senseglove_joint_name(name: Any) -> str:
    text = str(name).strip().lower()
    if len(text) > 2 and text[1] == "_" and text[0] in ("l", "r"):
        text = text[2:]
    return text


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def collect_and_save_calibration(
    bridge: SenseGloveUdpBridge,
    node: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    calibration_path = args.calibration_file.expanduser()
    if calibration_path.exists():
        print(
            "[senseglove_bridge] A fresh calibration is mandatory for every live run; "
            f"{calibration_path} will be replaced.",
            flush=True,
        )
    else:
        print(
            "[senseglove_bridge] A fresh calibration is mandatory for every live run; "
            f"the result will be saved to {calibration_path}.",
            flush=True,
        )

    wait_for_both_hands(bridge, node, args)
    imu_axis_calibration = build_fixed_imu_axis_calibration()
    open_features, imu_neutral, open_points = collect_calibration_pose(
        bridge,
        node,
        args,
        label="1/2 neutral open hands",
        instruction=(
            "hold both wrists still in the desired robot initial hand orientation, "
            "keep the palms facing the body center, and open both hands naturally "
            "with straight fingers"
        ),
        require_imu=True,
    )
    closed_features, _closed_imu, closed_points = collect_calibration_pose(
        bridge,
        node,
        args,
        label="2/2 closed fists",
        instruction="keep both wrists neutral and close both hands into fists",
        require_imu=False,
    )
    calibration = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "senseglove_ros/SenseGloveState",
        "method": "two_pose_neutral_open_and_closed_fixed_imu_axes",
        "calibration_method": args.calibration_method,
        "valid_window": {
            "sample_start_s": args.calibration_sample_start,
            "duration_s": args.calibration_duration,
        },
        "imu_calibration": {
            "mandatory_each_live_run": True,
            "pose": "wrists_still_palms_facing_body_center_hands_open",
            "captured_with": "open_hand_reference",
            "sample_start_s": args.calibration_sample_start,
            "duration_s": args.calibration_duration,
        },
        "open": open_features,
        "closed": closed_features,
        "hand_position_open_mm": open_points,
        "hand_position_closed_mm": closed_points,
        "imu_neutral": imu_neutral,
        "imu_axis_calibration": imu_axis_calibration,
        "imu_correction": args.imu_correction,
        "imu_w_repair": args.imu_w_repair,
        "hand_joint_order": list(ESROBO_HAND_JOINT_ORDER),
    }
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_path.open("w", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"[senseglove_bridge] Saved calibration to {calibration_path}.", flush=True)
    return calibration


def wait_for_both_hands(
    bridge: SenseGloveUdpBridge, node: Any, args: argparse.Namespace
) -> None:
    next_print = 0.0
    while True:
        spin_once(node, timeout_sec=0.05)
        if bridge.has_both_hands(args.max_state_age):
            print(
                f"[senseglove_bridge] Receiving both gloves: {bridge.topic_status()}.",
                flush=True,
            )
            return
        now = time.monotonic()
        if now >= next_print:
            print(
                f"[senseglove_bridge] Waiting for SenseGlove topics: {bridge.topic_status()}.",
                flush=True,
            )
            next_print = now + 1.0


def collect_calibration_pose(
    bridge: SenseGloveUdpBridge,
    node: Any,
    args: argparse.Namespace,
    label: str,
    instruction: str,
    require_imu: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], dict[str, list[list[float]]]]:
    print(f"[senseglove_bridge] Calibration {label}: {instruction}.", flush=True)
    if not args.no_prompt:
        input(
            "[senseglove_bridge] Press ENTER when ready to start this "
            f"{args.calibration_duration:.1f}s pose capture..."
        )

    left_samples: list[dict[str, float]] = []
    right_samples: list[dict[str, float]] = []
    left_imu_samples: list[list[float]] = []
    right_imu_samples: list[list[float]] = []
    left_point_samples: list[list[list[float]]] = []
    right_point_samples: list[list[list[float]]] = []
    start = time.monotonic()
    next_print = 0.0
    while True:
        spin_once(node, timeout_sec=0.01)
        now = time.monotonic()
        elapsed = now - start
        if elapsed >= args.calibration_sample_start and bridge.has_both_hands(
            args.max_state_age
        ):
            left_samples.append(dict(bridge.left.features))
            right_samples.append(dict(bridge.right.features))
            if (
                bridge.left.imu_quat_wxyz is not None
                and bridge.right.imu_quat_wxyz is not None
            ):
                left_imu_samples.append(list(bridge.left.imu_quat_wxyz))
                right_imu_samples.append(list(bridge.right.imu_quat_wxyz))
            if (
                len(bridge.left.hand_positions_mm) == 20
                and len(bridge.right.hand_positions_mm) == 20
            ):
                left_point_samples.append(
                    [list(point) for point in bridge.left.hand_positions_mm]
                )
                right_point_samples.append(
                    [list(point) for point in bridge.right.hand_positions_mm]
                )
        if now >= next_print:
            remaining = max(0.0, args.calibration_duration - elapsed)
            if elapsed < args.calibration_sample_start:
                until_record = math.ceil(args.calibration_sample_start - elapsed)
                print(
                    f"[senseglove_bridge] Hold {label}: {until_record}s until recording starts, "
                    f"{math.ceil(remaining)}s until this pose locks.",
                    flush=True,
                )
            else:
                print(
                    f"[senseglove_bridge] Recording {label}: {math.ceil(remaining)}s remaining, "
                    f"samples={len(left_samples)} imu_samples={len(left_imu_samples)}.",
                    flush=True,
                )
            next_print = now + 1.0
        if elapsed >= args.calibration_duration:
            break

    if not left_samples or not right_samples:
        raise RuntimeError(
            f"no valid SenseGlove samples captured for calibration {label}"
        )
    if require_imu and (not left_imu_samples or not right_imu_samples):
        raise RuntimeError(
            f"no valid SenseGlove IMU samples captured for calibration {label}"
        )
    if args.retargeting_mode == "dex_vector" and (
        not left_point_samples or not right_point_samples
    ):
        raise RuntimeError(
            f"no valid SenseGlove 20-point samples captured for calibration {label}"
        )

    left_features = reduce_samples(left_samples, args.calibration_method)
    right_features = reduce_samples(right_samples, args.calibration_method)
    hand_points = {
        "left": reduce_point_samples(left_point_samples, args.calibration_method),
        "right": reduce_point_samples(right_point_samples, args.calibration_method),
    }
    imu_neutral: dict[str, list[float]] = {}
    if left_imu_samples and right_imu_samples:
        imu_neutral = {
            "left": reduce_quat_samples(left_imu_samples, args.calibration_method),
            "right": reduce_quat_samples(right_imu_samples, args.calibration_method),
        }
    print(
        f"[senseglove_bridge] Recorded {label}: "
        f"left={format_feature_preview(left_features)} right={format_feature_preview(right_features)} "
        f"points_left={len(hand_points['left'])} points_right={len(hand_points['right'])} "
        f"imu_left={format_quat_preview(imu_neutral.get('left'))} "
        f"imu_right={format_quat_preview(imu_neutral.get('right'))}.",
        flush=True,
    )
    return {"left": left_features, "right": right_features}, imu_neutral, hand_points


def spin_once(node: Any, timeout_sec: float) -> None:
    import rclpy

    rclpy.spin_once(node, timeout_sec=timeout_sec)


def reduce_samples(samples: list[dict[str, float]], method: str) -> dict[str, float]:
    if method == "last":
        return dict(samples[-1])
    keys = sorted({key for sample in samples for key in sample.keys()})
    return {
        key: _mean([sample[key] for sample in samples if key in sample]) for key in keys
    }


def reduce_quat_samples(samples: list[list[float]], method: str) -> list[float]:
    if method == "last":
        return normalize_quat_wxyz(samples[-1])

    reference = normalize_quat_wxyz(samples[0])
    aligned: list[list[float]] = []
    for sample in samples:
        quat = normalize_quat_wxyz(sample)
        if quat_dot_wxyz(reference, quat) < 0.0:
            quat = [-value for value in quat]
        aligned.append(quat)
    averaged = [
        sum(quat[index] for quat in aligned) / len(aligned) for index in range(4)
    ]
    return normalize_quat_wxyz(averaged)


def reduce_point_samples(
    samples: list[list[list[float]]], method: str
) -> list[list[float]]:
    if not samples:
        return []
    if method == "last":
        return [[float(value) for value in point] for point in samples[-1]]
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (20, 3):
        return []
    return array.mean(axis=0).astype(float).tolist()


def format_feature_preview(features: dict[str, float]) -> str:
    return ",".join(f"{finger}={features.get(finger, 0.0):.3f}" for finger in FINGERS)


def format_quat_preview(quat: list[float] | None) -> str:
    if quat is None:
        return "none"
    return "[" + ",".join(f"{value:.3f}" for value in quat) + "]"


def format_retargeting_preview(
    retargeting_mode: str, diagnostics: dict[str, float]
) -> str:
    if retargeting_mode == "dex_vector":
        mean_error = diagnostics.get("dex_vector_mean_error_m", 0.0)
        max_error = diagnostics.get("dex_vector_max_error_m", 0.0)
        return f"dex_mean={mean_error * 1000.0:.1f}mm,dex_max={max_error * 1000.0:.1f}mm"
    return format_feature_preview(diagnostics)


def retarget_side(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    robot_side: str,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, float]]:
    if args.retargeting_mode == "vector":
        return retarget_side_vectors(
            features, open_features, closed_features, robot_side, args
        )

    curls = {
        finger: normalize_feature(features, open_features, closed_features, finger)
        for finger in FINGERS
    }
    spreads = {
        finger: normalize_feature(
            features, open_features, closed_features, f"{finger}_brake"
        )
        for finger in FINGERS
    }
    joint_values = {
        f"{robot_side}_thumb_cmc_roll": limit_lerp(
            f"{robot_side}_thumb_cmc_roll", args.thumb_roll_gain * curls["thumb"]
        ),
        f"{robot_side}_thumb_cmc_yaw": limit_lerp(
            f"{robot_side}_thumb_cmc_yaw", args.thumb_yaw_gain * curls["thumb"]
        ),
        f"{robot_side}_thumb_cmc_pitch": limit_lerp(
            f"{robot_side}_thumb_cmc_pitch", curls["thumb"]
        ),
        f"{robot_side}_index_mcp_roll": limit_lerp(
            f"{robot_side}_index_mcp_roll", args.spread_scale * spreads["index"]
        ),
        f"{robot_side}_index_mcp_pitch": limit_lerp(
            f"{robot_side}_index_mcp_pitch", curls["index"]
        ),
        f"{robot_side}_middle_mcp_pitch": limit_lerp(
            f"{robot_side}_middle_mcp_pitch", curls["middle"]
        ),
        f"{robot_side}_ring_mcp_roll": limit_lerp(
            f"{robot_side}_ring_mcp_roll", args.spread_scale * spreads["ring"]
        ),
        f"{robot_side}_ring_mcp_pitch": limit_lerp(
            f"{robot_side}_ring_mcp_pitch", curls["ring"]
        ),
        f"{robot_side}_pinky_mcp_roll": limit_lerp(
            f"{robot_side}_pinky_mcp_roll", args.spread_scale * spreads["pinky"]
        ),
        f"{robot_side}_pinky_mcp_pitch": limit_lerp(
            f"{robot_side}_pinky_mcp_pitch", curls["pinky"]
        ),
    }
    order = [
        joint for joint in ESROBO_HAND_JOINT_ORDER if joint.startswith(f"{robot_side}_")
    ]
    return [joint_values[joint] for joint in order], curls


def retarget_side_vectors(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    robot_side: str,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, float]]:
    """Map palm-relative human finger vectors onto one hand's ten active robot joints."""
    curls = {
        finger: normalize_feature(
            features, open_features, closed_features, f"vector_{finger}_curl"
        )
        for finger in FINGERS
    }
    thumb_roll = normalize_feature_with_fallback(
        features,
        open_features,
        closed_features,
        "vector_thumb_elevation",
        args.thumb_roll_gain * curls["thumb"],
    )
    thumb_yaw = normalize_feature_with_fallback(
        features,
        open_features,
        closed_features,
        "vector_thumb_spread",
        args.thumb_yaw_gain * curls["thumb"],
    )

    spread_range = math.radians(args.vector_spread_range_deg)
    outward_sign = {"index": 1.0, "ring": -1.0, "pinky": -1.0}
    spreads: dict[str, float] = {}
    for finger, sign in outward_sign.items():
        key = f"vector_{finger}_spread"
        delta = float(features.get(key, open_features.get(key, 0.0))) - float(
            open_features.get(key, 0.0)
        )
        spreads[finger] = clamp01(sign * delta / spread_range) * args.spread_scale

    joint_values = {
        f"{robot_side}_thumb_cmc_roll": limit_lerp(
            f"{robot_side}_thumb_cmc_roll", thumb_roll
        ),
        f"{robot_side}_thumb_cmc_yaw": limit_lerp(
            f"{robot_side}_thumb_cmc_yaw", thumb_yaw
        ),
        f"{robot_side}_thumb_cmc_pitch": limit_lerp(
            f"{robot_side}_thumb_cmc_pitch", curls["thumb"]
        ),
        f"{robot_side}_index_mcp_roll": limit_lerp(
            f"{robot_side}_index_mcp_roll", spreads["index"]
        ),
        f"{robot_side}_index_mcp_pitch": limit_lerp(
            f"{robot_side}_index_mcp_pitch", curls["index"]
        ),
        f"{robot_side}_middle_mcp_pitch": limit_lerp(
            f"{robot_side}_middle_mcp_pitch", curls["middle"]
        ),
        f"{robot_side}_ring_mcp_roll": limit_lerp(
            f"{robot_side}_ring_mcp_roll", spreads["ring"]
        ),
        f"{robot_side}_ring_mcp_pitch": limit_lerp(
            f"{robot_side}_ring_mcp_pitch", curls["ring"]
        ),
        f"{robot_side}_pinky_mcp_roll": limit_lerp(
            f"{robot_side}_pinky_mcp_roll", spreads["pinky"]
        ),
        f"{robot_side}_pinky_mcp_pitch": limit_lerp(
            f"{robot_side}_pinky_mcp_pitch", curls["pinky"]
        ),
    }
    order = [
        joint for joint in ESROBO_HAND_JOINT_ORDER if joint.startswith(f"{robot_side}_")
    ]
    diagnostics = dict(curls)
    diagnostics.update({f"{finger}_spread": value for finger, value in spreads.items()})
    return [joint_values[joint] for joint in order], diagnostics


def normalize_feature_with_fallback(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    key: str,
    fallback: float,
) -> float:
    open_value = float(open_features.get(key, 0.0))
    closed_value = float(closed_features.get(key, open_value))
    if abs(closed_value - open_value) < 1.0e-4:
        return clamp01(fallback)
    return normalize_feature(features, open_features, closed_features, key)


def normalize_feature(
    features: dict[str, float],
    open_features: dict[str, float],
    closed_features: dict[str, float],
    key: str,
) -> float:
    open_value = float(open_features.get(key, 0.0))
    closed_value = float(closed_features.get(key, open_value))
    denominator = closed_value - open_value
    if abs(denominator) < 1.0e-6:
        return 0.0
    value = float(features.get(key, open_value))
    return clamp01((value - open_value) / denominator)


def limit_lerp(joint_name: str, normalized_value: float) -> float:
    lower, upper = ESROBO_HAND_JOINT_LIMITS[joint_name]
    return lower + clamp01(normalized_value) * (upper - lower)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def orientation_delta_wxyz(
    current: list[float] | None, neutral: Any
) -> list[float] | None:
    if current is None or neutral is None:
        return None
    try:
        neutral_quat = [float(value) for value in neutral]
    except (TypeError, ValueError):
        return None
    current_quat = normalize_quat_wxyz(current)
    neutral_quat = normalize_quat_wxyz(neutral_quat)
    return normalize_quat_wxyz(
        quat_multiply_wxyz(current_quat, quat_conjugate_wxyz(neutral_quat))
    )


def orientation_delta_local_wxyz(
    current: list[float] | None, neutral: Any
) -> list[float] | None:
    """Return the orientation change expressed in the neutral glove frame."""
    if current is None or neutral is None:
        return None
    try:
        neutral_quat = [float(value) for value in neutral]
    except (TypeError, ValueError):
        return None
    current_quat = normalize_quat_wxyz(current)
    neutral_quat = normalize_quat_wxyz(neutral_quat)
    return normalize_quat_wxyz(
        quat_multiply_wxyz(quat_conjugate_wxyz(neutral_quat), current_quat)
    )


def quat_wxyz_to_rotvec(quat: list[float]) -> np.ndarray:
    normalized = np.asarray(normalize_quat_wxyz(quat), dtype=np.float64)
    if normalized[0] < 0.0:
        normalized *= -1.0
    vector_norm = float(np.linalg.norm(normalized[1:]))
    if vector_norm < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(normalized[0]))
    return normalized[1:] * (angle / vector_norm)


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> list[float]:
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-9:
        return [1.0, 0.0, 0.0, 0.0]
    half_angle = 0.5 * angle
    scale = math.sin(half_angle) / angle
    return normalize_quat_wxyz(
        [math.cos(half_angle), *(vector * scale).astype(float).tolist()]
    )


def build_fixed_imu_axis_calibration() -> dict[str, dict[str, Any]]:
    """Return the device-level Nova 2 axis mapping used by every live run."""
    return {
        side: {
            "source_to_semantic_rotvec": matrix.astype(float).tolist(),
            "source": "fixed_nova2_glove_mount",
            "semantic_axis_order": ["lateral_bend", "vertical_bend", "palm_roll"],
        }
        for side, matrix in IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS.items()
    }


def apply_imu_axis_calibration(
    delta_wxyz: list[float], side_calibration: Any, target_side: str | None = None
) -> list[float]:
    if not isinstance(side_calibration, dict):
        return delta_wxyz
    raw_mapping = side_calibration.get("source_to_semantic_rotvec")
    mapping_is_semantic = raw_mapping is not None
    if raw_mapping is None:
        # Compatibility with recordings produced before source and target sides
        # were separated. These matrices already include the robot-side signs.
        raw_mapping = side_calibration.get("source_to_target_rotvec")
    try:
        mapping = np.asarray(raw_mapping, dtype=np.float64).reshape(3, 3)
    except (TypeError, ValueError):
        return delta_wxyz
    if not np.all(np.isfinite(mapping)):
        return delta_wxyz
    signs = np.ones(3, dtype=np.float64)
    if mapping_is_semantic and target_side in IMU_ROBOT_LOCAL_AXIS_SIGNS:
        signs = IMU_ROBOT_LOCAL_AXIS_SIGNS[target_side]

    target_rotvec = (mapping @ quat_wxyz_to_rotvec(delta_wxyz)) * signs
    return rotvec_to_quat_wxyz(target_rotvec)


def source_side_for_target(target_side: str, swap_left_right: bool) -> str:
    if target_side not in ("left", "right"):
        raise ValueError(f"unsupported target side: {target_side!r}")
    if not swap_left_right:
        return target_side
    return "right" if target_side == "left" else "left"


def normalize_quat_wxyz(raw: list[float]) -> list[float]:
    values = [float(value) for value in raw]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in values]


def normalize_quat_xyzw(raw: list[float]) -> list[float] | None:
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8 or not math.isfinite(norm):
        return None
    return [value / norm for value in values]


def quat_dot_wxyz(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def quat_conjugate_wxyz(quat: list[float]) -> list[float]:
    w, x, y, z = quat
    return [w, -x, -y, -z]


def quat_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quat_multiply_xyzw(left: list[float], right: list[float]) -> list[float]:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from senseglove_msgs.msg import SenseGloveState
    except ImportError as exc:
        print(
            f"[senseglove_bridge] Missing ROS 2 Python module: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[senseglove_bridge] Source /opt/ros/humble/setup.bash and the built senseglove_ros install/setup.bash.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node("senseglove_to_esrobo_hand_bridge")
    qos_profile = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=args.qos_depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    bridge = SenseGloveUdpBridge(node, SenseGloveState, args, qos_profile)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[senseglove_bridge] Subscribing left topic: {args.left_topic}", flush=True)
    print(
        f"[senseglove_bridge] Subscribing right topic: {args.right_topic}", flush=True
    )
    print(
        f"[senseglove_bridge] Subscription QoS: best_effort keep_last depth={args.qos_depth}.",
        flush=True,
    )
    print(
        f"[senseglove_bridge] Sending ESROBO hand joints to UDP {args.host}:{args.port}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Hand retargeting: "
        f"mode={args.retargeting_mode}, robot_active_joints=10/hand, "
        f"vector_spread_range={args.vector_spread_range_deg:.1f}deg, spread_scale={args.spread_scale:.2f}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Hand IMU orientation: "
        f"{'disabled' if args.disable_imu_orientation else 'enabled'} "
        f"(correction={args.imu_correction}, w_repair={args.imu_w_repair}).",
        flush=True,
    )
    print(
        "[senseglove_bridge] Robot hand mapping: "
        f"left<={'right_glove' if args.swap_left_right_targets else 'left_glove'} "
        f"right<={'left_glove' if args.swap_left_right_targets else 'right_glove'}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Fixed Nova 2 source-to-semantic IMU axes: "
        f"left={IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS['left'].astype(int).tolist()} "
        f"right={IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS['right'].astype(int).tolist()}.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Left IMU semantics: "
        "lateral<-source_Y vertical<-source_Z palm_roll<-source_X; "
        "full relative rotation mapping enabled.",
        flush=True,
    )
    print(
        "[senseglove_bridge] Fixed IMU target-axis signs: "
        f"left={IMU_ROBOT_LOCAL_AXIS_SIGNS['left'].astype(int).tolist()} "
        f"right={IMU_ROBOT_LOCAL_AXIS_SIGNS['right'].astype(int).tolist()} "
        "for [lateral, vertical, palm_roll].",
        flush=True,
    )
    if args.record_path is not None:
        print(
            "[senseglove_bridge] Hand recording requested: "
            f"path={args.record_path.expanduser()} start_delay={args.record_start_delay_s:.1f}s "
            f"duration={'until Ctrl-C' if args.record_duration_s <= 0.0 else f'{args.record_duration_s:.1f}s'}.",
            flush=True,
        )

    record_writer: HandRecordingWriter | None = None
    try:
        calibration = collect_and_save_calibration(bridge, node, args)
        bridge.configure_calibration(calibration)
        if args.record_path is not None:
            record_writer = HandRecordingWriter(args.record_path, args, calibration)
        period = 1.0 / args.rate_hz
        next_send = time.monotonic()
        next_print = 0.0
        sent_count = 0
        last_print_sent = 0
        last_print_time = time.monotonic()
        while rclpy.ok():
            now = time.monotonic()
            spin_timeout = min(max(next_send - now, 0.0), 0.01)
            spin_once(node, timeout_sec=spin_timeout)
            now = time.monotonic()
            if now < next_send:
                continue
            next_send += period
            if next_send < now - period:
                next_send = now + period

            if not bridge.has_both_hands(args.max_state_age):
                if args.print_interval > 0.0 and now >= next_print:
                    print(
                        f"[senseglove_bridge] Waiting for fresh glove data: {bridge.topic_status()}.",
                        flush=True,
                    )
                    next_print = now + args.print_interval
                continue

            targets, curls, orientation_deltas, packet = bridge.send(sock, calibration)
            sent_count += 1
            recording_status = "off"
            if record_writer is not None:
                recording_status = record_writer.update(
                    now, packet, bridge.raw_recording_snapshot()
                )

            if args.print_interval > 0.0 and now >= next_print:
                elapsed = max(now - last_print_time, 1.0e-6)
                hz = (sent_count - last_print_sent) / elapsed
                has_skeleton = (
                    len(bridge.left.hand_positions_mm) == 20
                    and len(bridge.right.hand_positions_mm) == 20
                )
                skeleton_status = "sgcore_5x4" if has_skeleton else "unavailable"
                print(
                    "[senseglove_bridge] "
                    f"sent={sent_count} hz={hz:.1f} "
                    f"imu={'on' if orientation_deltas else 'off'} "
                    f"imu_w_repair=left:{bridge.left.imu_w_repaired}/right:{bridge.right.imu_w_repaired} "
                    f"skeleton={skeleton_status} "
                    f"recording={recording_status} "
                    f"left={format_retargeting_preview(args.retargeting_mode, curls['left'])} "
                    f"right={format_retargeting_preview(args.retargeting_mode, curls['right'])}",
                    flush=True,
                )
                if args.print_raw:
                    print(
                        "[senseglove_bridge raw] "
                        f"left={format_feature_preview(bridge.left.features)} "
                        f"right={format_feature_preview(bridge.right.features)} "
                        f"targets0={[round(value, 4) for value in targets[:6]]} "
                        f"left_imu_delta={format_quat_preview(orientation_deltas.get('left'))} "
                        f"right_imu_delta={format_quat_preview(orientation_deltas.get('right'))}",
                        flush=True,
                    )
                last_print_time = now
                last_print_sent = sent_count
                next_print = now + args.print_interval
    except (KeyboardInterrupt, ExternalShutdownException):
        print("\n[senseglove_bridge] stopped.", flush=True)
    except RuntimeError as exc:
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
        print("\n[senseglove_bridge] stopped during ROS 2 shutdown.", flush=True)
    except OSError as exc:
        print(f"[senseglove_bridge] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if record_writer is not None:
            record_writer.close()
            if record_writer.sequence > 0 and not record_writer.complete:
                duration = (
                    time.monotonic() - record_writer.started_monotonic
                    if record_writer.started_monotonic is not None
                    else 0.0
                )
                print(
                    "[senseglove_bridge] RECORDING SAVED: "
                    f"path={record_writer.path} frames={record_writer.sequence} duration={duration:.3f}s.",
                    flush=True,
                )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
