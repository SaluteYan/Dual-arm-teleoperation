# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UDP body-tracking device for fixed-base bimanual teleoperation."""

from __future__ import annotations

import json
import logging
import math
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


logger = logging.getLogger(__name__)


BODY_FRAME_ALIASES = {
    "pelvis": "waist",
    "waist": "waist",
    "spine1": "waist",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "neck": "neck",
    "head": "head",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
}
"""Frame names accepted by :class:`UdpBimanualBodyDevice`.

PICO full-body packets use uppercase enum names such as ``LEFT_WRIST``; incoming names
are lower-cased before this alias table is applied.
"""


class UdpBimanualBodyDevice(DeviceBase):
    """Consume body-tracking arm poses over UDP and output ESROBO bimanual IK commands.

    Expected output action layout:
        default: left elbow pose 7 + left wrist pose 7 + right elbow pose 7 + right wrist pose 7
        + left hand joints + right hand joints.

        If ``enable_elbow_ik_targets`` is disabled, the legacy layout is used:
        left wrist pose 7 + right wrist pose 7 + left hand joints + right hand joints.

    Accepted UDP JSON formats:

    1. Compact ``frames`` dictionary. Poses are ``[x, y, z, qw, qx, qy, qz]`` unless
       ``quat_xyzw`` is provided explicitly::

           {
             "timestamp": 123.4,
             "frames": {
               "waist": {"pos": [..], "quat_wxyz": [..]},
               "left_wrist": {"pos": [..], "quat_wxyz": [..]},
               "right_wrist": {"pos": [..], "quat_wxyz": [..]}
             }
           }

    2. IsaacTeleop full-body payload with ``joint_names``, ``joint_positions``,
       ``joint_orientations`` in xyzw order, and ``joint_valid``.

    3. Optional hand joint and hand IMU payloads containing ``hand_joints`` and
       ``hand_orientation_deltas``. These packets update only the action tail
       and the wrist orientation source, so they can share the UDP port with
       full-body packets.
    """

    def __init__(self, cfg: UdpBimanualBodyDeviceCfg):
        """Initialize the UDP bimanual body-tracking device."""
        super().__init__(retargeters=None)
        if cfg.hand_joint_names is not None:
            self._hand_joint_names = list(cfg.hand_joint_names)
            self._hand_joint_count = len(cfg.hand_joint_names)
        else:
            self._hand_joint_names = []
            self._hand_joint_count = cfg.hand_joint_count

        self._cfg = cfg
        self._sim_device = cfg.sim_device
        self._bind_host = cfg.host
        self._bind_port = cfg.port
        self._max_stale_time_s = cfg.max_stale_time_s
        self._use_waist_frame = cfg.use_waist_frame
        self._position_scale = cfg.position_scale
        self._position_delta_signs = np.asarray(
            cfg.position_delta_signs, dtype=np.float32
        ).reshape(3)
        self._retargeting_mode = str(cfg.retargeting_mode).strip().lower()
        if self._retargeting_mode not in ("arm_vector", "wrist_delta"):
            print(
                f"[bodytracking_udp] Unknown retargeting_mode={cfg.retargeting_mode!r}; falling back to wrist_delta.",
                flush=True,
            )
            self._retargeting_mode = "wrist_delta"
        self._arm_vector_position_mode = (
            str(cfg.arm_vector_position_mode).strip().lower()
        )
        if self._arm_vector_position_mode == "segment_direction":
            self._arm_vector_position_mode = "segment_direction_absolute"
        allowed_arm_vector_position_modes = (
            "segment_direction_absolute",
            "segment_direction_relative",
            "delta",
        )
        if self._arm_vector_position_mode not in allowed_arm_vector_position_modes:
            print(
                "[bodytracking_udp] Unknown arm_vector_position_mode="
                f"{cfg.arm_vector_position_mode!r}; using segment_direction_absolute.",
                flush=True,
            )
            self._arm_vector_position_mode = "segment_direction_absolute"
        self._print_calibration_events = cfg.print_calibration_events
        self._swap_left_right_wrist_targets = bool(cfg.swap_left_right_wrist_targets)
        self._enable_elbow_ik_targets = bool(cfg.enable_elbow_ik_targets)
        self._requires_calibration = bool(cfg.require_calibration)
        self._arm_vector_uses_reference = self._arm_vector_position_mode in (
            "segment_direction_relative",
            "delta",
        )
        self._auto_start_reference_enabled = (
            bool(cfg.auto_start_reference)
            and not self._requires_calibration
            and self._retargeting_mode == "arm_vector"
            and self._arm_vector_uses_reference
        )
        self._auto_start_reference_delay_s = max(
            float(cfg.auto_start_reference_delay_s), 0.0
        )
        self._auto_start_reference_sample_start_s = max(
            float(cfg.auto_start_reference_sample_start_s), 0.0
        )
        if self._auto_start_reference_delay_s > 0.0:
            self._auto_start_reference_sample_start_s = min(
                self._auto_start_reference_sample_start_s,
                self._auto_start_reference_delay_s,
            )
        else:
            self._auto_start_reference_sample_start_s = 0.0
        self._auto_start_reference_max_position_std_m = max(
            float(cfg.auto_start_reference_max_position_std_m), 0.0
        )
        self._auto_start_reference_min_samples = max(
            int(cfg.auto_start_reference_min_samples), 1
        )
        self._auto_start_reference_require_waist = bool(
            cfg.auto_start_reference_require_waist
        )
        self._calibration_delay_s = max(float(cfg.calibration_delay_s), 0.0)
        self._calibration_sample_start_s = max(
            float(cfg.calibration_sample_start_s), 0.0
        )
        if self._calibration_delay_s > 0.0:
            self._calibration_sample_start_s = min(
                self._calibration_sample_start_s, self._calibration_delay_s
            )
        else:
            self._calibration_sample_start_s = 0.0
        self._calibration_prompt_interval_s = max(
            float(cfg.calibration_prompt_interval_s), 0.1
        )
        self._source_to_robot_rotation = np.asarray(
            cfg.source_to_robot_rotation, dtype=np.float32
        ).reshape(3, 3)
        self._use_hand_imu_orientation = bool(cfg.use_hand_imu_orientation)
        self._hand_only_mode = bool(cfg.hand_only_mode)
        self._hand_orientation_inherit_arm_frame = bool(
            cfg.hand_orientation_inherit_arm_frame
        )
        self._hand_imu_relative_to_arm_frame = bool(
            cfg.hand_imu_relative_to_arm_frame
        )
        self._hand_imu_max_stale_time_s = max(float(cfg.hand_imu_max_stale_time_s), 0.0)
        hand_imu_source_to_robot_rotation = cfg.hand_imu_source_to_robot_rotation
        if hand_imu_source_to_robot_rotation is None:
            hand_imu_source_to_robot_rotation = cfg.source_to_robot_rotation
        self._hand_imu_source_to_robot_rotation = np.asarray(
            hand_imu_source_to_robot_rotation, dtype=np.float32
        ).reshape(3, 3)
        rotation_det = float(np.linalg.det(self._source_to_robot_rotation))
        print(
            "[bodytracking_udp] Source-to-robot rotation "
            f"det={rotation_det:.3f} rows={np.round(self._source_to_robot_rotation, 4).tolist()}.",
            flush=True,
        )
        hand_imu_rotation_det = float(
            np.linalg.det(self._hand_imu_source_to_robot_rotation)
        )
        print(
            "[bodytracking_udp] Hand-IMU source-to-robot rotation "
            f"det={hand_imu_rotation_det:.3f} rows={np.round(self._hand_imu_source_to_robot_rotation, 4).tolist()}.",
            flush=True,
        )
        if rotation_det < 0.0:
            print(
                "[bodytracking_udp] Warning: source-to-robot rotation is a reflection; tracker orientation may be mirrored.",
                flush=True,
            )

        self._initial_left_pose = np.asarray(
            cfg.initial_left_wrist_pose, dtype=np.float32
        )
        self._initial_right_pose = np.asarray(
            cfg.initial_right_wrist_pose, dtype=np.float32
        )
        self._initial_left_pose_matrix = _pose_array_to_matrix(self._initial_left_pose)
        self._initial_right_pose_matrix = _pose_array_to_matrix(
            self._initial_right_pose
        )
        self._hand_imu_left_local_axis_signs = np.asarray(
            cfg.hand_imu_left_local_axis_signs, dtype=np.float32
        ).reshape(3)
        self._hand_imu_left_axis_order = np.asarray(
            cfg.hand_imu_left_axis_order, dtype=np.int64
        ).reshape(3)
        self._hand_imu_right_local_axis_signs = np.asarray(
            cfg.hand_imu_right_local_axis_signs, dtype=np.float32
        ).reshape(3)
        self._hand_imu_right_swap_xy = bool(cfg.hand_imu_right_swap_xy)
        for side, signs in (
            ("left", self._hand_imu_left_local_axis_signs),
            ("right", self._hand_imu_right_local_axis_signs),
        ):
            if not np.allclose(np.abs(signs), 1.0):
                raise ValueError(
                    f"hand_imu_{side}_local_axis_signs must contain only +1 or -1, "
                    f"got {signs.tolist()}"
                )
        if sorted(self._hand_imu_left_axis_order.tolist()) != [0, 1, 2]:
            raise ValueError(
                "hand_imu_left_axis_order must be a permutation of 0, 1, 2, "
                f"got {self._hand_imu_left_axis_order.tolist()}"
            )
        print(
            "[bodytracking_udp] Real SenseGlove IMU local axis signs: "
            f"left={self._hand_imu_left_local_axis_signs.tolist()} "
            f"right={self._hand_imu_right_local_axis_signs.tolist()}; "
            f"left_axis_order={self._hand_imu_left_axis_order.tolist()}; "
            f"right_swap_xy={self._hand_imu_right_swap_xy}.",
            flush=True,
        )
        self._robot_left_shoulder_position = np.asarray(
            cfg.robot_left_shoulder_position, dtype=np.float32
        ).reshape(3)
        self._robot_right_shoulder_position = np.asarray(
            cfg.robot_right_shoulder_position, dtype=np.float32
        ).reshape(3)
        self._robot_left_elbow_position = np.asarray(
            cfg.robot_left_elbow_position, dtype=np.float32
        ).reshape(3)
        self._robot_right_elbow_position = np.asarray(
            cfg.robot_right_elbow_position, dtype=np.float32
        ).reshape(3)
        self._initial_left_elbow_pose = _make_position_pose(
            self._robot_left_elbow_position
        )
        self._initial_right_elbow_pose = _make_position_pose(
            self._robot_right_elbow_position
        )
        self._initial_left_elbow_pose_matrix = _pose_array_to_matrix(
            self._initial_left_elbow_pose
        )
        self._initial_right_elbow_pose_matrix = _pose_array_to_matrix(
            self._initial_right_elbow_pose
        )
        self._arm_vector_max_reach = max(float(cfg.arm_vector_max_reach), 0.0)
        self._arm_vector_prediction_horizon_s = max(
            float(cfg.arm_vector_prediction_horizon_s), 0.0
        )
        self._arm_vector_prediction_lookback_s = max(
            float(cfg.arm_vector_prediction_lookback_s), 1.0e-3
        )
        self._arm_vector_prediction_max_velocity_m_s = max(
            float(cfg.arm_vector_prediction_max_velocity_m_s), 0.0
        )
        self._arm_vector_prediction_max_displacement_m = max(
            float(cfg.arm_vector_prediction_max_displacement_m), 0.0
        )
        self._upper_arm_angular_deadband_deg = max(
            float(cfg.upper_arm_angular_deadband_deg), 0.0
        )
        self._forearm_angular_deadband_deg = max(
            float(cfg.forearm_angular_deadband_deg), 0.0
        )
        self._robot_left_reference_rotation = _arm_points_to_rotation(
            {
                "shoulder": self._robot_left_shoulder_position,
                "elbow": self._robot_left_elbow_position,
                "wrist": self._initial_left_pose_matrix[:3, 3],
            }
        )
        self._robot_right_reference_rotation = _arm_points_to_rotation(
            {
                "shoulder": self._robot_right_shoulder_position,
                "elbow": self._robot_right_elbow_position,
                "wrist": self._initial_right_pose_matrix[:3, 3],
            }
        )
        self._robot_left_forearm_reference_rotation = _forearm_points_to_rotation(
            {
                "shoulder": self._robot_left_shoulder_position,
                "elbow": self._robot_left_elbow_position,
                "wrist": self._initial_left_pose_matrix[:3, 3],
            }
        )
        self._robot_right_forearm_reference_rotation = _forearm_points_to_rotation(
            {
                "shoulder": self._robot_right_shoulder_position,
                "elbow": self._robot_right_elbow_position,
                "wrist": self._initial_right_pose_matrix[:3, 3],
            }
        )
        self._robot_left_upper_arm_length = float(
            np.linalg.norm(
                self._robot_left_elbow_position - self._robot_left_shoulder_position
            )
        )
        self._robot_right_upper_arm_length = float(
            np.linalg.norm(
                self._robot_right_elbow_position - self._robot_right_shoulder_position
            )
        )
        self._robot_left_forearm_length = float(
            np.linalg.norm(
                self._initial_left_pose_matrix[:3, 3] - self._robot_left_elbow_position
            )
        )
        self._robot_right_forearm_length = float(
            np.linalg.norm(
                self._initial_right_pose_matrix[:3, 3]
                - self._robot_right_elbow_position
            )
        )
        self._left_calibration_pose_matrix: np.ndarray | None = None
        self._right_calibration_pose_matrix: np.ndarray | None = None
        self._down_left_wrist_matrix: np.ndarray | None = None
        self._down_right_wrist_matrix: np.ndarray | None = None
        self._forward_left_wrist_matrix: np.ndarray | None = None
        self._forward_right_wrist_matrix: np.ndarray | None = None
        self._down_left_arm_points: dict[str, np.ndarray] | None = None
        self._down_right_arm_points: dict[str, np.ndarray] | None = None
        self._forward_left_arm_points: dict[str, np.ndarray] | None = None
        self._forward_right_arm_points: dict[str, np.ndarray] | None = None
        self._inferred_left_shoulder_pos: np.ndarray | None = None
        self._inferred_right_shoulder_pos: np.ndarray | None = None
        self._calibration_started_monotonic: float | None = None
        self._next_calibration_prompt_monotonic = 0.0
        self._printed_calibration_locked = False
        self._printed_visualization_active = False
        self._calibration_phase = "down"
        self._calibration_left_samples: list[np.ndarray] = []
        self._calibration_right_samples: list[np.ndarray] = []
        self._calibration_left_arm_samples: list[dict[str, np.ndarray]] = []
        self._calibration_right_arm_samples: list[dict[str, np.ndarray]] = []
        self._auto_reference_started_monotonic: float | None = None
        self._next_auto_reference_prompt_monotonic = 0.0
        self._auto_reference_left_arm_samples: list[dict[str, np.ndarray]] = []
        self._auto_reference_right_arm_samples: list[dict[str, np.ndarray]] = []
        self._auto_reference_left_wrist_samples: list[np.ndarray] = []
        self._auto_reference_right_wrist_samples: list[np.ndarray] = []
        self._auto_reference_waist_samples: list[np.ndarray] = []
        self._auto_reference_left_arm_points: dict[str, np.ndarray] | None = None
        self._auto_reference_right_arm_points: dict[str, np.ndarray] | None = None
        self._auto_reference_left_wrist_matrix: np.ndarray | None = None
        self._auto_reference_right_wrist_matrix: np.ndarray | None = None
        self._auto_reference_waist_matrix: np.ndarray | None = None

        self._previous_left_pose = self._initial_left_pose.copy()
        self._previous_right_pose = self._initial_right_pose.copy()
        self._previous_left_elbow_pose = self._initial_left_elbow_pose.copy()
        self._previous_right_elbow_pose = self._initial_right_elbow_pose.copy()
        self._hand_joint_targets = np.zeros(self._hand_joint_count, dtype=np.float32)
        self._hand_skeleton_positions: dict[str, np.ndarray] = {}
        self._hand_skeleton_frame = "source"
        self._hand_orientation_delta_matrices: dict[str, np.ndarray] = {}
        self._hand_orientation_is_forearm_relative = False
        self._hand_orientation_sample_times_s: dict[str, float] = {}
        self._hand_arm_reference_rotations: dict[str, np.ndarray] = {}
        self._hand_imu_reference_delta_matrices: dict[str, np.ndarray] = {}
        self._wrist_arm_reference_rotations: dict[str, np.ndarray] = {}
        self._hand_skeleton_arm_reference_rotations: dict[str, np.ndarray] = {}
        self._hand_skeleton_visual_rotations: dict[str, np.ndarray] = {}
        self._forearm_plane_normals: dict[str, np.ndarray] = {}
        self._previous_action = self._make_action(
            self._previous_left_pose,
            self._previous_right_pose,
            self._previous_left_elbow_pose,
            self._previous_right_elbow_pose,
        )
        self._frames: dict[str, np.ndarray] = {}
        self._body_frame_history: deque[tuple[float, dict[str, np.ndarray]]] = deque(
            maxlen=8
        )
        self._held_segment_directions: dict[str, dict[str, np.ndarray]] = {
            "left": {},
            "right": {},
        }
        self._last_packet_time_monotonic: float | None = None
        self._last_hand_packet_time_monotonic: float | None = None
        self._last_hand_orientation_packet_time_monotonic: float | None = None
        self._last_sender: tuple[str, int] | None = None
        self._last_hand_sender: tuple[str, int] | None = None
        self._last_hand_orientation_sender: tuple[str, int] | None = None
        self._additional_callbacks: dict[Any, Callable] = {}
        self._debug_print = False
        self._debug_print_interval = 60
        self._debug_frame = 0
        self._receive_debug_frame = 0
        self._drained_packet_total = 0
        self._empty_packet_total = 0
        self._printed_waiting = False
        self._printed_stale = False
        self._printed_arm_vector_fallback = False
        self._printed_hand_targets_active = False
        self._printed_hand_orientations_active = False
        self._printed_hand_imu_stale = False

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)
        self._socket.bind((self._bind_host, self._bind_port))
        logger.info(
            "UDP bimanual body device listening on %s:%s",
            self._bind_host,
            self._bind_port,
        )
        print(
            f"[bodytracking_udp] Listening for body-tracking UDP packets on {self._bind_host}:{self._bind_port}.",
            flush=True,
        )
        if self._print_calibration_events:
            if self._requires_calibration:
                print(
                    "[bodytracking_udp] Two-pose calibration: step 1/2 hold both arms naturally down, "
                    "step 2/2 raise and stretch both arms horizontally in front of the chest. "
                    f"Each pose locks after {self._calibration_delay_s:.1f}s; valid samples are recorded from "
                    f"{self._calibration_sample_start_s:.1f}s to {self._calibration_delay_s:.1f}s.",
                    flush=True,
                )
            else:
                if self._auto_start_reference_enabled:
                    print(
                        "[bodytracking_udp] Body calibration disabled; automatic startup reference is enabled. "
                        "Hold the neutral start pose still until the reference locks. "
                        f"Samples record from {self._auto_start_reference_sample_start_s:.1f}s to "
                        f"{self._auto_start_reference_delay_s:.1f}s.",
                        flush=True,
                    )
                else:
                    print(
                        "[bodytracking_udp] Body calibration disabled. In arm_vector mode, the first fresh "
                        "full-body shoulder/elbow/wrist packet drives the robot immediately.",
                        flush=True,
                    )
            print(
                "[bodytracking_udp] Wrist target mapping: "
                f"robot_left<=human_{self._source_side_for_target('left')} "
                f"robot_right<=human_{self._source_side_for_target('right')}.",
                flush=True,
            )
            print(
                f"[bodytracking_udp] Retargeting mode: {self._retargeting_mode}.",
                flush=True,
            )
            print(
                f"[bodytracking_udp] Arm-vector position mode: {self._arm_vector_position_mode}.",
                flush=True,
            )
            print(
                "[bodytracking_udp] Arm-vector prediction: "
                f"horizon={self._arm_vector_prediction_horizon_s:.3f}s "
                f"lookback={self._arm_vector_prediction_lookback_s:.3f}s "
                f"max_velocity={self._arm_vector_prediction_max_velocity_m_s:.2f}m/s "
                f"max_displacement={self._arm_vector_prediction_max_displacement_m:.3f}m.",
                flush=True,
            )
            print(
                "[bodytracking_udp] Arm-segment angular deadband: "
                f"upper_arm={self._upper_arm_angular_deadband_deg:.3f}deg "
                f"forearm={self._forearm_angular_deadband_deg:.3f}deg.",
                flush=True,
            )
            print(
                "[bodytracking_udp] SenseGlove IMU hand orientation: "
                f"{'enabled' if self._use_hand_imu_orientation else 'disabled'}; "
                + (
                    "hand-only mode fixes wrist positions at the robot initial pose; fresh SenseGlove IMU packets "
                    "drive wrist orientation."
                    if self._hand_only_mode
                    else "PICO/body tracking drives wrist position; only fresh SenseGlove IMU packets drive wrist "
                    "orientation, otherwise the initial wrist orientation is held."
                ),
                flush=True,
            )
            print(
                "[bodytracking_udp] Robot-frame position delta signs: "
                f"{np.round(self._position_delta_signs, 4).tolist()}.",
                flush=True,
            )

        self._enable_visualization = cfg.enable_visualization
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path=cfg.visualization_prim_path,
                markers={
                    "left_joint": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.7, 1.0)
                        ),
                    ),
                    "right_joint": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.55, 0.1)
                        ),
                    ),
                    "waist_joint": sim_utils.SphereCfg(
                        radius=0.028,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.2, 1.0, 0.35)
                        ),
                    ),
                    "body_joint": sim_utils.SphereCfg(
                        radius=0.018,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.85, 0.95, 1.0)
                        ),
                    ),
                    "left_bone": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.7, 1.0), roughness=1.0
                        ),
                    ),
                    "right_bone": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.55, 0.1), roughness=1.0
                        ),
                    ),
                    "body_bone": sim_utils.CylinderCfg(
                        radius=0.006,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.85, 0.95, 1.0), roughness=1.0
                        ),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)
            if not cfg.visualize_during_calibration:
                self._markers.set_visibility(False)

        self._enable_hand_skeleton_visualization = bool(
            cfg.enable_hand_skeleton_visualization
        )
        self._hand_skeleton_palms_face_each_other = bool(
            cfg.hand_skeleton_palms_face_each_other
        )
        if self._enable_hand_skeleton_visualization:
            hand_marker_cfg = VisualizationMarkersCfg(
                prim_path=cfg.hand_skeleton_visualization_prim_path,
                markers={
                    "left_joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.75, 1.0)
                        ),
                    ),
                    "right_joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.55, 0.1)
                        ),
                    ),
                    "left_bone": sim_utils.CylinderCfg(
                        radius=0.0025,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.1, 0.75, 1.0), roughness=1.0
                        ),
                    ),
                    "right_bone": sim_utils.CylinderCfg(
                        radius=0.0025,
                        height=1.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.55, 0.1), roughness=1.0
                        ),
                    ),
                },
            )
            self._hand_skeleton_markers = VisualizationMarkers(hand_marker_cfg)
            self._hand_skeleton_markers.set_visibility(False)
            print(
                "[bodytracking_udp] SenseGlove FK hand-skeleton visualization enabled at "
                f"{cfg.hand_skeleton_visualization_prim_path}; "
                "robot-consistent arm-frame mounting="
                f"{not cfg.hand_skeleton_palms_face_each_other}; "
                "legacy palms-face-each-other mounting="
                f"{cfg.hand_skeleton_palms_face_each_other}; "
                "left local forward-axis correction="
                f"{cfg.hand_skeleton_left_local_rotation_deg:.1f}deg.",
                flush=True,
            )

    def __del__(self):
        """Close the UDP socket when the device is destroyed."""
        if hasattr(self, "_socket"):
            self._socket.close()

    def __str__(self) -> str:
        """Return a short device description."""
        return (
            f"UDP Bimanual Body Tracking Device: {self.__class__.__name__}\n"
            f"\tListen: {self._bind_host}:{self._bind_port}\n"
            f"\tWaist-relative retargeting: {self._use_waist_frame}\n"
            f"\tRetargeting mode: {self._retargeting_mode}\n"
            f"\tArm-vector position mode: {self._arm_vector_position_mode}\n"
            f"\tElbow IK targets: {self._enable_elbow_ik_targets}\n"
            f"\tBody calibration required: {self._requires_calibration}\n"
            f"\tAuto startup reference: {self._auto_start_reference_enabled}\n"
            f"\tPosition delta signs: {np.round(self._position_delta_signs, 4).tolist()}\n"
            f"\tWrist target mapping: robot_left<=human_{self._source_side_for_target('left')} "
            f"robot_right<=human_{self._source_side_for_target('right')}\n"
            f"\tHand-only mode: {self._hand_only_mode}\n"
            f"\tSenseGlove IMU orientation: {self._use_hand_imu_orientation}\n"
            f"\tHand joint targets: {self._hand_joint_count}\n"
            f"\tAction dimensions: {self._action_dimension()}"
        )

    def reset(self):
        """Reset body-tracking state and hold the robot at its configured initial wrist poses."""
        self._left_calibration_pose_matrix = None
        self._right_calibration_pose_matrix = None
        self._down_left_wrist_matrix = None
        self._down_right_wrist_matrix = None
        self._forward_left_wrist_matrix = None
        self._forward_right_wrist_matrix = None
        self._down_left_arm_points = None
        self._down_right_arm_points = None
        self._forward_left_arm_points = None
        self._forward_right_arm_points = None
        self._inferred_left_shoulder_pos = None
        self._inferred_right_shoulder_pos = None
        self._calibration_started_monotonic = None
        self._next_calibration_prompt_monotonic = 0.0
        self._printed_calibration_locked = False
        self._printed_visualization_active = False
        self._calibration_phase = "down"
        self._calibration_left_samples = []
        self._calibration_right_samples = []
        self._calibration_left_arm_samples = []
        self._calibration_right_arm_samples = []
        self._auto_reference_started_monotonic = None
        self._next_auto_reference_prompt_monotonic = 0.0
        self._auto_reference_left_arm_samples = []
        self._auto_reference_right_arm_samples = []
        self._auto_reference_left_wrist_samples = []
        self._auto_reference_right_wrist_samples = []
        self._auto_reference_waist_samples = []
        self._auto_reference_left_arm_points = None
        self._auto_reference_right_arm_points = None
        self._auto_reference_left_wrist_matrix = None
        self._auto_reference_right_wrist_matrix = None
        self._auto_reference_waist_matrix = None
        self._previous_left_pose = self._initial_left_pose.copy()
        self._previous_right_pose = self._initial_right_pose.copy()
        self._previous_left_elbow_pose = self._initial_left_elbow_pose.copy()
        self._previous_right_elbow_pose = self._initial_right_elbow_pose.copy()
        self._previous_action = self._make_action(
            self._previous_left_pose,
            self._previous_right_pose,
            self._previous_left_elbow_pose,
            self._previous_right_elbow_pose,
        )
        self._printed_waiting = False
        self._printed_stale = False
        self._printed_arm_vector_fallback = False
        self._printed_hand_targets_active = False
        self._printed_hand_orientations_active = False
        self._printed_hand_imu_stale = False
        self._body_frame_history.clear()
        self._held_segment_directions = {"left": {}, "right": {}}
        self._hand_arm_reference_rotations.clear()
        self._hand_imu_reference_delta_matrices.clear()
        self._wrist_arm_reference_rotations.clear()
        self._hand_skeleton_arm_reference_rotations.clear()
        self._hand_skeleton_visual_rotations.clear()
        self._hand_orientation_sample_times_s.clear()
        self._forearm_plane_normals.clear()
        if hasattr(self, "_markers") and not self._cfg.visualize_during_calibration:
            self._markers.set_visibility(False)
        if hasattr(self, "_hand_skeleton_markers"):
            self._hand_skeleton_markers.set_visibility(False)
        if self._print_calibration_events and self._requires_calibration:
            print(
                "[bodytracking_udp] Calibration reset. Step 1/2: hold both arms naturally down; "
                f"valid samples record from {self._calibration_sample_start_s:.1f}s to "
                f"{self._calibration_delay_s:.1f}s once both wrist frames are visible.",
                flush=True,
            )
        elif self._print_calibration_events:
            if self._auto_start_reference_enabled:
                print(
                    "[bodytracking_udp] Automatic startup reference reset. Hold the neutral start pose still; "
                    "robot targets and skeleton visualization stay at the initial pose until the reference locks.",
                    flush=True,
                )
            else:
                print(
                    "[bodytracking_udp] Body calibration remains disabled after reset; waiting for fresh "
                    "full-body shoulder/elbow/wrist packets.",
                    flush=True,
                )

    def reset_retargeters(self):
        """Compatibility hook used by the teleoperation script's start/reset callback."""
        self.reset()

    def add_callback(self, key: Any, func: Callable):
        """Store callback bindings for API compatibility."""
        self._additional_callbacks[key] = func

    def set_debug_print(self, enabled: bool = True, interval: int = 60):
        """Enable periodic debug printing."""
        self._debug_print = enabled
        self._debug_print_interval = max(int(interval), 1)

    def advance(self) -> torch.Tensor:
        """Return the latest body-tracking command, or hold the previous command when data is stale."""
        self._receive_latest_packet()
        if self._hand_only_mode:
            return self._advance_hand_only()
        if not self._has_fresh_packet():
            return torch.tensor(
                self._previous_action, dtype=torch.float32, device=self._sim_device
            )

        if (
            self._requires_calibration
            and self._calibration_pending()
            and not self._advance_two_pose_calibration()
        ):
            if self._enable_visualization and self._cfg.visualize_during_calibration:
                self._visualize_body_pose()
            return torch.tensor(
                self._previous_action, dtype=torch.float32, device=self._sim_device
            )

        if (
            self._startup_reference_pending()
            and not self._advance_auto_start_reference()
        ):
            if self._enable_visualization and self._cfg.visualize_during_calibration:
                self._visualize_body_pose()
            return torch.tensor(
                self._previous_action, dtype=torch.float32, device=self._sim_device
            )

        was_calibrating = self._calibration_pending()
        left_elbow_pose, left_pose = self._retarget_target_pair("left")
        right_elbow_pose, right_pose = self._retarget_target_pair("right")
        action = self._make_action(
            left_pose, right_pose, left_elbow_pose, right_elbow_pose
        )
        self._previous_action = action
        if (
            was_calibrating
            and not self._calibration_pending()
            and not self._printed_calibration_locked
        ):
            self._printed_calibration_locked = True
            if self._print_calibration_events:
                print(
                    "[bodytracking_udp] Calibration locked. You can start moving now.",
                    flush=True,
                )

        if self._enable_visualization:
            self._visualize_body_pose()
        if self._enable_hand_skeleton_visualization and self._has_fresh_hand_packet():
            self._visualize_hand_skeletons()
        if self._debug_print:
            self._print_debug(left_pose, right_pose, left_elbow_pose, right_elbow_pose)

        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    def _advance_hand_only(self) -> torch.Tensor:
        """Drive fixed-position wrists and hand joints from SenseGlove packets only."""
        if not self._has_fresh_hand_packet():
            return torch.tensor(
                self._previous_action, dtype=torch.float32, device=self._sim_device
            )

        left_pose = self._hand_only_wrist_pose("left")
        right_pose = self._hand_only_wrist_pose("right")
        self._previous_left_pose = left_pose
        self._previous_right_pose = right_pose
        action = self._make_action(
            left_pose,
            right_pose,
            self._initial_left_elbow_pose,
            self._initial_right_elbow_pose,
        )
        self._previous_action = action
        if self._enable_hand_skeleton_visualization:
            self._visualize_hand_skeletons()
        if self._debug_print:
            self._print_debug(left_pose, right_pose)
        return torch.tensor(action, dtype=torch.float32, device=self._sim_device)

    def _has_fresh_hand_packet(self) -> bool:
        if self._last_hand_packet_time_monotonic is None:
            return False
        age = time.monotonic() - self._last_hand_packet_time_monotonic
        if age > self._max_stale_time_s:
            if not self._printed_stale:
                print(
                    f"[bodytracking_udp] SenseGlove stream stale for {age:.2f}s; holding the previous hand target.",
                    flush=True,
                )
                self._printed_stale = True
            return False
        return True

    def _hand_only_wrist_pose(self, side: str) -> np.ndarray:
        initial_pose = (
            self._initial_left_pose_matrix
            if side == "left"
            else self._initial_right_pose_matrix
        )
        target_pose = initial_pose.copy()
        rotation_delta = self._hand_imu_delta_matrix_for_target(side)
        if rotation_delta is not None:
            target_pose[:3, :3] = rotation_delta @ initial_pose[:3, :3]
        return _pose_matrix_to_array(target_pose)

    def _calibration_pending(self) -> bool:
        if not self._requires_calibration:
            return False
        return (
            self._left_calibration_pose_matrix is None
            or self._right_calibration_pose_matrix is None
        )

    def _startup_reference_pending(self) -> bool:
        if not self._auto_start_reference_enabled:
            return False
        return (
            self._auto_reference_left_arm_points is None
            or self._auto_reference_right_arm_points is None
        )

    def _advance_two_pose_calibration(self) -> bool:
        left_matrix, left_frame_name = self._current_wrist_matrix("left")
        right_matrix, right_frame_name = self._current_wrist_matrix("right")
        now = time.monotonic()

        if left_matrix is None or right_matrix is None:
            self._calibration_started_monotonic = None
            if (
                self._print_calibration_events
                and now >= self._next_calibration_prompt_monotonic
            ):
                visible = [
                    name
                    for name, matrix in (
                        (left_frame_name, left_matrix),
                        (right_frame_name, right_matrix),
                    )
                    if matrix is not None
                ]
                visible_text = ",".join(visible) if visible else "none"
                print(
                    f"[bodytracking_udp] Calibration step {self._calibration_phase_label()}: "
                    f"waiting for both wrist frames. visible=[{visible_text}]",
                    flush=True,
                )
                self._next_calibration_prompt_monotonic = (
                    now + self._calibration_prompt_interval_s
                )
            return False

        if self._calibration_started_monotonic is None:
            self._calibration_started_monotonic = now
            self._next_calibration_prompt_monotonic = 0.0
            self._calibration_left_samples = []
            self._calibration_right_samples = []
            self._calibration_left_arm_samples = []
            self._calibration_right_arm_samples = []
            if self._print_calibration_events:
                print(
                    f"[bodytracking_udp] Calibration step {self._calibration_phase_label()} started.",
                    flush=True,
                )

        elapsed = now - self._calibration_started_monotonic
        remaining = self._calibration_delay_s - elapsed
        sample_elapsed = elapsed - self._calibration_sample_start_s
        if sample_elapsed >= 0.0:
            self._calibration_left_samples.append(left_matrix.copy())
            self._calibration_right_samples.append(right_matrix.copy())
            left_arm_points = self._current_arm_points("left")
            right_arm_points = self._current_arm_points("right")
            if left_arm_points is not None:
                self._calibration_left_arm_samples.append(left_arm_points)
            if right_arm_points is not None:
                self._calibration_right_arm_samples.append(right_arm_points)
        if remaining > 0.0:
            if (
                self._print_calibration_events
                and now >= self._next_calibration_prompt_monotonic
            ):
                if sample_elapsed < 0.0:
                    print(
                        f"[bodytracking_udp] Hold calibration step {self._calibration_phase_label()}: "
                        f"{math.ceil(-sample_elapsed)}s until valid recording starts, "
                        f"{math.ceil(remaining)}s until this pose locks.",
                        flush=True,
                    )
                else:
                    print(
                        f"[bodytracking_udp] Recording calibration step {self._calibration_phase_label()}: "
                        f"{math.ceil(remaining)}s remaining, valid_samples={len(self._calibration_left_samples)}.",
                        flush=True,
                    )
                self._next_calibration_prompt_monotonic = (
                    now + self._calibration_prompt_interval_s
                )
            return False

        if not self._calibration_left_samples or not self._calibration_right_samples:
            self._calibration_left_samples.append(left_matrix.copy())
            self._calibration_right_samples.append(right_matrix.copy())

        averaged_left_matrix = _average_pose_matrices(self._calibration_left_samples)
        averaged_right_matrix = _average_pose_matrices(self._calibration_right_samples)
        averaged_left_arm_points = _average_arm_points(
            self._calibration_left_arm_samples
        )
        averaged_right_arm_points = _average_arm_points(
            self._calibration_right_arm_samples
        )

        if self._calibration_phase == "down":
            self._down_left_wrist_matrix = averaged_left_matrix
            self._down_right_wrist_matrix = averaged_right_matrix
            self._down_left_arm_points = averaged_left_arm_points
            self._down_right_arm_points = averaged_right_arm_points
            self._calibration_phase = "forward"
            self._calibration_started_monotonic = None
            self._next_calibration_prompt_monotonic = 0.0
            if self._print_calibration_events:
                print(
                    "[bodytracking_udp] Recorded step 1/2 down pose. "
                    "Step 2/2: raise and stretch both arms horizontally in front of your chest.",
                    flush=True,
                )
                self._print_calibration_pose(
                    "down",
                    self._down_left_wrist_matrix,
                    self._down_right_wrist_matrix,
                    len(self._calibration_left_samples),
                )
            return False

        self._forward_left_wrist_matrix = averaged_left_matrix
        self._forward_right_wrist_matrix = averaged_right_matrix
        self._forward_left_arm_points = averaged_left_arm_points
        self._forward_right_arm_points = averaged_right_arm_points
        self._infer_shoulders_from_two_pose_calibration()
        self._left_calibration_pose_matrix = self._down_left_wrist_matrix.copy()
        self._right_calibration_pose_matrix = self._down_right_wrist_matrix.copy()
        if self._print_calibration_events:
            print("[bodytracking_udp] Recorded step 2/2 forward pose.", flush=True)
            self._print_calibration_pose(
                "forward",
                self._forward_left_wrist_matrix,
                self._forward_right_wrist_matrix,
                len(self._calibration_left_samples),
            )
            self._print_inferred_shoulders()
            print(
                "[bodytracking_udp] Calibration locked. You can start teleoperation now.",
                flush=True,
            )
            self._printed_calibration_locked = True
        return True

    def _advance_auto_start_reference(self) -> bool:
        left_arm_points = self._current_arm_points("left")
        right_arm_points = self._current_arm_points("right")
        left_wrist_matrix, left_wrist_frame_name = self._current_wrist_matrix("left")
        right_wrist_matrix, right_wrist_frame_name = self._current_wrist_matrix("right")
        waist_matrix = self._current_raw_frame_matrix("waist")
        now = time.monotonic()

        waist_required = (
            self._use_waist_frame and self._auto_start_reference_require_waist
        )
        if (
            left_arm_points is None
            or right_arm_points is None
            or left_wrist_matrix is None
            or right_wrist_matrix is None
            or (waist_required and waist_matrix is None)
        ):
            self._reset_auto_reference_sampling()
            if (
                self._print_calibration_events
                and now >= self._next_auto_reference_prompt_monotonic
            ):
                visible_frames = ",".join(sorted(self._frames.keys())) or "none"
                missing = []
                if waist_required and waist_matrix is None:
                    missing.append("waist")
                if left_arm_points is None or left_wrist_matrix is None:
                    missing.append(f"left_shoulder/left_elbow/{left_wrist_frame_name}")
                if right_arm_points is None or right_wrist_matrix is None:
                    missing.append(
                        f"right_shoulder/right_elbow/{right_wrist_frame_name}"
                    )
                print(
                    "[bodytracking_udp] Automatic startup reference: waiting for full-body frames. "
                    f"missing=[{','.join(missing)}] visible=[{visible_frames}]",
                    flush=True,
                )
                self._next_auto_reference_prompt_monotonic = (
                    now + self._calibration_prompt_interval_s
                )
            return False

        if self._auto_reference_started_monotonic is None:
            self._auto_reference_started_monotonic = now
            self._next_auto_reference_prompt_monotonic = 0.0
            self._clear_auto_reference_samples()
            if self._print_calibration_events:
                print(
                    "[bodytracking_udp] Automatic startup reference started. "
                    "Keep the neutral start pose still; robot targets are held at the initial pose.",
                    flush=True,
                )

        elapsed = now - self._auto_reference_started_monotonic
        remaining = self._auto_start_reference_delay_s - elapsed
        sample_elapsed = elapsed - self._auto_start_reference_sample_start_s
        if sample_elapsed >= 0.0:
            self._auto_reference_left_arm_samples.append(
                _copy_arm_points(left_arm_points)
            )
            self._auto_reference_right_arm_samples.append(
                _copy_arm_points(right_arm_points)
            )
            self._auto_reference_left_wrist_samples.append(left_wrist_matrix.copy())
            self._auto_reference_right_wrist_samples.append(right_wrist_matrix.copy())
            if waist_matrix is not None:
                self._auto_reference_waist_samples.append(waist_matrix.copy())

        sample_count = min(
            len(self._auto_reference_left_arm_samples),
            len(self._auto_reference_right_arm_samples),
        )
        if remaining > 0.0:
            if (
                self._print_calibration_events
                and now >= self._next_auto_reference_prompt_monotonic
            ):
                if sample_elapsed < 0.0:
                    print(
                        "[bodytracking_udp] Hold automatic startup reference: "
                        f"{math.ceil(-sample_elapsed)}s until valid recording starts, "
                        f"{math.ceil(remaining)}s until reference check.",
                        flush=True,
                    )
                else:
                    jitter = self._auto_reference_position_std()
                    print(
                        "[bodytracking_udp] Recording automatic startup reference: "
                        f"{math.ceil(remaining)}s remaining, samples={sample_count}, "
                        f"max_position_std={jitter:.4f}m.",
                        flush=True,
                    )
                self._next_auto_reference_prompt_monotonic = (
                    now + self._calibration_prompt_interval_s
                )
            return False

        if sample_count < self._auto_start_reference_min_samples:
            if (
                self._print_calibration_events
                and now >= self._next_auto_reference_prompt_monotonic
            ):
                print(
                    "[bodytracking_udp] Automatic startup reference needs more samples: "
                    f"{sample_count}/{self._auto_start_reference_min_samples}. Keep still.",
                    flush=True,
                )
                self._next_auto_reference_prompt_monotonic = (
                    now + self._calibration_prompt_interval_s
                )
            return False

        jitter = self._auto_reference_position_std()
        if (
            self._auto_start_reference_max_position_std_m > 0.0
            and jitter > self._auto_start_reference_max_position_std_m
        ):
            if self._print_calibration_events:
                print(
                    "[bodytracking_udp] Automatic startup reference was not stable enough "
                    f"(max_position_std={jitter:.4f}m > "
                    f"{self._auto_start_reference_max_position_std_m:.4f}m). Restarting the reference window; "
                    "keep the neutral start pose still.",
                    flush=True,
                )
            self._reset_auto_reference_sampling()
            return False

        self._lock_auto_start_reference(jitter=jitter)
        return True

    def _clear_auto_reference_samples(self) -> None:
        self._auto_reference_left_arm_samples = []
        self._auto_reference_right_arm_samples = []
        self._auto_reference_left_wrist_samples = []
        self._auto_reference_right_wrist_samples = []
        self._auto_reference_waist_samples = []

    def _reset_auto_reference_sampling(self) -> None:
        self._auto_reference_started_monotonic = None
        self._clear_auto_reference_samples()

    def _auto_reference_position_std(self) -> float:
        return max(
            _arm_points_max_position_std(self._auto_reference_left_arm_samples),
            _arm_points_max_position_std(self._auto_reference_right_arm_samples),
            _pose_matrices_max_position_std(self._auto_reference_waist_samples),
        )

    def _lock_auto_start_reference(self, jitter: float) -> None:
        self._auto_reference_left_arm_points = _average_arm_points(
            self._auto_reference_left_arm_samples
        )
        self._auto_reference_right_arm_points = _average_arm_points(
            self._auto_reference_right_arm_samples
        )
        self._auto_reference_left_wrist_matrix = _average_pose_matrices(
            self._auto_reference_left_wrist_samples
        )
        self._auto_reference_right_wrist_matrix = _average_pose_matrices(
            self._auto_reference_right_wrist_samples
        )
        if self._auto_reference_waist_samples:
            self._auto_reference_waist_matrix = _average_pose_matrices(
                self._auto_reference_waist_samples
            )

        self._left_calibration_pose_matrix = (
            self._auto_reference_left_wrist_matrix.copy()
        )
        self._right_calibration_pose_matrix = (
            self._auto_reference_right_wrist_matrix.copy()
        )
        self._auto_reference_started_monotonic = None

        if self._print_calibration_events:
            left_reach = self._reference_reach_length(
                self._auto_reference_left_arm_points
            )
            right_reach = self._reference_reach_length(
                self._auto_reference_right_arm_points
            )
            print(
                "[bodytracking_udp] Automatic startup reference locked. "
                f"samples={len(self._auto_reference_left_arm_samples)}, "
                f"max_position_std={jitter:.4f}m, "
                f"left_reach={left_reach:.3f}m, right_reach={right_reach:.3f}m. "
                "Skeleton visualization and robot teleoperation are now active.",
                flush=True,
            )

    def _reference_reach_length(self, points: dict[str, np.ndarray] | None) -> float:
        if points is None:
            return float("nan")
        return float(np.linalg.norm(points["wrist"] - points["shoulder"]))

    def _source_side_for_target(self, target_side: str) -> str:
        if not self._swap_left_right_wrist_targets:
            return target_side
        return "right" if target_side == "left" else "left"

    def _retarget_target_pair(self, target_side: str) -> tuple[np.ndarray, np.ndarray]:
        source_side = self._source_side_for_target(target_side)
        if self._retargeting_mode == "arm_vector":
            poses = self._retarget_arm_vector_pair(source_side, target_side)
            if poses is not None:
                return poses
            if not self._printed_arm_vector_fallback and self._print_calibration_events:
                print(
                    "[bodytracking_udp] Arm-vector retargeting is waiting for full arm frames "
                    "(shoulder, elbow, wrist); falling back to wrist_delta for now.",
                    flush=True,
                )
                self._printed_arm_vector_fallback = True

        wrist_pose = self._retarget_wrist(source_side, target_side=target_side)
        elbow_pose = (
            self._previous_left_elbow_pose.copy()
            if target_side == "left"
            else self._previous_right_elbow_pose.copy()
        )
        return elbow_pose, wrist_pose

    def _retarget_target(self, target_side: str) -> np.ndarray:
        return self._retarget_target_pair(target_side)[1]

    def _calibration_phase_label(self) -> str:
        if self._calibration_phase == "down":
            return "1/2 arms-down pose"
        return "2/2 chest-forward horizontal pose"

    def _infer_shoulders_from_two_pose_calibration(self) -> None:
        if (
            self._down_left_arm_points is not None
            and self._down_right_arm_points is not None
        ):
            self._inferred_left_shoulder_pos = self._down_left_arm_points[
                "shoulder"
            ].copy()
            self._inferred_right_shoulder_pos = self._down_right_arm_points[
                "shoulder"
            ].copy()
            return

        if (
            self._down_left_wrist_matrix is None
            or self._down_right_wrist_matrix is None
            or self._forward_left_wrist_matrix is None
            or self._forward_right_wrist_matrix is None
        ):
            return

        self._inferred_left_shoulder_pos = self._infer_single_shoulder_position(
            self._down_left_wrist_matrix[:3, 3],
            self._forward_left_wrist_matrix[:3, 3],
        )
        self._inferred_right_shoulder_pos = self._infer_single_shoulder_position(
            self._down_right_wrist_matrix[:3, 3],
            self._forward_right_wrist_matrix[:3, 3],
        )

    def _infer_single_shoulder_position(
        self, down_wrist: np.ndarray, forward_wrist: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [
                down_wrist[0],
                0.5 * (down_wrist[1] + forward_wrist[1]),
                forward_wrist[2],
            ],
            dtype=np.float32,
        )

    def _print_calibration_pose(
        self,
        name: str,
        left_matrix: np.ndarray,
        right_matrix: np.ndarray,
        sample_count: int,
    ) -> None:
        print(
            f"[bodytracking_udp] {name} pose valid_window_samples={sample_count} "
            f"left={np.round(left_matrix[:3, 3], 4).tolist()} "
            f"right={np.round(right_matrix[:3, 3], 4).tolist()}",
            flush=True,
        )

    def _print_inferred_shoulders(self) -> None:
        if (
            self._inferred_left_shoulder_pos is None
            or self._inferred_right_shoulder_pos is None
        ):
            return
        left_down_len = float(
            np.linalg.norm(
                self._down_left_wrist_matrix[:3, 3] - self._inferred_left_shoulder_pos
            )
        )
        left_forward_len = float(
            np.linalg.norm(
                self._forward_left_wrist_matrix[:3, 3]
                - self._inferred_left_shoulder_pos
            )
        )
        right_down_len = float(
            np.linalg.norm(
                self._down_right_wrist_matrix[:3, 3] - self._inferred_right_shoulder_pos
            )
        )
        right_forward_len = float(
            np.linalg.norm(
                self._forward_right_wrist_matrix[:3, 3]
                - self._inferred_right_shoulder_pos
            )
        )
        print(
            "[bodytracking_udp] Inferred shoulders "
            f"left={np.round(self._inferred_left_shoulder_pos, 4).tolist()} "
            f"right={np.round(self._inferred_right_shoulder_pos, 4).tolist()} "
            f"arm_lengths left_down/forward={left_down_len:.3f}/{left_forward_len:.3f} "
            f"right_down/forward={right_down_len:.3f}/{right_forward_len:.3f}",
            flush=True,
        )

    def _receive_latest_packet(self) -> None:
        drained_count = 0
        parsed_body = False
        while True:
            try:
                payload, sender = self._socket.recvfrom(self._cfg.max_packet_bytes)
            except BlockingIOError:
                break
            except OSError as exc:
                logger.warning("UDP body-tracking receive failed: %s", exc)
                break
            drained_count += 1

            try:
                (
                    frames,
                    hand_joints,
                    hand_orientation_deltas,
                    hand_skeleton_positions,
                    packet_time_s,
                    hand_orientation_sample_times_s,
                    hand_orientation_is_forearm_relative,
                ) = self._parse_packet(payload)
            except Exception as exc:
                logger.warning("Failed to parse body-tracking UDP packet: %s", exc)
                if (
                    self._debug_print
                    and self._receive_debug_frame % self._debug_print_interval == 0
                ):
                    print(
                        "[bodytracking_udp debug] "
                        f"failed to parse UDP from {sender}: {exc}. "
                        f"drained_total={self._drained_packet_total + drained_count}",
                        flush=True,
                    )
                continue

            if frames:
                self._frames = frames
                if packet_time_s is None:
                    packet_time_s = time.monotonic()
                if self._body_frame_history:
                    previous_packet_time_s = self._body_frame_history[-1][0]
                    if (
                        packet_time_s <= previous_packet_time_s
                        or packet_time_s - previous_packet_time_s > 1.0
                    ):
                        self._body_frame_history.clear()
                self._body_frame_history.append((packet_time_s, frames))
                self._last_sender = sender
                self._last_packet_time_monotonic = time.monotonic()
                self._printed_stale = False
                parsed_body = True

            if hand_joints is not None:
                self._hand_joint_targets = hand_joints
                self._hand_skeleton_positions = dict(hand_skeleton_positions)
                self._last_hand_sender = sender
                self._last_hand_packet_time_monotonic = time.monotonic()
                if self._hand_only_mode:
                    self._printed_stale = False
                if not self._printed_hand_targets_active:
                    print(
                        "[bodytracking_udp] Receiving hand joint targets "
                        f"({self._hand_joint_count} values) from {sender}.",
                        flush=True,
                    )
                    self._printed_hand_targets_active = True

            if hand_orientation_deltas:
                self._hand_orientation_delta_matrices.update(hand_orientation_deltas)
                self._hand_orientation_is_forearm_relative = (
                    hand_orientation_is_forearm_relative
                )
                self._hand_orientation_sample_times_s.update(
                    hand_orientation_sample_times_s
                )
                self._last_hand_orientation_sender = sender
                self._last_hand_orientation_packet_time_monotonic = time.monotonic()
                self._printed_hand_imu_stale = False
                if not self._printed_hand_orientations_active:
                    print(
                        "[bodytracking_udp] Receiving calibrated hand IMU orientation deltas "
                        f"for {sorted(hand_orientation_deltas.keys())} from {sender}.",
                        flush=True,
                    )
                    self._printed_hand_orientations_active = True

            if (
                not frames
                and hand_joints is None
                and not hand_orientation_deltas
                and not hand_skeleton_positions
            ):
                self._empty_packet_total += 1
                if (
                    self._debug_print
                    and self._receive_debug_frame % self._debug_print_interval == 0
                ):
                    payload_keys, frame_keys, wrist_map = _packet_debug_summary(payload)
                    print(
                        "[bodytracking_udp debug] "
                        f"received UDP from {sender} but parsed no known frames or hand joints. "
                        f"payload_keys=[{payload_keys}] frame_keys=[{frame_keys}] wrist_map=[{wrist_map}] "
                        f"empty_total={self._empty_packet_total} drained_total={self._drained_packet_total + drained_count}",
                        flush=True,
                    )

        self._drained_packet_total += drained_count

        if self._hand_only_mode:
            if (
                self._last_hand_packet_time_monotonic is None
                and not self._printed_waiting
            ):
                print(
                    "[bodytracking_udp] Waiting for SenseGlove hand-joint and IMU packets on "
                    f"{self._bind_host}:{self._bind_port}.",
                    flush=True,
                )
                self._printed_waiting = True
            elif self._last_hand_packet_time_monotonic is not None:
                self._printed_waiting = False
        elif self._last_packet_time_monotonic is None and not self._printed_waiting:
            print(
                "[bodytracking_udp] Waiting for body-tracking packets. Start scripts/run_xrobotoolkit_body_to_esrobo_teleop_bridge.sh "
                "for arm-vector retargeting, or send body JSON poses to UDP "
                f"{self._bind_host}:{self._bind_port}.",
                flush=True,
            )
            self._printed_waiting = True

        if parsed_body:
            self._printed_waiting = False
        self._receive_debug_frame += 1

    def _parse_packet(self, payload: bytes) -> tuple[
        dict[str, np.ndarray],
        np.ndarray | None,
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        float | None,
        dict[str, float],
        bool,
    ]:
        message = json.loads(payload.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError(f"expected JSON object, got {type(message).__name__}")
        frames: dict[str, np.ndarray] = {}

        if "joint_names" in message and "joint_positions" in message:
            frames.update(self._parse_full_body_arrays(message))

        frame_payload = message.get("frames")
        if isinstance(frame_payload, dict):
            for raw_name, raw_pose in frame_payload.items():
                frame_name = BODY_FRAME_ALIASES.get(str(raw_name).lower())
                if frame_name is None:
                    continue
                pose = self._parse_pose(raw_pose)
                if pose is not None:
                    frames[frame_name] = self._transform_source_pose_to_robot_frame(
                        pose
                    )

        return (
            frames,
            self._parse_hand_joints(message),
            self._parse_hand_orientation_deltas(message),
            self._parse_hand_skeleton_positions(message),
            _body_packet_time_seconds(message),
            self._parse_hand_orientation_sample_times(message),
            bool(message.get("hand_orientation_is_forearm_relative", False)),
        )

    @staticmethod
    def _parse_hand_orientation_sample_times(
        message: dict[str, Any]
    ) -> dict[str, float]:
        raw = message.get("hand_orientation_sample_monotonic_ns")
        if not isinstance(raw, dict):
            packet_time = _body_packet_time_seconds(message)
            return (
                {side: packet_time for side in ("left", "right")}
                if packet_time is not None
                else {}
            )
        parsed: dict[str, float] = {}
        for side in ("left", "right"):
            value = raw.get(side)
            if value is None:
                continue
            try:
                parsed[side] = float(value) / 1.0e9
            except (TypeError, ValueError):
                continue
        return parsed

    def _parse_hand_skeleton_positions(
        self, message: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        raw_skeleton = message.get("hand_skeleton_positions")
        if not isinstance(raw_skeleton, dict):
            return {}
        unit = str(message.get("hand_skeleton_unit", "mm")).strip().lower()
        self._hand_skeleton_frame = (
            str(message.get("hand_skeleton_frame", "source")).strip().lower()
        )
        unit_scale = 0.001 if unit in ("mm", "millimeter", "millimeters") else 1.0
        parsed: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            raw_points = raw_skeleton.get(side)
            if raw_points is None:
                continue
            points = np.asarray(raw_points, dtype=np.float32)
            if points.shape != (20, 3):
                raise ValueError(
                    f"{side} hand skeleton must have shape (20, 3), got {points.shape}"
                )
            if not np.all(np.isfinite(points)):
                raise ValueError(f"{side} hand skeleton contains non-finite values")
            parsed[side] = points * unit_scale
        return parsed

    def _parse_hand_joints(self, message: dict[str, Any]) -> np.ndarray | None:
        raw_targets = message.get("hand_joints")
        if raw_targets is None:
            raw_targets = message.get("hand_joints_robot_order")

        if isinstance(raw_targets, dict):
            left_targets = raw_targets.get("left")
            right_targets = raw_targets.get("right")
            if left_targets is None or right_targets is None:
                return None
            raw_targets = list(left_targets) + list(right_targets)
        elif raw_targets is None:
            left_targets = message.get("left_hand_joints")
            right_targets = message.get("right_hand_joints")
            if left_targets is None or right_targets is None:
                return None
            raw_targets = list(left_targets) + list(right_targets)

        targets = np.asarray(raw_targets, dtype=np.float32).reshape(-1)
        if targets.shape[0] != self._hand_joint_count:
            raise ValueError(
                f"hand_joints length must be {self._hand_joint_count}, got {targets.shape[0]}"
            )
        if not np.all(np.isfinite(targets)):
            raise ValueError("hand_joints contains non-finite values")
        return targets.astype(np.float32)

    def _parse_hand_orientation_deltas(
        self, message: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        if not self._use_hand_imu_orientation:
            return {}

        raw_deltas = message.get("hand_orientation_deltas")
        if raw_deltas is None:
            raw_deltas = message.get("hand_orientation_delta")

        axis_calibrated = bool(message.get("hand_orientation_axis_calibrated", False))
        absolute = message.get("hand_orientations_absolute")
        neutral = message.get("hand_orientation_neutral")
        if (
            not axis_calibrated
            and isinstance(absolute, dict)
            and isinstance(neutral, dict)
        ):
            raw_deltas = {}
            for side in ("left", "right"):
                if side not in absolute or side not in neutral:
                    continue
                current_quat = self._parse_hand_orientation_quat(
                    absolute[side], message
                )
                neutral_quat = self._parse_hand_orientation_quat(
                    neutral[side], message
                )
                raw_deltas[side] = _matrix_to_quat_wxyz(
                    _quat_wxyz_to_matrix(current_quat)
                    @ _quat_wxyz_to_matrix(neutral_quat).T
                )

        sides: dict[str, Any] = {}
        if isinstance(raw_deltas, dict):
            for side in ("left", "right"):
                if side in raw_deltas:
                    sides[side] = raw_deltas[side]
        elif raw_deltas is not None:
            raw_array = np.asarray(raw_deltas, dtype=np.float32).reshape(-1)
            if raw_array.shape[0] == 8:
                sides["left"] = raw_array[:4]
                sides["right"] = raw_array[4:]
            else:
                raise ValueError(
                    f"hand_orientation_deltas must contain 8 numbers, got {raw_array.shape[0]}"
                )
        else:
            for side in ("left", "right"):
                value = message.get(f"{side}_hand_orientation_delta")
                if value is not None:
                    sides[side] = value

        orientation_deltas: dict[str, np.ndarray] = {}
        for side, raw_quat in sides.items():
            quat = self._parse_hand_orientation_quat(raw_quat, message)
            source_rotation_delta = _quat_wxyz_to_matrix(quat)
            initial_rotation = (
                self._initial_left_pose_matrix[:3, :3]
                if side == "left"
                else self._initial_right_pose_matrix[:3, :3]
            )
            if axis_calibrated:
                robot_rotation_delta = (
                    initial_rotation
                    @ source_rotation_delta
                    @ initial_rotation.T
                )
                orientation_deltas[side] = robot_rotation_delta.astype(np.float32)
                continue
            robot_rotation_delta = (
                self._hand_imu_source_to_robot_rotation
                @ source_rotation_delta
                @ self._hand_imu_source_to_robot_rotation.T
            )
            if self._is_real_senseglove_orientation(message):
                local_axis_signs = (
                    self._hand_imu_left_local_axis_signs
                    if side == "left"
                    else self._hand_imu_right_local_axis_signs
                )
                local_rotation_delta = (
                    initial_rotation.T
                    @ robot_rotation_delta
                    @ initial_rotation
                )
                local_rotvec = _rotation_matrix_to_rotvec(local_rotation_delta)
                if side == "left":
                    local_rotvec = local_rotvec[self._hand_imu_left_axis_order]
                elif self._hand_imu_right_swap_xy:
                    local_rotvec[[0, 1]] = local_rotvec[[1, 0]]
                local_rotvec *= local_axis_signs
                local_rotation_delta = _rotvec_to_rotation_matrix(local_rotvec)
                robot_rotation_delta = (
                    initial_rotation
                    @ local_rotation_delta
                    @ initial_rotation.T
                )
            orientation_deltas[side] = robot_rotation_delta.astype(np.float32)
        return orientation_deltas

    @staticmethod
    def _is_real_senseglove_orientation(message: dict[str, Any]) -> bool:
        if str(message.get("source", "")).strip().lower() == "senseglove_ros":
            return True
        metadata = message.get("hand_orientation_retargeting")
        if not isinstance(metadata, dict):
            return False
        return (
            str(metadata.get("method", "")).strip().lower()
            == "calibrated_senseglove_imu_delta"
        )

    def _parse_hand_orientation_quat(
        self, raw_quat: Any, message: dict[str, Any]
    ) -> np.ndarray:
        order = str(message.get("hand_orientation_delta_order", "wxyz")).strip().lower()
        if isinstance(raw_quat, dict):
            if "quat_wxyz" in raw_quat:
                quat = raw_quat["quat_wxyz"]
                order = "wxyz"
            elif "quat_xyzw" in raw_quat:
                quat = raw_quat["quat_xyzw"]
                order = "xyzw"
            elif "orientation" in raw_quat:
                quat = raw_quat["orientation"]
            elif "quat" in raw_quat:
                quat = raw_quat["quat"]
            else:
                raise ValueError("hand orientation delta dict has no quaternion field")
        else:
            quat = raw_quat

        quat_array = np.asarray(quat, dtype=np.float32).reshape(-1)
        if quat_array.shape[0] != 4:
            raise ValueError(
                f"hand orientation delta quaternion must have 4 numbers, got {quat_array.shape[0]}"
            )
        if order == "xyzw":
            quat_array = np.asarray(
                [quat_array[3], quat_array[0], quat_array[1], quat_array[2]],
                dtype=np.float32,
            )
        elif order != "wxyz":
            raise ValueError(f"unsupported hand_orientation_delta_order={order!r}")
        if not np.all(np.isfinite(quat_array)):
            raise ValueError("hand orientation delta contains non-finite values")
        return _normalize_quat_wxyz(quat_array)

    def _parse_full_body_arrays(self, message: dict[str, Any]) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        joint_names = message["joint_names"]
        joint_positions = np.asarray(message["joint_positions"], dtype=np.float32)
        joint_orientations = np.asarray(
            message.get("joint_orientations", []), dtype=np.float32
        )
        joint_valid = np.asarray(
            message.get("joint_valid", np.ones(len(joint_names))), dtype=bool
        )

        for index, raw_name in enumerate(joint_names):
            if (
                index >= len(joint_positions)
                or index >= len(joint_valid)
                or not joint_valid[index]
            ):
                continue
            frame_name = BODY_FRAME_ALIASES.get(str(raw_name).lower())
            if frame_name is None:
                continue
            if joint_orientations.shape[0] > index and joint_orientations.shape[1] >= 4:
                quat_xyzw = joint_orientations[index, :4]
                quat_wxyz = np.asarray(
                    [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                    dtype=np.float32,
                )
            else:
                quat_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            pose = np.concatenate([joint_positions[index, :3], quat_wxyz]).astype(
                np.float32
            )
            frames[frame_name] = self._transform_source_pose_to_robot_frame(pose)
        return frames

    def _parse_pose(self, raw_pose: Any) -> np.ndarray | None:
        if isinstance(raw_pose, dict) and not raw_pose.get("valid", True):
            return None

        if isinstance(raw_pose, dict):
            position = raw_pose.get("pos", raw_pose.get("position"))
            if "quat_wxyz" in raw_pose:
                quat = raw_pose["quat_wxyz"]
            elif "quat_xyzw" in raw_pose:
                quat_xyzw = raw_pose["quat_xyzw"]
                quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            elif "quat" in raw_pose:
                quat = raw_pose["quat"]
                if self._cfg.packet_quaternion_order == "xyzw":
                    quat = [quat[3], quat[0], quat[1], quat[2]]
            elif "orientation" in raw_pose:
                quat = raw_pose["orientation"]
                if self._cfg.packet_quaternion_order == "xyzw":
                    quat = [quat[3], quat[0], quat[1], quat[2]]
            else:
                quat = [1.0, 0.0, 0.0, 0.0]
            if position is None:
                return None
            pose = np.concatenate(
                [
                    np.asarray(position, dtype=np.float32),
                    np.asarray(quat, dtype=np.float32),
                ]
            )
        else:
            pose = np.asarray(raw_pose, dtype=np.float32)
            if pose.shape[0] != 7:
                return None
            if self._cfg.packet_quaternion_order == "xyzw":
                pose = np.asarray(
                    [pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]],
                    dtype=np.float32,
                )

        if pose.shape[0] != 7 or not np.all(np.isfinite(pose)):
            return None
        pose[3:] = _normalize_quat_wxyz(pose[3:])
        return pose.astype(np.float32)

    def _transform_source_pose_to_robot_frame(self, pose: np.ndarray) -> np.ndarray:
        position = self._source_to_robot_rotation @ pose[:3]
        rotation = (
            self._source_to_robot_rotation
            @ _quat_wxyz_to_matrix(pose[3:])
            @ self._source_to_robot_rotation.T
        )
        quat = _matrix_to_quat_wxyz(rotation)
        return np.concatenate([position, quat]).astype(np.float32)

    def _has_fresh_packet(self) -> bool:
        if self._last_packet_time_monotonic is None:
            return False
        age = time.monotonic() - self._last_packet_time_monotonic
        if age > self._max_stale_time_s:
            if not self._printed_stale:
                print(
                    f"[bodytracking_udp] Packet stream stale for {age:.2f}s; holding previous robot target.",
                    flush=True,
                )
                self._printed_stale = True
            return False
        return True

    def _current_wrist_matrix(self, side: str) -> tuple[np.ndarray | None, str]:
        frame_name = f"{side}_wrist"
        wrist_matrix = self._current_frame_matrix(frame_name)
        if wrist_matrix is None:
            frame_name = f"{side}_hand"
            wrist_matrix = self._current_frame_matrix(frame_name)
        if wrist_matrix is None:
            return None, frame_name
        return wrist_matrix, frame_name

    def _current_frame_matrix(self, frame_name: str) -> np.ndarray | None:
        pose = self._frames.get(frame_name)
        if pose is None:
            return None
        matrix = _pose_array_to_matrix(pose)
        if self._use_waist_frame and "waist" in self._frames:
            waist_matrix = _pose_array_to_matrix(self._frames["waist"])
            matrix = _invert_pose_matrix(waist_matrix) @ matrix
        return matrix

    def _current_raw_frame_matrix(self, frame_name: str) -> np.ndarray | None:
        pose = self._frames.get(frame_name)
        if pose is None:
            return None
        return _pose_array_to_matrix(pose)

    def _current_arm_points(self, side: str) -> dict[str, np.ndarray] | None:
        shoulder_matrix = self._current_frame_matrix(f"{side}_shoulder")
        elbow_matrix = self._current_frame_matrix(f"{side}_elbow")
        wrist_matrix, _frame_name = self._current_wrist_matrix(side)
        if shoulder_matrix is None or elbow_matrix is None or wrist_matrix is None:
            return None
        return {
            "shoulder": shoulder_matrix[:3, 3].copy(),
            "elbow": elbow_matrix[:3, 3].copy(),
            "wrist": wrist_matrix[:3, 3].copy(),
        }

    def _predicted_arm_points(self, side: str) -> dict[str, np.ndarray] | None:
        """Predict a bounded short distance ahead from recent body packets."""
        current_points = self._current_arm_points(side)
        if current_points is None or self._arm_vector_prediction_horizon_s <= 0.0:
            return current_points
        if len(self._body_frame_history) < 2:
            return current_points

        current_time_s = self._body_frame_history[-1][0]
        candidates = [
            sample
            for sample in self._body_frame_history
            if 0.008 <= current_time_s - sample[0] <= 0.10
        ]
        if not candidates:
            return current_points
        previous_time_s, previous_frames = min(
            candidates,
            key=lambda sample: abs(
                (current_time_s - sample[0]) - self._arm_vector_prediction_lookback_s
            ),
        )
        sample_dt = current_time_s - previous_time_s
        previous_points = self._arm_points_from_frames(previous_frames, side)
        if previous_points is None or sample_dt <= 1.0e-6:
            return current_points

        predicted_points: dict[str, np.ndarray] = {}
        for point_name in ("shoulder", "elbow", "wrist"):
            velocity = (
                current_points[point_name] - previous_points[point_name]
            ) / sample_dt
            velocity_norm = float(np.linalg.norm(velocity))
            if self._arm_vector_prediction_max_velocity_m_s <= 0.0:
                velocity.fill(0.0)
            elif velocity_norm > self._arm_vector_prediction_max_velocity_m_s:
                velocity *= self._arm_vector_prediction_max_velocity_m_s / velocity_norm

            displacement = velocity * self._arm_vector_prediction_horizon_s
            displacement_norm = float(np.linalg.norm(displacement))
            if self._arm_vector_prediction_max_displacement_m <= 0.0:
                displacement.fill(0.0)
            elif displacement_norm > self._arm_vector_prediction_max_displacement_m:
                displacement *= (
                    self._arm_vector_prediction_max_displacement_m / displacement_norm
                )
            predicted_points[point_name] = (
                current_points[point_name] + displacement
            ).astype(np.float32)
        return predicted_points

    def _arm_points_from_frames(
        self,
        frames: dict[str, np.ndarray],
        side: str,
    ) -> dict[str, np.ndarray] | None:
        """Extract one arm's points from an arbitrary packet in the live retargeting frame."""
        waist_inverse = None
        if self._use_waist_frame and "waist" in frames:
            waist_inverse = _invert_pose_matrix(_pose_array_to_matrix(frames["waist"]))

        points: dict[str, np.ndarray] = {}
        for point_name in ("shoulder", "elbow", "wrist"):
            pose = frames.get(f"{side}_{point_name}")
            if pose is None and point_name == "wrist":
                pose = frames.get(f"{side}_hand")
            if pose is None:
                return None
            matrix = _pose_array_to_matrix(pose)
            if waist_inverse is not None:
                matrix = waist_inverse @ matrix
            points[point_name] = matrix[:3, 3].copy()
        return points

    def _hand_imu_delta_matrix_for_target(self, target_side: str) -> np.ndarray | None:
        if not self._use_hand_imu_orientation:
            return None
        if self._last_hand_orientation_packet_time_monotonic is None:
            return None
        age = time.monotonic() - self._last_hand_orientation_packet_time_monotonic
        if age > self._hand_imu_max_stale_time_s:
            if not self._printed_hand_imu_stale and self._print_calibration_events:
                print(
                    f"[bodytracking_udp] Hand IMU orientation stream stale for {age:.2f}s; "
                    "falling back to the arm-parent wrist orientation.",
                    flush=True,
                )
                self._printed_hand_imu_stale = True
            return None
        return self._hand_orientation_delta_matrices.get(target_side)

    def _hand_imu_relative_rotation_world(
        self,
        target_side: str,
        current_arm_rotation: np.ndarray | None,
    ) -> np.ndarray | None:
        """Return the forearm-relative hand IMU rotation since the sync lock.

        Locks the synchronized arm/IMU reference on the first call. Returns
        ``None`` when the IMU stream is inactive or forearm-relative mode is
        disabled.
        """
        rotation_delta = self._hand_imu_delta_matrix_for_target(target_side)
        synchronized_arm_rotation = self._synchronized_human_forearm_rotation(
            target_side
        )
        if synchronized_arm_rotation is not None:
            current_arm_rotation = synchronized_arm_rotation
        if (
            rotation_delta is None
            or not self._hand_imu_relative_to_arm_frame
            or current_arm_rotation is None
        ):
            return None

        arm_reference = self._hand_arm_reference_rotations.get(target_side)
        imu_reference = self._hand_imu_reference_delta_matrices.get(target_side)
        if arm_reference is None or imu_reference is None:
            self._hand_arm_reference_rotations[target_side] = (
                current_arm_rotation.copy()
            )
            self._hand_imu_reference_delta_matrices[target_side] = (
                rotation_delta.copy()
            )
            return np.eye(3, dtype=np.float32)

        parent_delta_world = current_arm_rotation @ arm_reference.T
        hand_delta_world = rotation_delta @ imu_reference.T
        if self._hand_orientation_is_forearm_relative:
            return hand_delta_world.astype(np.float32)
        return (parent_delta_world.T @ hand_delta_world).astype(np.float32)

    def _synchronized_human_forearm_rotation(
        self, target_side: str
    ) -> np.ndarray | None:
        """Build the human forearm frame nearest the corresponding IMU sample."""
        sample_time = self._hand_orientation_sample_times_s.get(target_side)
        source_side = self._source_side_for_target(target_side)
        points = (
            self._interpolated_arm_points_at_time(source_side, sample_time)
            if sample_time is not None
            else None
        )
        if points is None:
            points = self._current_arm_points(source_side)
        if points is None:
            return None
        points = _arm_points_relative_to_shoulder(points, self._position_delta_signs)
        key = f"imu_{target_side}"
        rotation = _forearm_points_to_rotation(
            points, self._forearm_plane_normals.get(key)
        )
        if rotation is not None:
            self._forearm_plane_normals[key] = rotation[:, 2].copy()
        return rotation

    def _interpolated_arm_points_at_time(
        self, source_side: str, sample_time: float
    ) -> dict[str, np.ndarray] | None:
        """Interpolate body arm positions onto a hand-IMU sample timestamp."""
        history = list(self._body_frame_history)
        if not history:
            return None
        before = [sample for sample in history if sample[0] <= sample_time]
        after = [sample for sample in history if sample[0] >= sample_time]
        if before and after:
            before_time, before_frames = before[-1]
            after_time, after_frames = after[0]
            if after_time - before_time <= 0.10:
                before_points = self._arm_points_from_frames(
                    before_frames, source_side
                )
                after_points = self._arm_points_from_frames(after_frames, source_side)
                if before_points is not None and after_points is not None:
                    denominator = max(after_time - before_time, 1.0e-9)
                    alpha = float(np.clip((sample_time - before_time) / denominator, 0.0, 1.0))
                    return {
                        name: (
                            (1.0 - alpha) * before_points[name]
                            + alpha * after_points[name]
                        ).astype(np.float32)
                        for name in ("shoulder", "elbow", "wrist")
                    }
        nearest_time, nearest_frames = min(
            history, key=lambda sample: abs(sample[0] - sample_time)
        )
        if abs(nearest_time - sample_time) > 0.25:
            return None
        return self._arm_points_from_frames(nearest_frames, source_side)

    def _compose_wrist_rotation(
        self,
        target_side: str,
        target_arm_points: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Combine the arm-parent frame with the hand's IMU-relative rotation."""
        if target_side == "left":
            initial_rotation = self._initial_left_pose_matrix[:3, :3]
        else:
            initial_rotation = self._initial_right_pose_matrix[:3, :3]

        current_arm_rotation = None
        if target_arm_points is not None:
            key = f"target_{target_side}"
            current_arm_rotation = _forearm_points_to_rotation(
                target_arm_points, self._forearm_plane_normals.get(key)
            )
            if current_arm_rotation is not None:
                self._forearm_plane_normals[key] = current_arm_rotation[:, 2].copy()
        rotation_delta = self._hand_imu_delta_matrix_for_target(target_side)

        arm_mounting_reference = self._wrist_arm_reference_rotations.get(
            target_side
        )
        if current_arm_rotation is not None and arm_mounting_reference is None:
            arm_mounting_reference = current_arm_rotation.copy()
            self._wrist_arm_reference_rotations[target_side] = arm_mounting_reference

        arm_follow_rotation = initial_rotation.copy()
        if current_arm_rotation is not None and arm_mounting_reference is not None:
            arm_follow_rotation = (
                current_arm_rotation
                @ arm_mounting_reference.T
                @ initial_rotation
            ).astype(np.float32)

        relative_rotation_world = self._hand_imu_relative_rotation_world(
            target_side, current_arm_rotation
        )
        if relative_rotation_world is not None and arm_mounting_reference is not None:
            arm_reference = self._hand_arm_reference_rotations[target_side]
            relative_rotation_local = (
                arm_reference.T @ relative_rotation_world @ arm_reference
            )
            return (
                current_arm_rotation
                @ relative_rotation_local
                @ arm_mounting_reference.T
                @ initial_rotation
            ).astype(np.float32)

        if rotation_delta is not None:
            return (rotation_delta @ initial_rotation).astype(np.float32)

        if not self._hand_orientation_inherit_arm_frame:
            return initial_rotation.copy().astype(np.float32)
        return arm_follow_rotation

    def _retarget_arm_vector_pair(
        self, source_side: str, target_side: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        current_points = self._predicted_arm_points(source_side)
        if current_points is None:
            return None

        wrist_matrix, _frame_name = self._current_wrist_matrix(source_side)
        if wrist_matrix is None:
            return None

        initial_wrist_pose = (
            self._initial_left_pose_matrix
            if target_side == "left"
            else self._initial_right_pose_matrix
        )
        initial_elbow_pose = (
            self._initial_left_elbow_pose_matrix
            if target_side == "left"
            else self._initial_right_elbow_pose_matrix
        )
        target_wrist_pose = initial_wrist_pose.copy()
        target_elbow_pose = initial_elbow_pose.copy()

        current_reach = current_points["wrist"] - current_points["shoulder"]
        current_upper_arm = current_points["elbow"] - current_points["shoulder"]
        calibration_points: dict[str, np.ndarray] | None = None
        wrist_calibration_pose: np.ndarray | None = None
        auto_reference_points = (
            self._auto_reference_left_arm_points
            if source_side == "left"
            else self._auto_reference_right_arm_points
        )
        if self._requires_calibration:
            calibration_points = (
                self._down_left_arm_points
                if source_side == "left"
                else self._down_right_arm_points
            )
            if calibration_points is None:
                return None

            wrist_calibration_pose = (
                self._left_calibration_pose_matrix
                if source_side == "left"
                else self._right_calibration_pose_matrix
            )
            if wrist_calibration_pose is None:
                return None

        reference_points = (
            calibration_points if self._requires_calibration else auto_reference_points
        )
        if self._arm_vector_position_mode in (
            "segment_direction_absolute",
            "segment_direction_relative",
        ):
            segment_positions = self._retarget_arm_segment_direction_positions(
                current_points,
                reference_points,
                target_side,
            )
            if segment_positions is not None:
                target_elbow_pose[:3, 3], target_wrist_pose[:3, 3] = segment_positions
            else:
                return None
        elif self._requires_calibration:
            calibration_reach = (
                calibration_points["wrist"] - calibration_points["shoulder"]
            )
            calibration_upper_arm = (
                calibration_points["elbow"] - calibration_points["shoulder"]
            )
            position_delta = (
                current_reach - calibration_reach
            ) * self._position_delta_signs
            elbow_position_delta = (
                current_upper_arm - calibration_upper_arm
            ) * self._position_delta_signs
            target_wrist_pose[:3, 3] = (
                initial_wrist_pose[:3, 3] + position_delta * self._position_scale
            )
            target_elbow_pose[:3, 3] = (
                initial_elbow_pose[:3, 3] + elbow_position_delta * self._position_scale
            )
        elif auto_reference_points is not None:
            reference_reach = (
                auto_reference_points["wrist"] - auto_reference_points["shoulder"]
            )
            reference_upper_arm = (
                auto_reference_points["elbow"] - auto_reference_points["shoulder"]
            )
            position_delta = (
                current_reach - reference_reach
            ) * self._position_delta_signs
            elbow_position_delta = (
                current_upper_arm - reference_upper_arm
            ) * self._position_delta_signs
            target_wrist_pose[:3, 3] = (
                initial_wrist_pose[:3, 3] + position_delta * self._position_scale
            )
            target_elbow_pose[:3, 3] = (
                initial_elbow_pose[:3, 3] + elbow_position_delta * self._position_scale
            )
        else:
            robot_shoulder = (
                self._robot_left_shoulder_position
                if target_side == "left"
                else self._robot_right_shoulder_position
            )
            scaled_reach = (
                current_reach * self._position_delta_signs * self._position_scale
            )
            scaled_reach = self._clamp_reach_vector(scaled_reach)
            scaled_upper_arm = (
                current_upper_arm * self._position_delta_signs * self._position_scale
            )
            target_wrist_pose[:3, 3] = robot_shoulder + scaled_reach
            target_elbow_pose[:3, 3] = robot_shoulder + scaled_upper_arm

        robot_shoulder = (
            self._robot_left_shoulder_position
            if target_side == "left"
            else self._robot_right_shoulder_position
        )
        target_wrist_pose[:3, :3] = self._compose_wrist_rotation(
            target_side,
            {
                "shoulder": robot_shoulder,
                "elbow": target_elbow_pose[:3, 3],
                "wrist": target_wrist_pose[:3, 3],
            },
        )

        elbow_pose = _pose_matrix_to_array(target_elbow_pose)
        wrist_pose = _pose_matrix_to_array(target_wrist_pose)
        if target_side == "left":
            self._previous_left_elbow_pose = elbow_pose
            self._previous_left_pose = wrist_pose
        else:
            self._previous_right_elbow_pose = elbow_pose
            self._previous_right_pose = wrist_pose
        return elbow_pose, wrist_pose

    def _retarget_arm_segment_direction_positions(
        self,
        current_points: dict[str, np.ndarray],
        reference_points: dict[str, np.ndarray] | None,
        target_side: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Retarget human upper-arm/forearm directions to robot segment lengths."""
        robot_shoulder, robot_elbow, robot_wrist, robot_reference_rotation = (
            self._robot_arm_reference(target_side)
        )
        upper_length, forearm_length = self._robot_arm_segment_lengths(target_side)

        signed_current = _arm_points_relative_to_shoulder(
            current_points, self._position_delta_signs
        )
        current_upper = signed_current["elbow"] - signed_current["shoulder"]
        current_forearm = signed_current["wrist"] - signed_current["elbow"]

        alignment_rotation = np.eye(3, dtype=np.float32)
        if (
            self._arm_vector_position_mode == "segment_direction_relative"
            and reference_points is not None
        ):
            signed_reference = _arm_points_relative_to_shoulder(
                reference_points, self._position_delta_signs
            )
            reference_rotation = _arm_points_to_rotation(signed_reference)
            if reference_rotation is not None and robot_reference_rotation is not None:
                alignment_rotation = robot_reference_rotation @ reference_rotation.T

        mapped_upper = alignment_rotation @ current_upper
        mapped_forearm = alignment_rotation @ current_forearm
        mapped_upper = self._apply_segment_direction_deadband(
            target_side,
            "upper_arm",
            mapped_upper,
            self._upper_arm_angular_deadband_deg,
        )
        mapped_forearm = self._apply_segment_direction_deadband(
            target_side,
            "forearm",
            mapped_forearm,
            self._forearm_angular_deadband_deg,
        )
        upper_vector = _scale_vector_to_length(
            mapped_upper, upper_length, robot_elbow - robot_shoulder
        )
        forearm_vector = _scale_vector_to_length(
            mapped_forearm, forearm_length, robot_wrist - robot_elbow
        )
        if upper_vector is None or forearm_vector is None:
            return None

        target_elbow_position = robot_shoulder + upper_vector
        target_wrist_position = target_elbow_position + forearm_vector
        return target_elbow_position.astype(np.float32), target_wrist_position.astype(
            np.float32
        )

    def _apply_segment_direction_deadband(
        self,
        target_side: str,
        segment_name: str,
        vector: np.ndarray,
        deadband_deg: float,
    ) -> np.ndarray:
        """Suppress tiny jitter while retaining a continuous response to real motion."""
        direction = _normalize_vector(vector)
        if direction is None:
            return vector

        previous_direction = self._held_segment_directions[target_side].get(
            segment_name
        )
        if previous_direction is not None and deadband_deg > 0.0:
            cosine = float(np.clip(np.dot(previous_direction, direction), -1.0, 1.0))
            angular_change_deg = float(np.degrees(np.arccos(cosine)))
            if angular_change_deg < deadband_deg:
                blend = float(np.clip(angular_change_deg / deadband_deg, 0.08, 1.0))
                blended = _normalize_vector(
                    (1.0 - blend) * previous_direction + blend * direction
                )
                if blended is not None:
                    direction = blended

        self._held_segment_directions[target_side][segment_name] = direction.copy()
        return direction

    def _robot_arm_reference(
        self, target_side: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if target_side == "left":
            return (
                self._robot_left_shoulder_position,
                self._robot_left_elbow_position,
                self._initial_left_pose_matrix[:3, 3],
                self._robot_left_reference_rotation,
            )
        return (
            self._robot_right_shoulder_position,
            self._robot_right_elbow_position,
            self._initial_right_pose_matrix[:3, 3],
            self._robot_right_reference_rotation,
        )

    def _robot_arm_segment_lengths(self, target_side: str) -> tuple[float, float]:
        if target_side == "left":
            return self._robot_left_upper_arm_length, self._robot_left_forearm_length
        return self._robot_right_upper_arm_length, self._robot_right_forearm_length

    def _retarget_arm_vector(
        self, source_side: str, target_side: str
    ) -> np.ndarray | None:
        poses = self._retarget_arm_vector_pair(source_side, target_side)
        if poses is None:
            return None
        return poses[1]

    def _clamp_reach_vector(self, reach: np.ndarray) -> np.ndarray:
        if self._arm_vector_max_reach <= 0.0:
            return reach
        reach_norm = float(np.linalg.norm(reach))
        if reach_norm <= self._arm_vector_max_reach or reach_norm < 1.0e-6:
            return reach
        return (reach / reach_norm * self._arm_vector_max_reach).astype(np.float32)

    def _retarget_wrist(
        self, source_side: str, target_side: str | None = None
    ) -> np.ndarray:
        target_side = source_side if target_side is None else target_side
        wrist_matrix, frame_name = self._current_wrist_matrix(source_side)
        if wrist_matrix is None:
            return (
                self._previous_left_pose.copy()
                if target_side == "left"
                else self._previous_right_pose.copy()
            )

        if source_side == "left":
            calibration_pose = self._left_calibration_pose_matrix
        else:
            calibration_pose = self._right_calibration_pose_matrix

        if target_side == "left":
            initial_pose = self._initial_left_pose_matrix
        else:
            initial_pose = self._initial_right_pose_matrix

        if calibration_pose is None:
            if source_side == "left":
                self._left_calibration_pose_matrix = wrist_matrix.copy()
            else:
                self._right_calibration_pose_matrix = wrist_matrix.copy()
            if self._print_calibration_events:
                print(
                    f"[bodytracking_udp] Calibrated human_{source_side} wrist for robot_{target_side} target at "
                    f"{np.round(wrist_matrix[:3, 3], 4).tolist()} in "
                    f"{'waist' if self._use_waist_frame and 'waist' in self._frames else 'source'} frame "
                    f"from {frame_name}.",
                    flush=True,
                )
            return (
                self._initial_left_pose.copy()
                if target_side == "left"
                else self._initial_right_pose.copy()
            )

        target_pose = initial_pose.copy()
        position_delta = (
            wrist_matrix[:3, 3] - calibration_pose[:3, 3]
        ) * self._position_delta_signs
        target_pose[:3, 3] = initial_pose[:3, 3] + position_delta * self._position_scale
        rotation_delta = self._hand_imu_delta_matrix_for_target(target_side)
        if rotation_delta is not None:
            target_pose[:3, :3] = rotation_delta @ initial_pose[:3, :3]

        pose = _pose_matrix_to_array(target_pose)
        if target_side == "left":
            self._previous_left_pose = pose
        else:
            self._previous_right_pose = pose
        return pose

    def _make_action(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_elbow_pose: np.ndarray | None = None,
        right_elbow_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        hand_joints = self._hand_joint_targets.copy()
        if self._enable_elbow_ik_targets:
            if left_elbow_pose is None:
                left_elbow_pose = self._previous_left_elbow_pose
            if right_elbow_pose is None:
                right_elbow_pose = self._previous_right_elbow_pose
            return np.concatenate(
                [left_elbow_pose, left_pose, right_elbow_pose, right_pose, hand_joints]
            ).astype(np.float32)
        return np.concatenate([left_pose, right_pose, hand_joints]).astype(np.float32)

    def _action_dimension(self) -> int:
        frame_pose_count = 4 if self._enable_elbow_ik_targets else 2
        return frame_pose_count * 7 + self._hand_joint_count

    def _hand_skeleton_visual_rotation(
        self,
        side: str,
        body_arm_points: dict[str, np.ndarray],
        other_wrist: np.ndarray | None,
    ) -> np.ndarray | None:
        """Mount the attached hand with the palm facing the other wrist.

        Builds the world rotation that maps the canonical robot-hand-local
        skeleton points onto the displayed wrist: fingers follow the forearm
        axis, the palm points toward the other hand, and the forearm-relative
        SenseGlove IMU rotation is applied on top. Returns ``None`` when the
        palm direction is degenerate so the caller can fall back to the
        arm-frame mounting.
        """
        current_arm_rotation = _forearm_points_to_rotation(body_arm_points)
        if current_arm_rotation is None or other_wrist is None:
            return None

        forearm_axis = current_arm_rotation[:, 0]
        lateral = other_wrist - body_arm_points["wrist"]
        palm_world = _normalize_vector(
            lateral - float(np.dot(lateral, forearm_axis)) * forearm_axis
        )
        if palm_world is None:
            return None

        if side == "left":
            thumb_world = _normalize_vector(np.cross(palm_world, forearm_axis))
            if thumb_world is None:
                return None
            mounting = np.stack([-palm_world, -thumb_world, forearm_axis], axis=1)
        else:
            thumb_world = _normalize_vector(np.cross(forearm_axis, palm_world))
            if thumb_world is None:
                return None
            mounting = np.stack([palm_world, thumb_world, forearm_axis], axis=1)

        if self._hand_imu_relative_to_arm_frame:
            relative_rotation_world = self._hand_imu_relative_rotation_world(
                side, current_arm_rotation
            )
            if relative_rotation_world is not None:
                return (relative_rotation_world @ mounting).astype(np.float32)
            return mounting.astype(np.float32)

        rotation_delta = self._hand_imu_delta_matrix_for_target(side)
        if rotation_delta is not None:
            return (rotation_delta @ mounting).astype(np.float32)
        return mounting.astype(np.float32)

    def _attached_hand_skeleton_rotation(
        self,
        side: str,
        body_arm_points: dict[str, np.ndarray],
        initial_mounting_rotation: np.ndarray,
    ) -> np.ndarray:
        """Follow the displayed forearm while preserving the initial hand mounting."""
        key = f"visual_{side}"
        current_arm_rotation = _forearm_points_to_rotation(
            body_arm_points, self._forearm_plane_normals.get(key)
        )
        if current_arm_rotation is None:
            return initial_mounting_rotation.astype(np.float32)
        self._forearm_plane_normals[key] = current_arm_rotation[:, 2].copy()

        arm_reference = self._hand_skeleton_arm_reference_rotations.get(side)
        if arm_reference is None:
            arm_reference = current_arm_rotation.copy()
            self._hand_skeleton_arm_reference_rotations[side] = arm_reference

        relative_rotation_world = self._hand_imu_relative_rotation_world(
            side, current_arm_rotation
        )
        if relative_rotation_world is not None:
            imu_arm_reference = self._hand_arm_reference_rotations.get(side)
            if imu_arm_reference is not None:
                relative_rotation_local = (
                    imu_arm_reference.T
                    @ relative_rotation_world
                    @ imu_arm_reference
                )
                return (
                    current_arm_rotation
                    @ relative_rotation_local
                    @ arm_reference.T
                    @ initial_mounting_rotation
                ).astype(np.float32)

        rotation = (
            current_arm_rotation @ arm_reference.T @ initial_mounting_rotation
        )
        if not self._hand_imu_relative_to_arm_frame:
            rotation_delta = self._hand_imu_delta_matrix_for_target(side)
            if rotation_delta is not None:
                rotation = rotation_delta @ rotation
        return rotation.astype(np.float32)

    def _visualize_hand_skeletons(self) -> None:
        translations: list[torch.Tensor] = []
        orientations: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        marker_indices: list[int] = []
        line_starts: list[torch.Tensor] = []
        line_ends: list[torch.Tensor] = []
        line_marker_indices: list[int] = []

        display_offset = np.asarray(
            self._cfg.hand_skeleton_visualization_offset, dtype=np.float32
        )
        display_scale = float(self._cfg.hand_skeleton_visualization_scale)
        body_points = (
            self._body_points_for_visualization()
            if self._cfg.hand_skeleton_attach_to_body_wrist
            and not self._hand_only_mode
            else {}
        )
        for side, joint_marker, bone_marker in (("left", 0, 2), ("right", 1, 3)):
            local_points, edges, uses_sgcore_points = self._senseglove_hand_points(side)
            if (
                side == "left"
                and uses_sgcore_points
                and self._hand_skeleton_frame == "robot_hand_local"
            ):
                # SGCore's two physical hands are chiral. The bridge already aligns
                # their axes for retargeting, so restore the left-hand reflection
                # only for display before mounting it on the mirrored robot wrist.
                local_points = {
                    name: np.asarray([-point[0], point[1], point[2]], dtype=np.float32)
                    for name, point in local_points.items()
                }
            initial_pose = (
                self._initial_left_pose_matrix
                if side == "left"
                else self._initial_right_pose_matrix
            )
            hand_visual_reference_rotation = (
                initial_pose[:3, :3]
                if not uses_sgcore_points
                or self._hand_skeleton_frame == "robot_hand_local"
                else self._hand_imu_source_to_robot_rotation
            )
            if side == "left" and uses_sgcore_points:
                left_local_correction = _rotation_z(
                    math.radians(
                        self._cfg.hand_skeleton_left_local_rotation_deg
                    )
                )
                hand_visual_reference_rotation = (
                    hand_visual_reference_rotation @ left_local_correction
                )
            body_wrist = body_points.get(f"{side}_wrist")
            if body_wrist is not None:
                origin = body_wrist
                body_arm_points = {
                    name: body_points[f"{side}_{name}"]
                    for name in ("shoulder", "elbow", "wrist")
                    if f"{side}_{name}" in body_points
                }
                rotation = None
                if (
                    len(body_arm_points) == 3
                    and self._hand_skeleton_palms_face_each_other
                    and self._hand_skeleton_frame == "robot_hand_local"
                ):
                    other_side = "right" if side == "left" else "left"
                    rotation = self._hand_skeleton_visual_rotation(
                        side,
                        body_arm_points,
                        body_points.get(f"{other_side}_wrist"),
                    )
                if rotation is None and len(body_arm_points) == 3:
                    rotation = self._attached_hand_skeleton_rotation(
                        side,
                        body_arm_points,
                        hand_visual_reference_rotation,
                    )
                if rotation is None:
                    rotation = hand_visual_reference_rotation
            else:
                origin = initial_pose[:3, 3] + display_offset
                rotation = hand_visual_reference_rotation
                rotation_delta = self._hand_imu_delta_matrix_for_target(side)
                if rotation_delta is not None:
                    rotation = rotation_delta @ rotation
            self._hand_skeleton_visual_rotations[side] = rotation.copy()
            local_wrist = local_points.get(
                "wrist", np.zeros(3, dtype=np.float32)
            )
            world_points = {
                name: origin + display_scale * (rotation @ (point - local_wrist))
                for name, point in local_points.items()
            }

            for point in world_points.values():
                translations.append(
                    torch.tensor(point, device=self._sim_device, dtype=torch.float32)
                )
                orientations.append(
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._sim_device)
                )
                scales.append(torch.ones(3, device=self._sim_device))
                marker_indices.append(joint_marker)
            for start_name, end_name in edges:
                line_starts.append(
                    torch.tensor(
                        world_points[start_name],
                        device=self._sim_device,
                        dtype=torch.float32,
                    )
                )
                line_ends.append(
                    torch.tensor(
                        world_points[end_name],
                        device=self._sim_device,
                        dtype=torch.float32,
                    )
                )
                line_marker_indices.append(bone_marker)

        if line_starts:
            lines_pos, lines_quat, lines_length = self._get_connecting_lines(
                torch.stack(line_starts), torch.stack(line_ends)
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
        if not self._hand_skeleton_markers.is_visible():
            self._hand_skeleton_markers.set_visibility(True)
        self._hand_skeleton_markers.visualize(
            translations=torch.stack(translations),
            orientations=torch.stack(orientations),
            scales=torch.stack(scales),
            marker_indices=torch.tensor(
                marker_indices, device=self._sim_device, dtype=torch.int64
            ),
        )

    def _senseglove_hand_points(
        self, side: str
    ) -> tuple[dict[str, np.ndarray], list[tuple[str, str]], bool]:
        raw_points = self._hand_skeleton_positions.get(side)
        if raw_points is None:
            fallback_points, fallback_edges = self._senseglove_hand_fk(side)
            return fallback_points, fallback_edges, False

        palm = np.mean(raw_points[[4, 8, 12, 16]], axis=0).astype(np.float32)
        points: dict[str, np.ndarray] = {
            "wrist": np.zeros(3, dtype=np.float32),
            "palm": palm,
        }
        edges: list[tuple[str, str]] = [("wrist", "palm")]
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        for finger_index, finger in enumerate(fingers):
            previous_name = "palm"
            for joint_index in range(4):
                point_name = f"{finger}_{joint_index}"
                points[point_name] = raw_points[finger_index * 4 + joint_index].copy()
                edges.append((previous_name, point_name))
                previous_name = point_name
        return points, edges, True

    def _senseglove_hand_fk(
        self, side: str
    ) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
        """Reconstruct a display skeleton from retargeted hand joints, not measured 3D landmarks."""
        values = {
            name: float(self._hand_joint_targets[index])
            for index, name in enumerate(self._hand_joint_names)
            if index < self._hand_joint_targets.shape[0]
        }
        side_sign = -1.0 if side == "left" else 1.0
        points: dict[str, np.ndarray] = {
            "wrist": np.asarray([0.0, 0.0, -0.045], dtype=np.float32),
            "palm": np.zeros(3, dtype=np.float32),
        }
        edges: list[tuple[str, str]] = [("wrist", "palm")]

        finger_specs = (
            ("index", 0.030, 0.045, 0.029, 0.022, 1.0),
            ("middle", 0.010, 0.048, 0.031, 0.023, 0.0),
            ("ring", -0.012, 0.045, 0.029, 0.022, -1.0),
            ("pinky", -0.032, 0.039, 0.025, 0.019, -1.0),
        )
        for finger, lateral, proximal, middle, distal, spread_sign in finger_specs:
            base_name = f"{finger}_base"
            base = np.asarray([0.0, side_sign * lateral, 0.025], dtype=np.float32)
            points[base_name] = base
            edges.append(("palm", base_name))
            pitch = values.get(f"{side}_{finger}_mcp_pitch", 0.0)
            roll = values.get(f"{side}_{finger}_mcp_roll", 0.0)
            spread_rotation = _rotation_x(side_sign * spread_sign * roll)
            cumulative_angles = (pitch, pitch + 1.3 * pitch, pitch + 1.76 * pitch)
            current = base.copy()
            previous_name = base_name
            for segment_index, (length, angle) in enumerate(
                zip((proximal, middle, distal), cumulative_angles), start=1
            ):
                direction = spread_rotation @ (
                    _rotation_y(angle) @ np.asarray([0.0, 0.0, 1.0])
                )
                current = current + float(length) * direction.astype(np.float32)
                point_name = f"{finger}_{segment_index}"
                points[point_name] = current.copy()
                edges.append((previous_name, point_name))
                previous_name = point_name

        thumb_base = np.asarray([0.0, side_sign * 0.043, 0.005], dtype=np.float32)
        points["thumb_base"] = thumb_base
        edges.append(("palm", "thumb_base"))
        thumb_roll = values.get(f"{side}_thumb_cmc_roll", 0.0)
        thumb_yaw = values.get(f"{side}_thumb_cmc_yaw", 0.0)
        thumb_pitch = values.get(f"{side}_thumb_cmc_pitch", 0.0)
        # Point the thumb away from the palm center for both mirrored hands.
        thumb_spread = _rotation_x(
            -side_sign * (0.75 + 0.45 * thumb_yaw)
        ) @ _rotation_z(side_sign * 0.35 * thumb_roll)
        current = thumb_base.copy()
        previous_name = "thumb_base"
        for segment_index, (length, angle) in enumerate(
            zip(
                (0.040, 0.030, 0.023),
                (thumb_pitch, 2.38 * thumb_pitch, 3.87 * thumb_pitch),
            ),
            start=1,
        ):
            direction = thumb_spread @ (
                _rotation_y(angle) @ np.asarray([0.0, 0.0, 1.0])
            )
            current = current + float(length) * direction.astype(np.float32)
            point_name = f"thumb_{segment_index}"
            points[point_name] = current.copy()
            edges.append((previous_name, point_name))
            previous_name = point_name
        return points, edges

    def _visualize_body_pose(self) -> None:
        points = self._body_points_for_visualization()
        if not points:
            return

        translations: list[torch.Tensor] = []
        orientations: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        marker_indices: list[int] = []
        line_start_positions: list[torch.Tensor] = []
        line_end_positions: list[torch.Tensor] = []
        line_marker_indices: list[int] = []

        self._append_optional_point(
            points, "waist", 2, translations, orientations, scales, marker_indices
        )
        for point_name in (
            "neck",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
        ):
            self._append_optional_point(
                points,
                point_name,
                3,
                translations,
                orientations,
                scales,
                marker_indices,
            )
        self._append_optional_point(
            points, "left_wrist", 0, translations, orientations, scales, marker_indices
        )
        self._append_optional_point(
            points, "right_wrist", 1, translations, orientations, scales, marker_indices
        )

        for start, end in (
            ("waist", "neck"),
            ("neck", "head"),
            ("left_shoulder", "right_shoulder"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
        ):
            self._append_optional_line(
                points,
                start,
                end,
                6,
                line_start_positions,
                line_end_positions,
                line_marker_indices,
            )
        for side, marker_index in (("left", 4), ("right", 5)):
            self._append_optional_line(
                points,
                f"{side}_shoulder",
                f"{side}_elbow",
                marker_index,
                line_start_positions,
                line_end_positions,
                line_marker_indices,
            )
            self._append_optional_line(
                points,
                f"{side}_elbow",
                f"{side}_wrist",
                marker_index,
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

        if not self._markers.is_visible():
            self._markers.set_visibility(True)
        if (
            not self._calibration_pending()
            and not self._startup_reference_pending()
            and not self._printed_visualization_active
            and self._print_calibration_events
        ):
            print(
                "[bodytracking_udp] Shoulder/dual-arm skeleton visualization active at "
                f"{self._cfg.visualization_prim_path}; shoulders/elbows/wrists update from live full-body frames "
                "when present, with helper estimates filling any missing joints.",
                flush=True,
            )
            self._printed_visualization_active = True
        self._markers.visualize(
            translations=torch.stack(translations),
            orientations=torch.stack(orientations),
            scales=torch.stack(scales),
            marker_indices=torch.tensor(
                marker_indices, device=self._sim_device, dtype=torch.int64
            ),
        )

    def _body_points_for_visualization(self) -> dict[str, np.ndarray]:
        if "waist" in self._frames and self._use_waist_frame:
            waist_matrix = _pose_array_to_matrix(self._frames["waist"])
            world_to_body = _invert_pose_matrix(waist_matrix)
        else:
            world_to_body = np.eye(4, dtype=np.float32)

        offset = np.asarray(self._cfg.visualization_offset, dtype=np.float32)
        scale = self._cfg.visualization_scale
        points: dict[str, np.ndarray] = {"waist": offset.copy()}

        for name in (
            "neck",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hand",
            "right_hand",
        ):
            pose = self._frames.get(name)
            if pose is None:
                continue
            point_matrix = world_to_body @ _pose_array_to_matrix(pose)
            points[name] = offset + scale * point_matrix[:3, 3]

        if "left_wrist" not in points and "left_hand" in points:
            points["left_wrist"] = points["left_hand"]
        if "right_wrist" not in points and "right_hand" in points:
            points["right_wrist"] = points["right_hand"]

        if (
            self._inferred_left_shoulder_pos is not None
            and "left_shoulder" not in points
        ):
            points["left_shoulder"] = offset + scale * self._inferred_left_shoulder_pos
        if (
            self._inferred_right_shoulder_pos is not None
            and "right_shoulder" not in points
        ):
            points["right_shoulder"] = (
                offset + scale * self._inferred_right_shoulder_pos
            )
        if (
            self._inferred_left_shoulder_pos is not None
            and self._inferred_right_shoulder_pos is not None
        ):
            shoulder_center = 0.5 * (
                self._inferred_left_shoulder_pos + self._inferred_right_shoulder_pos
            )
            if "neck" not in points:
                points["neck"] = offset + scale * (
                    shoulder_center
                    + np.asarray(
                        [0.0, 0.0, self._cfg.estimated_neck_above_shoulders],
                        dtype=np.float32,
                    )
                )
            if "head" not in points:
                points["head"] = offset + scale * (
                    shoulder_center
                    + np.asarray(
                        [0.0, 0.0, self._cfg.estimated_head_above_shoulders],
                        dtype=np.float32,
                    )
                )

        if "neck" not in points:
            points["neck"] = offset + np.asarray(
                [0.0, 0.0, self._cfg.estimated_shoulder_height], dtype=np.float32
            )

        if "left_shoulder" not in points:
            points["left_shoulder"] = offset + np.asarray(
                [
                    0.0,
                    0.5 * self._cfg.estimated_shoulder_width,
                    self._cfg.estimated_shoulder_height,
                ],
                dtype=np.float32,
            )
        if "right_shoulder" not in points:
            points["right_shoulder"] = offset + np.asarray(
                [
                    0.0,
                    -0.5 * self._cfg.estimated_shoulder_width,
                    self._cfg.estimated_shoulder_height,
                ],
                dtype=np.float32,
            )

        for side in ("left", "right"):
            elbow_name = f"{side}_elbow"
            wrist_name = f"{side}_wrist"
            shoulder_name = f"{side}_shoulder"
            if (
                elbow_name not in points
                and wrist_name in points
                and shoulder_name in points
            ):
                points[elbow_name] = 0.5 * (points[shoulder_name] + points[wrist_name])
                points[elbow_name] += np.asarray(
                    [0.0, 0.0, -self._cfg.estimated_elbow_drop], dtype=np.float32
                )

        return points

    def _append_optional_point(
        self,
        points: dict[str, np.ndarray],
        point_name: str,
        marker_index: int,
        translations: list[torch.Tensor],
        orientations: list[torch.Tensor],
        scales: list[torch.Tensor],
        marker_indices: list[int],
    ) -> None:
        point = points.get(point_name)
        if point is None:
            return
        translations.append(
            torch.tensor(point, device=self._sim_device, dtype=torch.float32)
        )
        orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._sim_device))
        scales.append(torch.ones(3, device=self._sim_device))
        marker_indices.append(marker_index)

    def _append_optional_line(
        self,
        points: dict[str, np.ndarray],
        start_name: str,
        end_name: str,
        marker_index: int,
        line_start_positions: list[torch.Tensor],
        line_end_positions: list[torch.Tensor],
        line_marker_indices: list[int],
    ) -> None:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            return
        line_start_positions.append(
            torch.tensor(start, device=self._sim_device, dtype=torch.float32)
        )
        line_end_positions.append(
            torch.tensor(end, device=self._sim_device, dtype=torch.float32)
        )
        line_marker_indices.append(marker_index)

    def _get_connecting_lines(
        self, start_pos: torch.Tensor, end_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = end_pos - start_pos
        lengths = torch.norm(direction, dim=-1)
        positions = (start_pos + end_pos) / 2

        default_direction = torch.tensor(
            [0.0, 0.0, 1.0], device=self._sim_device
        ).expand(start_pos.size(0), -1)
        direction_norm = math_utils.normalize(direction)
        rotation_axis = torch.linalg.cross(default_direction, direction_norm)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1)
        mask = rotation_axis_norm > 1e-6
        rotation_axis = torch.where(
            mask.unsqueeze(-1),
            math_utils.normalize(rotation_axis),
            torch.tensor([1.0, 0.0, 0.0], device=self._sim_device).expand(
                start_pos.size(0), -1
            ),
        )
        cos_angle = torch.sum(default_direction * direction_norm, dim=-1)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        orientations = math_utils.quat_from_angle_axis(angle, rotation_axis)

        return positions, orientations, lengths

    def _print_debug(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_elbow_pose: np.ndarray | None = None,
        right_elbow_pose: np.ndarray | None = None,
    ) -> None:
        if self._debug_frame % self._debug_print_interval == 0:
            age = (
                time.monotonic() - self._last_packet_time_monotonic
                if self._last_packet_time_monotonic is not None
                else float("inf")
            )
            hand_age = (
                time.monotonic() - self._last_hand_packet_time_monotonic
                if self._last_hand_packet_time_monotonic is not None
                else float("inf")
            )
            hand_imu_age = (
                time.monotonic() - self._last_hand_orientation_packet_time_monotonic
                if self._last_hand_orientation_packet_time_monotonic is not None
                else float("inf")
            )
            names = ", ".join(sorted(self._frames.keys()))
            hand_preview = np.round(
                self._hand_joint_targets[: min(6, self._hand_joint_count)], 4
            ).tolist()
            hand_imu_sides = (
                ",".join(sorted(self._hand_orientation_delta_matrices.keys())) or "none"
            )
            elbow_text = ""
            if left_elbow_pose is not None and right_elbow_pose is not None:
                elbow_text = (
                    f" left_elbow_pos={np.round(left_elbow_pose[:3], 4).tolist()} "
                    f"right_elbow_pos={np.round(right_elbow_pose[:3], 4).tolist()}"
                )
            skeleton_text = ""
            if self._hand_skeleton_visual_rotations:
                skeleton_forwards = {
                    side: np.round(
                        rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
                        4,
                    ).tolist()
                    for side, rotation in self._hand_skeleton_visual_rotations.items()
                }
                skeleton_palms = {
                    side: np.round(
                        rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                        4,
                    ).tolist()
                    for side, rotation in self._hand_skeleton_visual_rotations.items()
                }
                skeleton_text = (
                    f" hand_skeleton_forwards={skeleton_forwards}"
                    f" hand_skeleton_palms={skeleton_palms}"
                )
            print(
                "[bodytracking_udp debug] "
                f"frame={self._debug_frame} sender={self._last_sender} age={age:.3f}s frames=[{names}] "
                f"hand_sender={self._last_hand_sender} hand_age={hand_age:.3f}s hand0={hand_preview} "
                f"hand_imu_sender={self._last_hand_orientation_sender} hand_imu_age={hand_imu_age:.3f}s "
                f"hand_imu_sides=[{hand_imu_sides}] "
                f"left_pos={np.round(left_pose[:3], 4).tolist()} right_pos={np.round(right_pose[:3], 4).tolist()}"
                f" left_quat={np.round(left_pose[3:], 4).tolist()}"
                f" right_quat={np.round(right_pose[3:], 4).tolist()}"
                f"{elbow_text}",
                f"{skeleton_text}",
                flush=True,
            )
        self._debug_frame += 1


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def _packet_debug_summary(payload: bytes) -> tuple[str, str, str]:
    try:
        message = json.loads(payload.decode("utf-8"))
    except Exception:
        return "unparseable", "unparseable", "unparseable"
    if not isinstance(message, dict):
        return type(message).__name__, "none", "none"

    payload_keys = ",".join(sorted(str(key) for key in message.keys())) or "none"
    frame_payload = message.get("frames")
    if isinstance(frame_payload, dict):
        frame_keys = (
            ",".join(sorted(str(key) for key in frame_payload.keys())) or "none"
        )
    else:
        frame_keys = "none"
    wrist_map_payload = message.get("wrist_map")
    if isinstance(wrist_map_payload, dict):
        wrist_map = " ".join(
            f"{key}={wrist_map_payload.get(key) or 'none'}" for key in ("left", "right")
        )
    else:
        wrist_map = "none"
    return payload_keys, frame_keys, wrist_map


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat)
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
    return _normalize_quat_wxyz(np.asarray([w, x, y, z], dtype=np.float32))


def _pose_array_to_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = pose[:3]
    matrix[:3, :3] = _quat_wxyz_to_matrix(pose[3:])
    return matrix


def _pose_matrix_to_array(matrix: np.ndarray) -> np.ndarray:
    quat = _matrix_to_quat_wxyz(matrix[:3, :3])
    return np.concatenate([matrix[:3, 3], quat]).astype(np.float32)


def _make_position_pose(position: np.ndarray) -> np.ndarray:
    return np.asarray(
        [position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )


def _average_pose_matrices(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.eye(4, dtype=np.float32)

    positions = np.stack([sample[:3, 3] for sample in samples], axis=0)
    quaternions = np.stack(
        [_matrix_to_quat_wxyz(sample[:3, :3]) for sample in samples], axis=0
    )
    reference = quaternions[0]
    aligned_quaternions = []
    for quat in quaternions:
        if float(np.dot(reference, quat)) < 0.0:
            quat = -quat
        aligned_quaternions.append(quat)
    avg_quat = _normalize_quat_wxyz(
        np.mean(np.stack(aligned_quaternions, axis=0), axis=0)
    )

    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.mean(positions, axis=0)
    matrix[:3, :3] = _quat_wxyz_to_matrix(avg_quat)
    return matrix


def _average_arm_points(
    samples: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray] | None:
    if not samples:
        return None
    return {
        key: np.mean(
            np.stack([sample[key] for sample in samples], axis=0), axis=0
        ).astype(np.float32)
        for key in ("shoulder", "elbow", "wrist")
    }


def _copy_arm_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: points[key].copy() for key in ("shoulder", "elbow", "wrist")}


def _arm_points_relative_to_shoulder(
    points: dict[str, np.ndarray], position_delta_signs: np.ndarray
) -> dict[str, np.ndarray]:
    shoulder = points["shoulder"]
    return {
        "shoulder": np.zeros(3, dtype=np.float32),
        "elbow": ((points["elbow"] - shoulder) * position_delta_signs).astype(
            np.float32
        ),
        "wrist": ((points["wrist"] - shoulder) * position_delta_signs).astype(
            np.float32
        ),
    }


def _arm_points_max_position_std(samples: list[dict[str, np.ndarray]]) -> float:
    if len(samples) < 2:
        return 0.0
    stacked = np.stack(
        [
            np.concatenate([sample[key] for key in ("shoulder", "elbow", "wrist")])
            for sample in samples
        ],
        axis=0,
    )
    return float(np.max(np.std(stacked, axis=0)))


def _pose_matrices_max_position_std(samples: list[np.ndarray]) -> float:
    if len(samples) < 2:
        return 0.0
    positions = np.stack([sample[:3, 3] for sample in samples], axis=0)
    return float(np.max(np.std(positions, axis=0)))


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-6:
        return None
    return (vector / norm).astype(np.float32)


def _scale_vector_to_length(
    vector: np.ndarray, length: float, fallback: np.ndarray
) -> np.ndarray | None:
    direction = _normalize_vector(vector)
    if direction is None:
        direction = _normalize_vector(fallback)
    if direction is None:
        return None
    return (direction * float(length)).astype(np.float32)


def _arm_points_to_rotation(points: dict[str, np.ndarray]) -> np.ndarray | None:
    reach_axis = _normalize_vector(points["wrist"] - points["shoulder"])
    upper_axis = _normalize_vector(points["elbow"] - points["shoulder"])
    if reach_axis is None:
        return None

    bend_axis = None
    if upper_axis is not None:
        bend_axis = _normalize_vector(
            upper_axis - float(np.dot(upper_axis, reach_axis)) * reach_axis
        )
    if bend_axis is None:
        for reference in (
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ):
            bend_axis = _normalize_vector(
                reference - float(np.dot(reference, reach_axis)) * reach_axis
            )
            if bend_axis is not None:
                break
    if bend_axis is None:
        return None

    plane_normal = _normalize_vector(np.cross(bend_axis, reach_axis))
    if plane_normal is None:
        return None

    side_axis = _normalize_vector(np.cross(plane_normal, reach_axis))
    if side_axis is None:
        return None

    return np.stack([reach_axis, side_axis, plane_normal], axis=1).astype(np.float32)


def _forearm_points_to_rotation(
    points: dict[str, np.ndarray],
    previous_plane_normal: np.ndarray | None = None,
) -> np.ndarray | None:
    """Build the wrist parent frame from the forearm axis and elbow bend plane."""
    forearm_axis = _normalize_vector(points["wrist"] - points["elbow"])
    upper_arm_axis = _normalize_vector(points["elbow"] - points["shoulder"])
    if forearm_axis is None:
        return None

    plane_normal = None
    if upper_arm_axis is not None:
        plane_normal = _normalize_vector(np.cross(upper_arm_axis, forearm_axis))
    if plane_normal is None and previous_plane_normal is not None:
        plane_normal = _normalize_vector(
            previous_plane_normal
            - float(np.dot(previous_plane_normal, forearm_axis)) * forearm_axis
        )
    if plane_normal is None:
        for reference in (
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ):
            plane_normal = _normalize_vector(np.cross(reference, forearm_axis))
            if plane_normal is not None:
                break
    if plane_normal is None:
        return None
    if previous_plane_normal is not None and np.dot(plane_normal, previous_plane_normal) < 0.0:
        plane_normal = -plane_normal

    bend_axis = _normalize_vector(np.cross(forearm_axis, plane_normal))
    if bend_axis is None:
        return None
    return np.stack([forearm_axis, bend_axis, plane_normal], axis=1).astype(
        np.float32
    )


def _invert_pose_matrix(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float32)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


def _rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Return the principal rotation vector for a 3x3 rotation matrix."""
    quat = _matrix_to_quat_wxyz(matrix)
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:].astype(np.float64)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1.0e-8:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(quat[0]))
    return (vector * (angle / vector_norm)).astype(np.float32)


def _rotvec_to_rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Construct a 3x3 rotation matrix from a rotation vector."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-8:
        quat = np.asarray([1.0, *(0.5 * rotvec)], dtype=np.float32)
    else:
        half_angle = 0.5 * angle
        quat = np.concatenate(
            (
                np.asarray([math.cos(half_angle)]),
                rotvec * (math.sin(half_angle) / angle),
            )
        ).astype(np.float32)
    return _quat_wxyz_to_matrix(quat)


def _rotation_x(angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine)), dtype=np.float32
    )


def _rotation_y(angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.asarray(
        ((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)), dtype=np.float32
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine = math.cos(float(angle))
    sine = math.sin(float(angle))
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)), dtype=np.float32
    )


def _body_packet_time_seconds(message: dict[str, Any]) -> float | None:
    """Return a monotonic-like packet timestamp in seconds when available."""
    for key in (
        "replay_sender_monotonic_ns",
        "sender_monotonic_ns",
        "xrt_body_timestamp_ns",
        "xrt_timestamp_ns",
    ):
        value = message.get(key)
        if value is not None:
            try:
                return float(value) / 1.0e9
            except (TypeError, ValueError):
                continue
    value = message.get("timestamp")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return None


@dataclass
class UdpBimanualBodyDeviceCfg(DeviceCfg):
    """Configuration for UDP body-tracking bimanual teleoperation."""

    host: str = "0.0.0.0"
    port: int = 15050
    max_packet_bytes: int = 65535
    max_stale_time_s: float = 0.5
    hand_joint_names: list[str] | None = None
    hand_joint_count: int = 20
    use_waist_frame: bool = False
    position_scale: float = 1.0
    position_delta_signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    retargeting_mode: str = "arm_vector"
    swap_left_right_wrist_targets: bool = False
    enable_elbow_ik_targets: bool = True
    require_calibration: bool = False
    auto_start_reference: bool = True
    auto_start_reference_delay_s: float = 2.0
    auto_start_reference_sample_start_s: float = 0.75
    auto_start_reference_max_position_std_m: float = 0.05
    auto_start_reference_min_samples: int = 5
    auto_start_reference_require_waist: bool = True
    calibration_delay_s: float = 10.0
    calibration_sample_start_s: float = 6.0
    calibration_prompt_interval_s: float = 1.0
    packet_quaternion_order: str = "wxyz"
    use_hand_imu_orientation: bool = True
    hand_only_mode: bool = False
    hand_orientation_inherit_arm_frame: bool = True
    hand_imu_relative_to_arm_frame: bool = True
    hand_imu_max_stale_time_s: float = 0.5
    hand_imu_source_to_robot_rotation: (
        tuple[float, float, float, float, float, float, float, float, float] | None
    ) = None
    hand_imu_left_local_axis_signs: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    hand_imu_left_axis_order: tuple[int, int, int] = (2, 0, 1)
    hand_imu_right_local_axis_signs: tuple[float, float, float] = (-1.0, 1.0, 1.0)
    hand_imu_right_swap_xy: bool = False
    source_to_robot_rotation: tuple[
        float, float, float, float, float, float, float, float, float
    ] = (
        0.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
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
    robot_left_shoulder_position: tuple[float, float, float] = (
        -0.146646,
        0.245584,
        1.529004,
    )
    robot_right_shoulder_position: tuple[float, float, float] = (
        -0.146803,
        -0.235015,
        1.529003,
    )
    robot_left_elbow_position: tuple[float, float, float] = (
        -0.146645,
        0.245585,
        1.219004,
    )
    robot_right_elbow_position: tuple[float, float, float] = (
        -0.146802,
        -0.235014,
        1.219003,
    )
    arm_vector_max_reach: float = 0.72
    arm_vector_prediction_horizon_s: float = 0.02
    arm_vector_prediction_lookback_s: float = 0.03
    arm_vector_prediction_max_velocity_m_s: float = 2.5
    arm_vector_prediction_max_displacement_m: float = 0.035
    upper_arm_angular_deadband_deg: float = 0.25
    forearm_angular_deadband_deg: float = 0.30
    arm_vector_position_mode: str = "segment_direction_absolute"
    enable_visualization: bool = True
    visualize_during_calibration: bool = False
    visualization_prim_path: str = "/Visuals/esrobo_body_capture_pose"
    visualization_offset: tuple[float, float, float] = (0.0, 1.1, 0.0)
    visualization_scale: float = 1.0
    enable_hand_skeleton_visualization: bool = False
    hand_skeleton_visualization_prim_path: str = (
        "/Visuals/esrobo_senseglove_hand_skeletons"
    )
    hand_skeleton_visualization_offset: tuple[float, float, float] = (0.55, 0.0, 0.18)
    hand_skeleton_visualization_scale: float = 1.6
    hand_skeleton_attach_to_body_wrist: bool = True
    hand_skeleton_palms_face_each_other: bool = False
    """Mount each attached hand with its palm facing the other wrist (robot_hand_local data).

    When disabled (default), the attached hands are mounted with the same
    hand-relative-to-arm rotation as the robot dexterous hands, so the skeleton
    hands follow the arms exactly like the robot hands do.
    """
    hand_skeleton_left_local_rotation_deg: float = 0.0
    estimated_shoulder_width: float = 0.38
    estimated_shoulder_height: float = 0.55
    estimated_neck_above_shoulders: float = 0.18
    estimated_head_above_shoulders: float = 0.34
    estimated_elbow_drop: float = 0.08
    print_calibration_events: bool = True
    class_type: type[DeviceBase] = field(default=UdpBimanualBodyDevice, init=False)


__all__ = [
    "BODY_FRAME_ALIASES",
    "UdpBimanualBodyDevice",
    "UdpBimanualBodyDeviceCfg",
]
