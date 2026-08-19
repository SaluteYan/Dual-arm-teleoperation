# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ESROBO waist-with-head dual-arm robot."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
ESROBO_PROJECT_ROOT = Path(
    os.environ.get("DUAL_ARM_TELEOP_ROOT", str(_DEFAULT_PROJECT_ROOT))
).expanduser().resolve()
ESROBO_USD_PATH = Path(
    os.environ.get(
        "ESROBO_USD_PATH",
        str(ESROBO_PROJECT_ROOT / "assets" / "esrobo" / "esrobo_waist_with_head.usd"),
    )
).expanduser().resolve()
ESROBO_URDF_PATH = Path(
    os.environ.get(
        "ESROBO_URDF_PATH",
        str(ESROBO_PROJECT_ROOT / "urdf" / "esrobo_waist_with_head" / "urdf" / "esrobo_waist_with_head.urdf"),
    )
).expanduser().resolve()
ESROBO_URDF_MESH_PATH = Path(
    os.environ.get(
        "ESROBO_URDF_MESH_PATH",
        str(ESROBO_PROJECT_ROOT / "urdf"),
    )
).expanduser().resolve()
ESROBO_ROOT_Z_OFFSET = float(os.environ.get("ESROBO_ROOT_Z_OFFSET", "0.3364894688129425"))
ESROBO_ARM_EFFORT_LIMIT_SIM = _env_float("ESROBO_ARM_EFFORT_LIMIT_SIM", 3500.0)
ESROBO_ARM_VELOCITY_LIMIT_SIM = _env_float("ESROBO_ARM_VELOCITY_LIMIT_SIM", 180.0)
ESROBO_ARM_STIFFNESS = _env_float("ESROBO_ARM_STIFFNESS", 8800.0)
ESROBO_ARM_DAMPING = _env_float("ESROBO_ARM_DAMPING", 320.0)
ESROBO_WAIST_HEAD_EFFORT_LIMIT_SIM = _env_float("ESROBO_WAIST_HEAD_EFFORT_LIMIT_SIM", 2500.0)
ESROBO_WAIST_HEAD_VELOCITY_LIMIT_SIM = _env_float("ESROBO_WAIST_HEAD_VELOCITY_LIMIT_SIM", 20.0)
ESROBO_WAIST_HEAD_STIFFNESS = _env_float("ESROBO_WAIST_HEAD_STIFFNESS", 8000.0)
ESROBO_WAIST_HEAD_DAMPING = _env_float("ESROBO_WAIST_HEAD_DAMPING", 800.0)

ESROBO_LOCKED_JOINTS = [
    "waist_joint1",
    "waist_joint2",
    "waist_joint3",
    "head_joint1",
    "head_joint2",
]

ESROBO_LEFT_ARM_JOINTS = [
    "leftArm_joint",
    "left_nero_joint2",
    "left_nero_joint3",
    "left_nero_joint4",
    "left_nero_joint5",
    "left_nero_joint6",
    "left_nero_joint7",
]

ESROBO_RIGHT_ARM_JOINTS = [
    "rightArm_joint",
    "right_nero_joint2",
    "right_nero_joint3",
    "right_nero_joint4",
    "right_nero_joint5",
    "right_nero_joint6",
    "right_nero_joint7",
]

ESROBO_LEFT_WRIST_JOINTS = ESROBO_LEFT_ARM_JOINTS[-3:]
ESROBO_RIGHT_WRIST_JOINTS = ESROBO_RIGHT_ARM_JOINTS[-3:]
"""Distal NERO joints 5-7, which provide the three wrist orientation axes."""

ESROBO_LEFT_HAND_ACTIVE_JOINTS = [
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
]

ESROBO_RIGHT_HAND_ACTIVE_JOINTS = [
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
]

ESROBO_HAND_MIMIC_JOINTS = {
    "left_thumb_mcp": ("left_thumb_cmc_pitch", 1.38, 0.0),
    "left_thumb_ip": ("left_thumb_cmc_pitch", 1.49, 0.0),
    "left_index_pip": ("left_index_mcp_pitch", 1.30, 0.0),
    "left_index_dip": ("left_index_mcp_pitch", 0.46, 0.0),
    "left_middle_pip": ("left_middle_mcp_pitch", 1.30, 0.0),
    "left_middle_dip": ("left_middle_mcp_pitch", 0.46, 0.0),
    "left_ring_pip": ("left_ring_mcp_pitch", 1.30, 0.0),
    "left_ring_dip": ("left_ring_mcp_pitch", 0.46, 0.0),
    "left_pinky_pip": ("left_pinky_mcp_pitch", 1.30, 0.0),
    "left_pinky_dip": ("left_pinky_mcp_pitch", 0.46, 0.0),
    "right_thumb_mcp": ("right_thumb_cmc_pitch", 1.38, 0.0),
    "right_thumb_ip": ("right_thumb_cmc_pitch", 1.49, 0.0),
    "right_index_pip": ("right_index_mcp_pitch", 1.30, 0.0),
    "right_index_dip": ("right_index_mcp_pitch", 0.46, 0.0),
    "right_middle_pip": ("right_middle_mcp_pitch", 1.30, 0.0),
    "right_middle_dip": ("right_middle_mcp_pitch", 0.46, 0.0),
    "right_ring_pip": ("right_ring_mcp_pitch", 1.34, 0.0),
    "right_ring_dip": ("right_ring_mcp_pitch", 0.46, 0.0),
    "right_pinky_pip": ("right_pinky_mcp_pitch", 1.34, 0.0),
    "right_pinky_dip": ("right_pinky_mcp_pitch", 0.46, 0.0),
}
"""Mimic relations imported from the ESROBO URDF and enforced by the action term."""

ESROBO_TELEOP_JOINTS = (
    ESROBO_LEFT_ARM_JOINTS
    + ESROBO_RIGHT_ARM_JOINTS
    + ESROBO_LEFT_HAND_ACTIVE_JOINTS
    + ESROBO_RIGHT_HAND_ACTIVE_JOINTS
)

ESROBO_WAIST_WITH_HEAD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ESROBO_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.05,
            angular_damping=0.05,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ESROBO_ROOT_Z_OFFSET),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_joint.*": 0.0,
            "head_joint.*": 0.0,
            "leftArm_joint": 0.0,
            "left_nero_joint[2-7]": 0.0,
            "rightArm_joint": 0.0,
            "right_nero_joint[2-7]": 0.0,
            "left_thumb_.*": 0.0,
            "left_index_.*": 0.0,
            "left_middle_.*": 0.0,
            "left_ring_.*": 0.0,
            "left_pinky_.*": 0.0,
            "right_thumb_.*": 0.0,
            "right_index_.*": 0.0,
            "right_middle_.*": 0.0,
            "right_ring_.*": 0.0,
            "right_pinky_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "waist_head_hold": ImplicitActuatorCfg(
            joint_names_expr=["waist_joint.*", "head_joint.*"],
            effort_limit_sim=ESROBO_WAIST_HEAD_EFFORT_LIMIT_SIM,
            velocity_limit_sim=ESROBO_WAIST_HEAD_VELOCITY_LIMIT_SIM,
            stiffness=ESROBO_WAIST_HEAD_STIFFNESS,
            damping=ESROBO_WAIST_HEAD_DAMPING,
            armature=0.1,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["leftArm_joint", "left_nero_joint[2-7]"],
            effort_limit_sim=ESROBO_ARM_EFFORT_LIMIT_SIM,
            velocity_limit_sim=ESROBO_ARM_VELOCITY_LIMIT_SIM,
            stiffness=ESROBO_ARM_STIFFNESS,
            damping=ESROBO_ARM_DAMPING,
            armature=0.05,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["rightArm_joint", "right_nero_joint[2-7]"],
            effort_limit_sim=ESROBO_ARM_EFFORT_LIMIT_SIM,
            velocity_limit_sim=ESROBO_ARM_VELOCITY_LIMIT_SIM,
            stiffness=ESROBO_ARM_STIFFNESS,
            damping=ESROBO_ARM_DAMPING,
            armature=0.05,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_thumb_cmc_.*",
                "left_index_mcp_.*",
                "left_middle_mcp_pitch",
                "left_ring_mcp_.*",
                "left_pinky_mcp_.*",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.005,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_thumb_cmc_.*",
                "right_index_mcp_.*",
                "right_middle_mcp_pitch",
                "right_ring_mcp_.*",
                "right_pinky_mcp_.*",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.005,
        ),
        "hand_mimic_passive": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_thumb_(mcp|ip)",
                "left_(index|middle|ring|pinky)_(pip|dip)",
                "right_thumb_(mcp|ip)",
                "right_(index|middle|ring|pinky)_(pip|dip)",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=2.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""ESROBO fixed-base dual-arm robot with active dexterous-hand joints for teleoperation."""

ESROBO_CFG = ESROBO_WAIST_WITH_HEAD_CFG
"""Alias for the default ESROBO fixed-base teleoperation robot configuration."""

__all__ = [
    "ESROBO_CFG",
    "ESROBO_ARM_DAMPING",
    "ESROBO_ARM_EFFORT_LIMIT_SIM",
    "ESROBO_ARM_STIFFNESS",
    "ESROBO_ARM_VELOCITY_LIMIT_SIM",
    "ESROBO_WAIST_HEAD_DAMPING",
    "ESROBO_WAIST_HEAD_EFFORT_LIMIT_SIM",
    "ESROBO_WAIST_HEAD_STIFFNESS",
    "ESROBO_WAIST_HEAD_VELOCITY_LIMIT_SIM",
    "ESROBO_LOCKED_JOINTS",
    "ESROBO_LEFT_ARM_JOINTS",
    "ESROBO_RIGHT_ARM_JOINTS",
    "ESROBO_LEFT_HAND_ACTIVE_JOINTS",
    "ESROBO_HAND_MIMIC_JOINTS",
    "ESROBO_LEFT_WRIST_JOINTS",
    "ESROBO_RIGHT_HAND_ACTIVE_JOINTS",
    "ESROBO_RIGHT_WRIST_JOINTS",
    "ESROBO_TELEOP_JOINTS",
    "ESROBO_ROOT_Z_OFFSET",
    "ESROBO_URDF_MESH_PATH",
    "ESROBO_URDF_PATH",
    "ESROBO_USD_PATH",
    "ESROBO_WAIST_WITH_HEAD_CFG",
]
