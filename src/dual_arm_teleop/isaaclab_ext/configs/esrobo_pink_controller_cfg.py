# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pink IK configuration for ESROBO fixed-base bimanual teleoperation."""

import os

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from dual_arm_teleop.isaaclab_ext.actions import ESROBOPinkInverseKinematicsAction
from dual_arm_teleop.isaaclab_ext.assets.esrobo import (
    ESROBO_LEFT_ARM_JOINTS,
    ESROBO_LEFT_HAND_ACTIVE_JOINTS,
    ESROBO_LEFT_WRIST_JOINTS,
    ESROBO_RIGHT_ARM_JOINTS,
    ESROBO_RIGHT_HAND_ACTIVE_JOINTS,
    ESROBO_RIGHT_WRIST_JOINTS,
    ESROBO_ROOT_Z_OFFSET,
    ESROBO_URDF_MESH_PATH,
    ESROBO_URDF_PATH,
)
from dual_arm_teleop.isaaclab_ext.controllers import ESROBOPinkIKControllerCfg


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


ESROBO_BASE_LINK_NAME = "waist_link3"
ESROBO_ENABLE_ELBOW_IK_TASKS = _env_bool("ESROBO_ENABLE_ELBOW_IK_TASKS", True)
ESROBO_LEFT_ELBOW_FRAME_NAME = "left_nero_link3"
ESROBO_RIGHT_ELBOW_FRAME_NAME = "right_nero_link3"
ESROBO_LEFT_HAND_FRAME_NAME = "leftHand_link"
ESROBO_RIGHT_HAND_FRAME_NAME = "rightHand_link"
ESROBO_CONTROLLED_FRAME_NAMES = (
    [
        ESROBO_LEFT_ELBOW_FRAME_NAME,
        ESROBO_LEFT_HAND_FRAME_NAME,
        ESROBO_RIGHT_ELBOW_FRAME_NAME,
        ESROBO_RIGHT_HAND_FRAME_NAME,
    ]
    if ESROBO_ENABLE_ELBOW_IK_TASKS
    else [ESROBO_LEFT_HAND_FRAME_NAME, ESROBO_RIGHT_HAND_FRAME_NAME]
)
ESROBO_WRIST_ONLY_IK = _env_bool("ESROBO_WRIST_ONLY_IK", False)
ESROBO_PINK_CONTROLLED_JOINT_NAMES = (
    ESROBO_LEFT_WRIST_JOINTS + ESROBO_RIGHT_WRIST_JOINTS
    if ESROBO_WRIST_ONLY_IK
    else ESROBO_LEFT_ARM_JOINTS + ESROBO_RIGHT_ARM_JOINTS
)
ESROBO_HAND_JOINT_NAMES = ESROBO_LEFT_HAND_ACTIVE_JOINTS + ESROBO_RIGHT_HAND_ACTIVE_JOINTS

ESROBO_IK_MAX_JOINT_POSITION_DELTA = float(os.environ.get("ESROBO_IK_MAX_JOINT_POSITION_DELTA", "0.105"))
ESROBO_IK_MAX_COMMAND_LEAD = float(os.environ.get("ESROBO_IK_MAX_COMMAND_LEAD", "0.25"))
ESROBO_IK_MAX_COMMAND_STEP = float(os.environ.get("ESROBO_IK_MAX_COMMAND_STEP", "0.0"))
ESROBO_IK_FRAME_TASK_GAIN = float(os.environ.get("ESROBO_IK_FRAME_TASK_GAIN", "1.0"))
ESROBO_IK_FRAME_TASK_LM_DAMPING = float(os.environ.get("ESROBO_IK_FRAME_TASK_LM_DAMPING", "0.012"))
ESROBO_IK_POSITION_COST = float(os.environ.get("ESROBO_IK_POSITION_COST", "32.0"))
ESROBO_IK_ORIENTATION_COST = float(os.environ.get("ESROBO_IK_ORIENTATION_COST", "0.05"))
ESROBO_IK_ELBOW_POSITION_COST = float(os.environ.get("ESROBO_IK_ELBOW_POSITION_COST", "80.0"))
ESROBO_IK_ELBOW_ORIENTATION_COST = float(os.environ.get("ESROBO_IK_ELBOW_ORIENTATION_COST", "0.0"))
ESROBO_IK_ELBOW_FRAME_TASK_GAIN = float(os.environ.get("ESROBO_IK_ELBOW_FRAME_TASK_GAIN", "1.0"))
ESROBO_IK_ELBOW_FRAME_TASK_LM_DAMPING = float(os.environ.get("ESROBO_IK_ELBOW_FRAME_TASK_LM_DAMPING", "0.006"))
ESROBO_IK_NULLSPACE_POSTURE_COST = float(os.environ.get("ESROBO_IK_NULLSPACE_POSTURE_COST", "0.001"))
ESROBO_IK_NULLSPACE_POSTURE_GAIN = float(os.environ.get("ESROBO_IK_NULLSPACE_POSTURE_GAIN", "0.001"))
ESROBO_IK_NULLSPACE_POSTURE_LM_DAMPING = float(
    os.environ.get("ESROBO_IK_NULLSPACE_POSTURE_LM_DAMPING", "0.5")
)
ESROBO_IK_ENFORCE_VELOCITY_LIMITS = _env_bool("ESROBO_IK_ENFORCE_VELOCITY_LIMITS", False)

ESROBO_DEFAULT_BODY_SOURCE_TO_ROBOT_ROTATION = (
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
"""Default PICO body pose axis mapping with the skeleton facing the ESROBO front."""

ESROBO_DEFAULT_BODY_POSITION_DELTA_SIGNS = (1.0, 1.0, 1.0)
"""Default robot-frame position delta signs after the source-to-robot rotation has aligned axes."""

ESROBO_DEFAULT_BODY_VISUALIZATION_OFFSET = (0.0, 1.1, 1.85)
"""Default offset that places the absolute PICO skeleton next to the ESROBO robot in the viewport."""

ESROBO_INITIAL_LEFT_WRIST_POSE_W = (
    -0.14664,
    0.24538,
    0.48300 + ESROBO_ROOT_Z_OFFSET,
    0.0,
    0.70711,
    -0.70710,
    0.0,
)
ESROBO_INITIAL_RIGHT_WRIST_POSE_W = (
    -0.14680,
    -0.23480,
    0.48300 + ESROBO_ROOT_Z_OFFSET,
    0.0,
    -0.70710,
    -0.70711,
    0.0,
)
"""Initial hand-link world poses after lifting the USD root so the base sits on the ground."""


def _make_frame_task(
    frame_name: str,
    position_cost: float,
    orientation_cost: float,
    gain: float | None = None,
    lm_damping: float | None = None,
) -> LocalFrameTask:
    return LocalFrameTask(
        frame_name,
        base_link_frame_name=ESROBO_BASE_LINK_NAME,
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=ESROBO_IK_FRAME_TASK_LM_DAMPING if lm_damping is None else lm_damping,
        gain=ESROBO_IK_FRAME_TASK_GAIN if gain is None else gain,
    )


def _make_variable_input_tasks() -> list:
    if ESROBO_ENABLE_ELBOW_IK_TASKS:
        frame_tasks = [
            _make_frame_task(
                ESROBO_LEFT_ELBOW_FRAME_NAME,
                ESROBO_IK_ELBOW_POSITION_COST,
                ESROBO_IK_ELBOW_ORIENTATION_COST,
                gain=ESROBO_IK_ELBOW_FRAME_TASK_GAIN,
                lm_damping=ESROBO_IK_ELBOW_FRAME_TASK_LM_DAMPING,
            ),
            _make_frame_task(ESROBO_LEFT_HAND_FRAME_NAME, ESROBO_IK_POSITION_COST, ESROBO_IK_ORIENTATION_COST),
            _make_frame_task(
                ESROBO_RIGHT_ELBOW_FRAME_NAME,
                ESROBO_IK_ELBOW_POSITION_COST,
                ESROBO_IK_ELBOW_ORIENTATION_COST,
                gain=ESROBO_IK_ELBOW_FRAME_TASK_GAIN,
                lm_damping=ESROBO_IK_ELBOW_FRAME_TASK_LM_DAMPING,
            ),
            _make_frame_task(ESROBO_RIGHT_HAND_FRAME_NAME, ESROBO_IK_POSITION_COST, ESROBO_IK_ORIENTATION_COST),
        ]
    else:
        frame_tasks = [
            _make_frame_task(ESROBO_LEFT_HAND_FRAME_NAME, ESROBO_IK_POSITION_COST, ESROBO_IK_ORIENTATION_COST),
            _make_frame_task(ESROBO_RIGHT_HAND_FRAME_NAME, ESROBO_IK_POSITION_COST, ESROBO_IK_ORIENTATION_COST),
        ]

    if not ESROBO_WRIST_ONLY_IK:
        frame_tasks.append(
            NullSpacePostureTask(
                cost=ESROBO_IK_NULLSPACE_POSTURE_COST,
                lm_damping=ESROBO_IK_NULLSPACE_POSTURE_LM_DAMPING,
                controlled_frames=ESROBO_CONTROLLED_FRAME_NAMES,
                controlled_joints=[
                    "leftArm_joint",
                    "left_nero_joint2",
                    "left_nero_joint3",
                    "rightArm_joint",
                    "right_nero_joint2",
                    "right_nero_joint3",
                ],
                gain=ESROBO_IK_NULLSPACE_POSTURE_GAIN,
            )
        )
    return frame_tasks


def get_esrobo_body_source_to_robot_rotation() -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return the body-tracking source-to-robot rotation, optionally overridden by environment."""
    raw = os.environ.get("ESROBO_BODY_SOURCE_TO_ROBOT_ROTATION")
    if raw is None or not raw.strip():
        return ESROBO_DEFAULT_BODY_SOURCE_TO_ROBOT_ROTATION

    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != 9:
        raise ValueError(
            "ESROBO_BODY_SOURCE_TO_ROBOT_ROTATION must contain 9 numbers, "
            f"but got {len(values)} from {raw!r}."
        )
    return values


def get_esrobo_hand_imu_source_to_robot_rotation() -> (
    tuple[float, float, float, float, float, float, float, float, float] | None
):
    """Return the SenseGlove IMU source-to-robot rotation, optionally overridden by environment."""
    raw = os.environ.get("ESROBO_HAND_IMU_SOURCE_TO_ROBOT_ROTATION")
    if raw is None or not raw.strip():
        return None

    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != 9:
        raise ValueError(
            "ESROBO_HAND_IMU_SOURCE_TO_ROBOT_ROTATION must contain 9 numbers, "
            f"but got {len(values)} from {raw!r}."
        )
    return values


def get_esrobo_body_position_delta_signs() -> tuple[float, float, float]:
    """Return robot-frame wrist position delta signs, optionally overridden by environment."""
    raw = os.environ.get("ESROBO_BODY_POSITION_DELTA_SIGNS")
    if raw is None or not raw.strip():
        return ESROBO_DEFAULT_BODY_POSITION_DELTA_SIGNS

    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != 3:
        raise ValueError(
            "ESROBO_BODY_POSITION_DELTA_SIGNS must contain 3 numbers, "
            f"but got {len(values)} from {raw!r}."
        )
    return values


def get_esrobo_body_visualization_offset() -> tuple[float, float, float]:
    """Return the body skeleton visualization offset, optionally overridden by environment."""
    raw = os.environ.get("ESROBO_BODY_VISUALIZATION_OFFSET")
    if raw is None or not raw.strip():
        return ESROBO_DEFAULT_BODY_VISUALIZATION_OFFSET

    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != 3:
        raise ValueError(
            "ESROBO_BODY_VISUALIZATION_OFFSET must contain 3 numbers, "
            f"but got {len(values)} from {raw!r}."
        )
    return values


ESROBO_UPPER_BODY_IK_CONTROLLER_CFG = ESROBOPinkIKControllerCfg(
    articulation_name="robot",
    urdf_path=str(ESROBO_URDF_PATH),
    mesh_path=str(ESROBO_URDF_MESH_PATH),
    base_link_name=ESROBO_BASE_LINK_NAME,
    num_hand_joints=len(ESROBO_HAND_JOINT_NAMES),
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    task_error_tolerance=1.0e-4,
    enforce_velocity_limits=ESROBO_IK_ENFORCE_VELOCITY_LIMITS,
    max_joint_position_delta=ESROBO_IK_MAX_JOINT_POSITION_DELTA,
    max_command_lead=ESROBO_IK_MAX_COMMAND_LEAD,
    max_command_step=ESROBO_IK_MAX_COMMAND_STEP,
    clamp_joint_positions_to_limits=True,
    variable_input_tasks=_make_variable_input_tasks(),
    fixed_input_tasks=[],
)
"""Pink IK controller for ESROBO's two hand-base frames relative to ``waist_link3``."""


ESROBO_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    class_type=ESROBOPinkInverseKinematicsAction,
    pink_controlled_joint_names=ESROBO_PINK_CONTROLLED_JOINT_NAMES,
    hand_joint_names=ESROBO_HAND_JOINT_NAMES,
    target_eef_link_names={
        "left_elbow": ESROBO_LEFT_ELBOW_FRAME_NAME,
        "left_wrist": ESROBO_LEFT_HAND_FRAME_NAME,
        "right_elbow": ESROBO_RIGHT_ELBOW_FRAME_NAME,
        "right_wrist": ESROBO_RIGHT_HAND_FRAME_NAME,
    },
    asset_name="robot",
    controller=ESROBO_UPPER_BODY_IK_CONTROLLER_CFG,
)
"""Action layout: optional elbow poses + left/right hand poses + left/right active hand joints."""


__all__ = [
    "ESROBO_BASE_LINK_NAME",
    "ESROBO_CONTROLLED_FRAME_NAMES",
    "ESROBO_DEFAULT_BODY_POSITION_DELTA_SIGNS",
    "ESROBO_DEFAULT_BODY_SOURCE_TO_ROBOT_ROTATION",
    "ESROBO_DEFAULT_BODY_VISUALIZATION_OFFSET",
    "ESROBO_ENABLE_ELBOW_IK_TASKS",
    "ESROBO_HAND_JOINT_NAMES",
    "ESROBO_IK_ELBOW_FRAME_TASK_GAIN",
    "ESROBO_IK_ELBOW_FRAME_TASK_LM_DAMPING",
    "ESROBO_IK_ELBOW_ORIENTATION_COST",
    "ESROBO_IK_ELBOW_POSITION_COST",
    "ESROBO_IK_ENFORCE_VELOCITY_LIMITS",
    "ESROBO_IK_FRAME_TASK_GAIN",
    "ESROBO_IK_FRAME_TASK_LM_DAMPING",
    "ESROBO_IK_MAX_JOINT_POSITION_DELTA",
    "ESROBO_IK_MAX_COMMAND_STEP",
    "ESROBO_IK_NULLSPACE_POSTURE_COST",
    "ESROBO_IK_NULLSPACE_POSTURE_GAIN",
    "ESROBO_IK_NULLSPACE_POSTURE_LM_DAMPING",
    "ESROBO_IK_ORIENTATION_COST",
    "ESROBO_IK_POSITION_COST",
    "ESROBO_INITIAL_LEFT_WRIST_POSE_W",
    "ESROBO_INITIAL_RIGHT_WRIST_POSE_W",
    "ESROBO_PINK_CONTROLLED_JOINT_NAMES",
    "ESROBO_UPPER_BODY_IK_ACTION_CFG",
    "ESROBO_UPPER_BODY_IK_CONTROLLER_CFG",
    "get_esrobo_body_position_delta_signs",
    "get_esrobo_body_source_to_robot_rotation",
    "get_esrobo_body_visualization_offset",
    "get_esrobo_hand_imu_source_to_robot_rotation",
]
