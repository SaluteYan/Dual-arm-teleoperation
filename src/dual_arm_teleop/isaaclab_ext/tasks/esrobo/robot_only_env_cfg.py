# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-only fixed-base bimanual Pink IK task for ESROBO teleoperation."""

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from dual_arm_teleop.isaaclab_ext.assets.esrobo import ESROBO_CFG
from dual_arm_teleop.isaaclab_ext.devices import ESROBOOpenXRDeviceCfg, UdpBimanualBodyDeviceCfg
from dual_arm_teleop.isaaclab_ext.retargeters import ESROBOUpperBodyRetargeterCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place import mdp as locomanip_mdp
from dual_arm_teleop.isaaclab_ext.configs.esrobo_pink_controller_cfg import (
    ESROBO_HAND_JOINT_NAMES,
    ESROBO_INITIAL_LEFT_WRIST_POSE_W,
    ESROBO_INITIAL_RIGHT_WRIST_POSE_W,
    ESROBO_UPPER_BODY_IK_ACTION_CFG,
    get_esrobo_body_position_delta_signs,
    get_esrobo_body_source_to_robot_rotation,
    get_esrobo_body_visualization_offset,
    get_esrobo_hand_imu_source_to_robot_rotation,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no", "off")


def _env_vec3(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    values = tuple(float(value) for value in raw.replace(",", " ").split())
    if len(values) != 3:
        raise ValueError(f"{name} must contain 3 numbers, got {len(values)} from {raw!r}")
    return values


def _env_axis_order(
    name: str, default: tuple[int, int, int]
) -> tuple[int, int, int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    values = tuple(int(value) for value in raw.replace(",", " ").split())
    if sorted(values) != [0, 1, 2]:
        raise ValueError(
            f"{name} must be a permutation of 0, 1, 2, got {values} from {raw!r}"
        )
    return values


@configclass
class FixedBaseBimanualIKESROBORobotOnlySceneCfg(InteractiveSceneCfg):
    """Scene with ESROBO, a ground plane, and a light for teleoperation bring-up."""

    robot: ArticulationCfg = ESROBO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.fix_root_link = True


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = ESROBO_UPPER_BODY_IK_ACTION_CFG


@configclass
class ObservationsCfg:
    """Robot-only observations for teleoperation."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_links_state = ObsTerm(func=manip_mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "leftHand_link"})
        left_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "leftHand_link"})
        right_eef_pos = ObsTerm(func=manip_mdp.get_eef_pos, params={"link_name": "rightHand_link"})
        right_eef_quat = ObsTerm(func=manip_mdp.get_eef_quat, params={"link_name": "rightHand_link"})
        hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state, params={"joint_names": ESROBO_HAND_JOINT_NAMES}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the robot-only MDP."""

    time_out = DoneTerm(func=locomanip_mdp.time_out, time_out=True)


@configclass
class EventsCfg:
    """Reset events for the robot-only teleoperation task."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class FixedBaseBimanualIKESROBORobotOnlyEnvCfg(ManagerBasedRLEnvCfg):
    """ESROBO robot-only fixed-base bimanual teleoperation environment."""

    scene: FixedBaseBimanualIKESROBORobotOnlySceneCfg = FixedBaseBimanualIKESROBORobotOnlySceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    commands = None
    rewards = None
    curriculum = None

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -0.45),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    def __post_init__(self):
        self.decimation = int(os.environ.get("ESROBO_DECIMATION", "1"))
        self.episode_length_s = 20.0
        self.sim.dt = float(os.environ.get("ESROBO_SIM_DT", str(1 / 60)))
        self.sim.render_interval = int(os.environ.get("ESROBO_RENDER_INTERVAL", "2"))

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/waist_link3"
        self.xr.fixed_anchor_height = True

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": ESROBOOpenXRDeviceCfg(
                    retargeters=[
                        ESROBOUpperBodyRetargeterCfg(
                            enable_visualization=True,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                            initial_left_wrist_pose=ESROBO_INITIAL_LEFT_WRIST_POSE_W,
                            initial_right_wrist_pose=ESROBO_INITIAL_RIGHT_WRIST_POSE_W,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "bodytracking_udp": UdpBimanualBodyDeviceCfg(
                    host=os.environ.get("ESROBO_BODY_UDP_HOST", "0.0.0.0"),
                    port=int(os.environ.get("ESROBO_BODY_UDP_PORT", "15050")),
                    sim_device=self.sim.device,
                    hand_joint_names=self.actions.upper_body_ik.hand_joint_names,
                    use_waist_frame=(
                        os.environ.get("ESROBO_BODY_USE_WAIST_FRAME", "0").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    position_scale=float(os.environ.get("ESROBO_BODY_POSITION_SCALE", "1.0")),
                    position_delta_signs=get_esrobo_body_position_delta_signs(),
                    retargeting_mode=os.environ.get("ESROBO_BODY_RETARGETING_MODE", "arm_vector"),
                    swap_left_right_wrist_targets=(
                        os.environ.get("ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS", "0").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    enable_elbow_ik_targets=(
                        os.environ.get("ESROBO_ENABLE_ELBOW_IK_TASKS", "1").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    require_calibration=(
                        os.environ.get("ESROBO_BODY_REQUIRE_CALIBRATION", "0").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    auto_start_reference=(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE", "1").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    auto_start_reference_delay_s=float(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE_DELAY_S", "2.0")
                    ),
                    auto_start_reference_sample_start_s=float(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE_SAMPLE_START_S", "0.75")
                    ),
                    auto_start_reference_max_position_std_m=float(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE_MAX_STD_M", "0.05")
                    ),
                    auto_start_reference_min_samples=int(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE_MIN_SAMPLES", "5")
                    ),
                    auto_start_reference_require_waist=(
                        os.environ.get("ESROBO_BODY_AUTO_START_REFERENCE_REQUIRE_WAIST", "1").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    calibration_delay_s=float(os.environ.get("ESROBO_BODY_CALIBRATION_DELAY_S", "10.0")),
                    calibration_sample_start_s=float(
                        os.environ.get("ESROBO_BODY_CALIBRATION_SAMPLE_START_S", "6.0")
                    ),
                    calibration_prompt_interval_s=float(
                        os.environ.get("ESROBO_BODY_CALIBRATION_PROMPT_INTERVAL_S", "1.0")
                    ),
                    arm_vector_max_reach=float(os.environ.get("ESROBO_BODY_ARM_VECTOR_MAX_REACH", "0.72")),
                    arm_vector_prediction_horizon_s=float(
                        os.environ.get("ESROBO_BODY_ARM_VECTOR_PREDICTION_HORIZON_S", "0.02")
                    ),
                    arm_vector_prediction_lookback_s=float(
                        os.environ.get("ESROBO_BODY_ARM_VECTOR_PREDICTION_LOOKBACK_S", "0.03")
                    ),
                    arm_vector_prediction_max_velocity_m_s=float(
                        os.environ.get("ESROBO_BODY_ARM_VECTOR_PREDICTION_MAX_VELOCITY_M_S", "2.5")
                    ),
                    arm_vector_prediction_max_displacement_m=float(
                        os.environ.get("ESROBO_BODY_ARM_VECTOR_PREDICTION_MAX_DISPLACEMENT_M", "0.035")
                    ),
                    upper_arm_angular_deadband_deg=float(
                        os.environ.get("ESROBO_BODY_UPPER_ARM_ANGULAR_DEADBAND_DEG", "0.25")
                    ),
                    forearm_angular_deadband_deg=float(
                        os.environ.get("ESROBO_BODY_FOREARM_ANGULAR_DEADBAND_DEG", "0.30")
                    ),
                    arm_vector_position_mode=os.environ.get(
                        "ESROBO_BODY_ARM_VECTOR_POSITION_MODE", "segment_direction_absolute"
                    ),
                    source_to_robot_rotation=get_esrobo_body_source_to_robot_rotation(),
                    use_hand_imu_orientation=(
                        os.environ.get("ESROBO_USE_HAND_IMU_ORIENTATION", "1").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    hand_only_mode=_env_bool("ESROBO_HAND_ONLY_MODE", False),
                    hand_orientation_inherit_arm_frame=_env_bool(
                        "ESROBO_HAND_ORIENTATION_INHERIT_ARM_FRAME", True
                    ),
                    hand_imu_relative_to_arm_frame=_env_bool(
                        "ESROBO_HAND_IMU_RELATIVE_TO_ARM_FRAME", True
                    ),
                    hand_imu_max_stale_time_s=float(os.environ.get("ESROBO_HAND_IMU_MAX_STALE_TIME_S", "0.5")),
                    hand_imu_source_to_robot_rotation=get_esrobo_hand_imu_source_to_robot_rotation(),
                    hand_imu_left_local_axis_signs=_env_vec3(
                        "ESROBO_HAND_IMU_LEFT_LOCAL_AXIS_SIGNS", (-1.0, -1.0, 1.0)
                    ),
                    hand_imu_left_axis_order=_env_axis_order(
                        "ESROBO_HAND_IMU_LEFT_AXIS_ORDER", (2, 0, 1)
                    ),
                    hand_imu_right_local_axis_signs=_env_vec3(
                        "ESROBO_HAND_IMU_RIGHT_LOCAL_AXIS_SIGNS", (-1.0, 1.0, 1.0)
                    ),
                    hand_imu_right_swap_xy=_env_bool(
                        "ESROBO_HAND_IMU_RIGHT_SWAP_XY", False
                    ),
                    initial_left_wrist_pose=ESROBO_INITIAL_LEFT_WRIST_POSE_W,
                    initial_right_wrist_pose=ESROBO_INITIAL_RIGHT_WRIST_POSE_W,
                    enable_visualization=(
                        os.environ.get("ESROBO_BODY_ENABLE_VISUALIZATION", "1").strip().lower()
                        not in ("0", "false", "no", "off")
                    ),
                    visualization_offset=get_esrobo_body_visualization_offset(),
                    visualization_scale=float(os.environ.get("ESROBO_BODY_VISUALIZATION_SCALE", "1.0")),
                    enable_hand_skeleton_visualization=_env_bool(
                        "ESROBO_HAND_SKELETON_VISUALIZATION", False
                    ),
                    hand_skeleton_visualization_offset=_env_vec3(
                        "ESROBO_HAND_SKELETON_VISUALIZATION_OFFSET", (0.55, 0.0, 0.18)
                    ),
                    hand_skeleton_visualization_scale=float(
                        os.environ.get("ESROBO_HAND_SKELETON_VISUALIZATION_SCALE", "1.6")
                    ),
                    hand_skeleton_attach_to_body_wrist=_env_bool(
                        "ESROBO_HAND_SKELETON_ATTACH_TO_BODY_WRIST", True
                    ),
                    hand_skeleton_palms_face_each_other=_env_bool(
                        "ESROBO_HAND_SKELETON_PALMS_FACE_EACH_OTHER", False
                    ),
                    hand_skeleton_left_local_rotation_deg=float(
                        os.environ.get(
                            "ESROBO_HAND_SKELETON_LEFT_LOCAL_ROTATION_DEG", "0.0"
                        )
                    ),
                ),
            }
        )


__all__ = [
    "FixedBaseBimanualIKESROBORobotOnlyEnvCfg",
    "FixedBaseBimanualIKESROBORobotOnlySceneCfg",
]
