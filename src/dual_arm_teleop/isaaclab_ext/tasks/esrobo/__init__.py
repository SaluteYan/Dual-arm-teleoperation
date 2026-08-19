"""Gymnasium task registrations for ESROBO teleoperation."""

from __future__ import annotations

import gymnasium as gym

from .pick_place_env_cfg import FixedBaseBimanualIKESROBOEnvCfg
from .robot_only_env_cfg import FixedBaseBimanualIKESROBORobotOnlyEnvCfg

ESROBO_PICK_PLACE_TASK_ID = "DualArmTeleop-ESROBO-PickPlace-BimanualIK-Abs-v0"
ESROBO_ROBOT_ONLY_TASK_ID = "DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0"


def _register_task(task_id: str, env_cfg_entry_point: type) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": env_cfg_entry_point},
        disable_env_checker=True,
    )


_register_task(ESROBO_PICK_PLACE_TASK_ID, FixedBaseBimanualIKESROBOEnvCfg)
_register_task(ESROBO_ROBOT_ONLY_TASK_ID, FixedBaseBimanualIKESROBORobotOnlyEnvCfg)

__all__ = [
    "ESROBO_PICK_PLACE_TASK_ID",
    "ESROBO_ROBOT_ONLY_TASK_ID",
    "FixedBaseBimanualIKESROBOEnvCfg",
    "FixedBaseBimanualIKESROBORobotOnlyEnvCfg",
]
