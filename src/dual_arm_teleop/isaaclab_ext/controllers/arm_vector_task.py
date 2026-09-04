"""Pink task for matching a robot arm segment to a target vector."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from pink.tasks import Task

from isaaclab.controllers.pink_ik.pink_kinematics_configuration import (
    PinkKinematicsConfiguration,
)


class ArmSegmentVectorTask(Task):
    """Match the vector between two robot frames in the controller base frame."""

    def __init__(
        self,
        origin_frame: str,
        task_frame: str,
        base_link_frame_name: str,
        target_task_frame: str,
        target_origin_frame: str | None,
        cost: float,
        gain: float = 1.0,
        lm_damping: float = 0.0,
        huber_delta: float = 0.04,
    ):
        super().__init__(cost=cost, gain=gain, lm_damping=lm_damping)
        self.origin_frame = origin_frame
        self.task_frame = task_frame
        self.base_link_frame_name = base_link_frame_name
        self.target_task_frame = target_task_frame
        self.target_origin_frame = target_origin_frame
        self.huber_delta = max(float(huber_delta), 1.0e-6)
        self.target_vector = np.zeros(3, dtype=np.float64)

    def set_target_from_configuration(
        self, configuration: PinkKinematicsConfiguration
    ) -> None:
        self.target_vector = self._current_vector(configuration)

    def update_target_from_frame_positions(
        self,
        frame_positions: Mapping[str, np.ndarray],
        configuration: PinkKinematicsConfiguration,
    ) -> None:
        task_position = frame_positions.get(self.target_task_frame)
        if task_position is None:
            return
        if self.target_origin_frame is None:
            origin_position = self._frame_position(configuration, self.origin_frame)
        else:
            origin_position = frame_positions.get(self.target_origin_frame)
            if origin_position is None:
                return
        target = np.asarray(task_position, dtype=np.float64) - np.asarray(
            origin_position, dtype=np.float64
        )
        if np.all(np.isfinite(target)) and np.linalg.norm(target) > 1.0e-8:
            self.target_vector = target

    def compute_error(
        self, configuration: PinkKinematicsConfiguration
    ) -> np.ndarray:
        return self._current_vector(configuration) - self.target_vector

    def compute_jacobian(
        self, configuration: PinkKinematicsConfiguration
    ) -> np.ndarray:
        origin_transform = configuration.get_transform(
            self.origin_frame, self.base_link_frame_name
        )
        task_transform = configuration.get_transform(
            self.task_frame, self.base_link_frame_name
        )
        origin_jacobian_local = configuration.get_frame_jacobian(
            self.origin_frame
        )[:3]
        task_jacobian_local = configuration.get_frame_jacobian(self.task_frame)[:3]
        origin_jacobian_base = origin_transform.rotation @ origin_jacobian_local
        task_jacobian_base = task_transform.rotation @ task_jacobian_local
        return task_jacobian_base - origin_jacobian_base

    def compute_qp_objective(
        self, configuration: PinkKinematicsConfiguration
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a vector-level Huber weight before building the Pink QP term."""
        error = self.compute_error(configuration)
        jacobian = self.compute_jacobian(configuration)
        error_norm = float(np.linalg.norm(error))
        robust_scale = (
            1.0
            if error_norm <= self.huber_delta
            else np.sqrt(self.huber_delta / max(error_norm, 1.0e-9))
        )
        cost = 1.0 if self.cost is None else float(self.cost)
        weighted_jacobian = robust_scale * cost * jacobian
        weighted_error = robust_scale * cost * (-self.gain * error)
        mu = self.lm_damping * float(weighted_error @ weighted_error)
        hessian = (
            weighted_jacobian.T @ weighted_jacobian
            + mu * configuration.tangent.eye
        )
        linear = -weighted_error.T @ weighted_jacobian
        return hessian, linear

    def _current_vector(
        self, configuration: PinkKinematicsConfiguration
    ) -> np.ndarray:
        return self._frame_position(
            configuration, self.task_frame
        ) - self._frame_position(configuration, self.origin_frame)

    def _frame_position(
        self, configuration: PinkKinematicsConfiguration, frame: str
    ) -> np.ndarray:
        return configuration.get_transform(
            frame, self.base_link_frame_name
        ).translation.copy()
