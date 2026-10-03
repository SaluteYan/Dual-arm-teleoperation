"""Frame math shared by robot wrist targets and the displayed human hands."""

from __future__ import annotations

import math

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm < 1.0e-6:
        return None
    return np.asarray(vector / norm, dtype=np.float32)


def forearm_rotation(
    points: dict[str, np.ndarray],
    previous_plane_normal: np.ndarray | None = None,
) -> np.ndarray | None:
    """Follow the forearm without deriving roll from an almost straight elbow.

    Below eight degrees of elbow bend the bend plane is unobservable: transport
    the previous normal onto the new forearm axis. Between eight and twenty
    degrees, smoothly restore the measured bend plane. This avoids normalizing
    a tiny, noisy cross product into an arbitrary palm rotation.
    """
    forearm = _unit(points["wrist"] - points["elbow"])
    upper = _unit(points["elbow"] - points["shoulder"])
    if forearm is None:
        return None
    transported = None
    if previous_plane_normal is not None:
        transported = _unit(
            previous_plane_normal - np.dot(previous_plane_normal, forearm) * forearm
        )
    cross = np.cross(upper, forearm) if upper is not None else np.zeros(3)
    measured = _unit(cross)
    if measured is not None and transported is not None:
        if np.dot(measured, transported) < 0.0:
            measured = -measured
        confidence = float(
            np.clip(
                (np.linalg.norm(cross) - math.sin(math.radians(8.0)))
                / (math.sin(math.radians(20.0)) - math.sin(math.radians(8.0))),
                0.0,
                1.0,
            )
        )
        confidence = confidence * confidence * (3.0 - 2.0 * confidence)
        normal = _unit((1.0 - confidence) * transported + confidence * measured)
    else:
        # A first almost straight frame cannot determine palm roll either.
        normal = transported
        if normal is None and np.linalg.norm(cross) >= math.sin(math.radians(8.0)):
            normal = measured
    if normal is None:
        for reference in np.eye(3)[[2, 1, 0]]:
            normal = _unit(np.cross(reference, forearm))
            if normal is not None:
                break
    if normal is None:
        return None
    bend = _unit(np.cross(normal, forearm))
    return np.stack([forearm, bend, normal], axis=1).astype(np.float32)


def hand_rotation_in_mounting_frame(
    hand_delta_world: np.ndarray,
    initial_hand_rotation: np.ndarray,
    arm_delta_world: np.ndarray | None = None,
) -> np.ndarray:
    """Express wrist motion in the initial hand axes, removing parent motion.

    A separately recorded, forearm-relative IMU needs no parent subtraction.
    A synchronized absolute IMU does, so pure arm motion is counted only once.
    """
    relative = hand_delta_world
    if arm_delta_world is not None:
        relative = arm_delta_world.T @ relative
    return (initial_hand_rotation.T @ relative @ initial_hand_rotation).astype(
        np.float32
    )


def compose_hand_rotation(
    current_forearm: np.ndarray,
    reference_forearm: np.ndarray,
    initial_hand_rotation: np.ndarray,
    relative_hand_rotation: np.ndarray,
) -> np.ndarray:
    """World hand pose = forearm follow * initial mounting * local wrist motion."""
    return (
        current_forearm
        @ reference_forearm.T
        @ initial_hand_rotation
        @ relative_hand_rotation
    ).astype(np.float32)


def rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first.T @ second)) - 1.0) * 0.5
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))
