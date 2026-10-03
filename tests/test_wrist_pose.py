"""Regression checks for forearm-follow plus independent local wrist motion."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dual_arm_teleop.wrist_pose import (
    compose_hand_rotation,
    forearm_rotation,
    hand_rotation_in_mounting_frame,
    rotation_distance_rad,
)


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross


def test_straight_arm_noise_does_not_turn_into_palm_roll():
    normal = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(73)
    initial = None
    for _ in range(200):
        elbow = np.array([0.3, 0.0, 0.0])
        wrist = np.array([0.6, 0.0, 0.0]) + rng.normal(0, 0.0002, 3)
        frame = forearm_rotation(
            dict(shoulder=np.zeros(3), elbow=elbow, wrist=wrist), normal
        )
        if initial is None:
            initial = frame
        assert rotation_distance_rad(initial, frame) < math.radians(0.25)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-6)
        assert np.linalg.det(frame) > 0.99999
        normal = frame[:, 2]


def test_first_straight_frame_does_not_pick_noise_as_roll_reference():
    elbow = np.array([0.3, 0.0, 0.0])
    frames = [
        forearm_rotation(
            dict(shoulder=np.zeros(3), elbow=elbow, wrist=np.array([0.6, dy, dz]))
        )
        for dy, dz in ((0.0001, 0), (0, 0.0001), (-0.0001, 0))
    ]
    assert max(rotation_distance_rad(frames[0], f) for f in frames) < 0.002


def test_straight_robot_rest_frame_keeps_observed_normal_without_palm_flip():
    rest_points = dict(
        shoulder=np.zeros(3), elbow=np.array([0.0, 0.0, -0.3]),
        wrist=np.array([0.0, 0.0, -0.6]),
    )
    observed = np.array([1.0, 0.0, 0.0])
    rest_frame = forearm_rotation(rest_points, observed)
    np.testing.assert_allclose(rest_frame[:, 2], observed)
    assert np.linalg.det(rest_frame) > 0.99999


def test_world_hand_follows_parent_and_local_wrist_in_correct_order():
    reference = rotation([0, 1, 0], 0.7)
    mounting = rotation([1, 1, 0], 1.1)
    parent_delta = rotation([0, 0, 1], 0.8)
    local_wrist = rotation([1, 0, 0], -0.5)
    result = compose_hand_rotation(
        parent_delta @ reference, reference, mounting, local_wrist
    )
    np.testing.assert_allclose(result, parent_delta @ mounting @ local_wrist, atol=1e-6)
    # Wrist motion is local, not pre-multiplied about the world's X axis.
    assert rotation_distance_rad(result, local_wrist @ parent_delta @ mounting) > 0.1


def test_synchronized_imu_pure_parent_motion_is_counted_once():
    mounting = rotation([1, 0, 0], 0.9)
    parent_delta = rotation([0, 1, 0], 0.8)
    local = hand_rotation_in_mounting_frame(parent_delta, mounting, parent_delta)
    np.testing.assert_allclose(local, np.eye(3), atol=1e-6)
    result = compose_hand_rotation(parent_delta, np.eye(3), mounting, local)
    np.testing.assert_allclose(result, parent_delta @ mounting, atol=1e-6)


def test_separate_imu_local_motion_does_not_cancel_parent():
    mounting = rotation([1, 0, 0], 0.9)
    wrist = rotation([0, 0, 1], 0.6)
    parent = rotation([0, 1, 0], 0.8)
    local = hand_rotation_in_mounting_frame(mounting @ wrist @ mounting.T, mounting)
    result = compose_hand_rotation(parent, np.eye(3), mounting, local)
    np.testing.assert_allclose(result, parent @ mounting @ wrist, atol=1e-6)


def test_robot_rest_mounting_keeps_neutral_fingers_along_current_forearm():
    rest_forearm = rotation([0, 1, 0], math.pi / 2)
    rest_hand = rotation([1, 0, 0], math.pi)
    first_live_forearm = rotation([0, 1, 0], 0.6) @ rest_forearm
    result = compose_hand_rotation(
        first_live_forearm, rest_forearm, rest_hand, np.eye(3)
    )
    np.testing.assert_allclose(result[:, 2], first_live_forearm[:, 0], atol=1e-6)
    # Locking the first live arm frame to the USD hand pose instead would keep
    # fingers pointing down even when the first arm target is already tilted.
    assert rotation_distance_rad(result, rest_hand) > 0.5
