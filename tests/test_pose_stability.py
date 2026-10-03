"""Run these tensor regressions in env_isaaclab (system Python has no torch)."""

import math
import sys
from pathlib import Path

import pytest

try:
    import torch
except ImportError:
    torch = None
pytestmark = pytest.mark.skipif(
    torch is None, reason="Run tensor tests in env_isaaclab"
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if torch is not None:
    from dual_arm_teleop.pose_stability import PoseTargetStability


def targets():
    poses = torch.zeros(1, 4, 7)
    poses[..., 3] = 1.0
    return poses


def tracker():
    return PoseTargetStability([[0, 1], [2, 3]], [1, 3], 0.003, math.radians(1), 12)


def test_quaternion_sign_flips_and_uncontrolled_elbow_rotation_do_not_prevent_hold():
    state = tracker()
    poses = targets()
    for i in range(15):
        poses[:, [1, 3], 3:] *= -1
        poses[:, 0, 3:] = (
            torch.tensor([0.0, 1.0, 0.0, 0.0])
            if i % 2
            else torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        mask = state.update(poses)
    assert mask.tolist() == [[True, True]]


def test_slow_motion_accumulates_against_anchor_and_releases_only_moving_arm():
    state = tracker()
    poses = targets()
    for _ in range(15):
        state.update(poses)
    for i in range(1, 9):
        poses[:, 1, 0] = 0.0005 * i
        mask = state.update(poses)
    assert mask.tolist() == [[False, True]]


def test_stationary_position_and_orientation_noise_reaches_hold():
    state = tracker()
    poses = targets()
    state.update(poses)
    for i in range(20):
        sign = 1 if i % 2 else -1
        poses[:, 1, 0] = sign * 0.001
        poses[:, 1, 3:] = torch.tensor(
            [math.cos(0.003), 0.0, math.sin(sign * 0.003), 0.0]
        )
        mask = state.update(poses)
    assert mask.tolist() == [[True, True]]


def test_independent_wrist_rotation_releases_hold_with_stationary_arm_position():
    state = tracker()
    poses = targets()
    for _ in range(15):
        state.update(poses)
    poses[:, 3, 3:] = torch.tensor([math.cos(0.1), math.sin(0.1), 0.0, 0.0])
    assert state.update(poses).tolist() == [[True, False]]
    state.reset()
    assert state.update(poses).tolist() == [[False, False]]
