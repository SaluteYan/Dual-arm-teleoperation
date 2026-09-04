from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts import senseglove_ros_to_esrobo_hand_bridge as bridge_module


def _pose_result(marker: float, imu=None):
    features = {
        side: {f"vector_{finger}_curl": marker for finger in bridge_module.FINGERS}
        for side in ("left", "right")
    }
    if imu is None:
        imu = {
            "left": [1.0, 0.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0, 0.0],
        }
    points = {
        side: [[marker, 0.0, 0.0] for _ in range(20)]
        for side in ("left", "right")
    }
    return features, imu, points


def test_live_calibration_collects_neutral_open_and_closed(tmp_path):
    args = SimpleNamespace(
        calibration_file=tmp_path / "calibration.json",
        calibration_method="last",
        calibration_sample_start=6.0,
        calibration_duration=10.0,
        imu_correction="unity_to_ros",
        imu_w_repair="auto",
    )
    pose_results = [
        _pose_result(0.4),
        _pose_result(0.5),
    ]

    with mock.patch.object(bridge_module, "wait_for_both_hands"), mock.patch.object(
        bridge_module, "collect_calibration_pose", side_effect=pose_results
    ) as collect:
        calibration = bridge_module.collect_and_save_calibration(
            mock.Mock(), mock.Mock(), args
        )

    assert [call.kwargs["label"] for call in collect.call_args_list] == [
        "1/2 neutral open hands",
        "2/2 closed fists",
    ]
    assert collect.call_args_list[0].kwargs["require_imu"] is True
    assert calibration["imu_neutral"] == pose_results[0][1]
    assert calibration["open"] == pose_results[0][0]
    assert calibration["hand_position_open_mm"] == pose_results[0][2]
    assert calibration["closed"] == pose_results[1][0]
    assert calibration["method"] == "two_pose_neutral_open_and_closed_fixed_imu_axes"
    assert calibration["imu_calibration"]["captured_with"] == "open_hand_reference"
    assert calibration["imu_calibration"]["mandatory_each_live_run"] is True
    assert set(calibration["imu_axis_calibration"]) == {"left", "right"}


def test_fixed_imu_axis_calibration_applies_robot_side_signs():
    calibration = bridge_module.build_fixed_imu_axis_calibration()
    for side in ("left", "right"):
        mapping = bridge_module.IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS[side]
        for source_axis in range(3):
            source = np.zeros(3)
            source[source_axis] = 0.2
            corrected = bridge_module.apply_imu_axis_calibration(
                bridge_module.rotvec_to_quat_wxyz(source),
                calibration[side],
                target_side=side,
            )
            expected_robot_target = bridge_module.IMU_ROBOT_LOCAL_AXIS_SIGNS[
                side
            ] * (mapping @ source)
            np.testing.assert_allclose(
                bridge_module.quat_wxyz_to_rotvec(corrected),
                expected_robot_target,
                atol=1.0e-7,
            )


def test_fixed_imu_axis_mapping_does_not_change_between_runs():
    first = bridge_module.build_fixed_imu_axis_calibration()
    second = bridge_module.build_fixed_imu_axis_calibration()
    assert first == second
    for side in ("left", "right"):
        assert first[side]["source"] == "fixed_nova2_glove_mount"


def test_imu_action_directions_match_measured_robot_wrist_directions():
    expected = {
        "left": np.asarray([1.0, -1.0, 1.0]),
        "right": np.asarray([-1.0, -1.0, 1.0]),
    }
    for side, signs in expected.items():
        np.testing.assert_array_equal(
            bridge_module.IMU_ROBOT_LOCAL_AXIS_SIGNS[side], signs
        )


def test_source_side_follows_target_swap_setting():
    assert bridge_module.source_side_for_target("left", False) == "left"
    assert bridge_module.source_side_for_target("right", False) == "right"
    assert bridge_module.source_side_for_target("left", True) == "right"
    assert bridge_module.source_side_for_target("right", True) == "left"
