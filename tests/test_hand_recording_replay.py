from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_esrobo_hand_recording as replay  # noqa: E402
import replay_esrobo_arm_hand_recordings as combined  # noqa: E402
from senseglove_ros_to_esrobo_hand_bridge import (  # noqa: E402
    apply_imu_axis_calibration, build_fixed_imu_axis_calibration,
    orientation_delta_local_wxyz,
)


def _raw_side(imu):
    points = [[float(index), float(index + 1), float(index + 2)] for index in range(20)]
    return {
        "hand_position_mm_used": points,
        "retargeting_features": {
            f"vector_{finger}_curl": float(index)
            for index, finger in enumerate(replay.FINGERS)
        },
        "imu_orientation_corrected_wxyz": imu,
    }


def test_dex_vector_replay_retargets_raw_hands_and_rebuilds_imu():
    left_points = _raw_side([1.0, 0.0, 0.0, 0.0])
    right_points = _raw_side([1.0, 0.0, 0.0, 0.0])
    raw = {"left": left_points, "right": right_points}
    frames = [(0.0, {"type": "esrobo_hand_joints", "hand_joints": [9.0] * 20}, raw)]
    header = {
        "calibration": {
            "hand_position_open_mm": {
                "left": left_points["hand_position_mm_used"],
                "right": right_points["hand_position_mm_used"],
            },
            "imu_neutral": {
                "left": [1.0, 0.0, 0.0, 0.0],
                "right": [1.0, 0.0, 0.0, 0.0],
            },
        }
    }
    args = SimpleNamespace(
        urdf_path=ROOT / "urdf/esrobo_waist_with_head/urdf/esrobo_waist_with_head.urdf",
        dex_iterations=1,
        dex_damping=0.025,
        dex_max_joint_step=0.16,
        dex_normal_delta=0.003,
        dex_jacobian_eps=0.0001,
        dex_direction_weight=0.35,
        dex_vector_set="fingertip",
        dex_curl_weight=0.015,
        dex_huber_delta=0.025,
        dex_joint_limit_margin=0.05,
        dex_joint_limit_weight=0.001,
    )

    fake_retargeter = mock.Mock()
    fake_retargeter.retarget.return_value = ([0.25] * 10, {"mean_error_m": 0.01})
    with mock.patch.object(
        replay, "ESROBOHandDexVectorRetargeter", return_value=fake_retargeter
    ):
        processed, references = replay.retarget_dex_vector(
            header, frames, frames, args
        )

    packet = processed[0][1]
    assert references == {"left": -1, "right": -1}
    assert packet["hand_joints"] == [0.25] * 20
    assert packet["retargeting"]["mode"] == "dex_vector"
    assert packet["hand_skeleton_source"].startswith("recorded_senseglove_raw")
    assert packet["hand_orientation_deltas"] == {
        "left": [1.0, 0.0, 0.0, 0.0],
        "right": [1.0, 0.0, 0.0, 0.0],
    }


def test_combined_legacy_imu_is_rebuilt_from_raw_with_current_fixed_axes():
    raw = {"left": _raw_side([0.98, 0.1, 0.1, 0.1]),
           "right": _raw_side([0.98, 0.0, 0.2, 0.0])}
    neutrals = dict.fromkeys(("left", "right"), [1.0, 0.0, 0.0, 0.0])
    packet = {"type": "esrobo_hand_joints", "hand_joints": [0.25] * 20,
              "hand_orientation_deltas": dict.fromkeys(("left", "right"), [1.0, 0.0, 0.0, 0.0])}
    upgraded = combined._upgrade_hand_imu_packet({"calibration": {"imu_neutral": neutrals}}, packet, raw)
    axes = build_fixed_imu_axis_calibration()
    for side in ("left", "right"):
        local = orientation_delta_local_wxyz(raw[side]["imu_orientation_corrected_wxyz"], neutrals[side])
        assert upgraded["hand_orientation_deltas"][side] == apply_imu_axis_calibration(local, axes[side], target_side=side)
    assert upgraded["hand_orientation_axis_calibrated"] is True
    assert upgraded["hand_joints"] == packet["hand_joints"]
    assert "hand_orientation_axis_calibrated" not in packet
    resent = combined._replay_packet(upgraded, 0, 0, "hand", neutrals)
    # A hand-local delta cannot be used to reconstruct a sensor-world absolute IMU.
    assert resent["hand_orientations_absolute"] == upgraded["hand_orientations_absolute"]


def test_combined_imu_upgrade_preserves_already_calibrated_and_missing_raw_packets():
    packet = {"hand_orientation_axis_calibrated": True, "hand_joints": [0.0] * 20}
    assert combined._upgrade_hand_imu_packet({}, packet, {}) is packet
    packet = {"hand_joints": [0.0] * 20}
    assert combined._upgrade_hand_imu_packet({}, packet, None) is packet
