"""Pure-Python regression tests for SenseGlove retargeting math."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from senseglove_ros_to_esrobo_hand_bridge import (  # noqa: E402
    DEFAULT_ESROBO_URDF_PATH,
    ESROBOHandDexVectorRetargeter,
    IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS,
    IMU_ROBOT_LOCAL_AXIS_SIGNS,
    apply_imu_axis_calibration,
    build_fixed_imu_axis_calibration,
    extract_hand_vector_features,
    normalize_quat_xyzw,
    quat_wxyz_to_rotvec,
    repair_senseglove_imu_w,
    rotvec_to_quat_wxyz,
    senseglove_points_with_wrist_m,
)
from replay_esrobo_arm_hand_recordings import _replay_packet  # noqa: E402


def _open_hand_points_mm() -> list[list[float]]:
    points: list[list[float]] = []
    bases = (
        (25.0, -28.0, 5.0),
        (18.0, -16.0, 12.0),
        (4.0, 0.0, 14.0),
        (-10.0, 14.0, 12.0),
        (-22.0, 26.0, 8.0),
    )
    lengths = (35.0, 42.0, 46.0, 43.0, 38.0)
    for (x, y, z), length in zip(bases, lengths):
        for fraction in (0.25, 0.50, 0.75, 1.0):
            points.append([x, y, z + length * fraction])
    return points


def _dex_args() -> SimpleNamespace:
    return SimpleNamespace(
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


class RetargetingMathTest(unittest.TestCase):
    def test_live_left_imu_maps_measured_axes_to_semantic_order(self) -> None:
        expected = np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        np.testing.assert_array_equal(
            IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS["left"], expected
        )

    def test_live_left_imu_pure_axes_follow_expected_permutation(self) -> None:
        calibration = build_fixed_imu_axis_calibration()["left"]
        angle = 0.2
        expected_axes = (2, 0, 1)
        expected_signs = (1.0, 1.0, -1.0)
        for axis in range(3):
            source = np.zeros(3, dtype=np.float64)
            source[axis] = angle
            mapped = apply_imu_axis_calibration(
                rotvec_to_quat_wxyz(source), calibration, target_side="left"
            )
            mapped_rotvec = quat_wxyz_to_rotvec(mapped)
            nonzero_axes = np.flatnonzero(np.abs(mapped_rotvec) > 1.0e-8)
            np.testing.assert_array_equal(
                nonzero_axes, np.asarray([expected_axes[axis]])
            )
            self.assertAlmostEqual(
                mapped_rotvec[expected_axes[axis]], expected_signs[axis] * angle
            )

    def test_live_left_imu_large_combined_rotation_maps_once(self) -> None:
        source_rotvec = np.asarray([0.9, -0.55, 0.7], dtype=np.float64)
        calibration = build_fixed_imu_axis_calibration()["left"]
        mapped = apply_imu_axis_calibration(
            rotvec_to_quat_wxyz(source_rotvec), calibration, target_side="left"
        )
        expected = (
            IMU_FIXED_SOURCE_TO_SEMANTIC_ROTATIONS["left"] @ source_rotvec
        ) * IMU_ROBOT_LOCAL_AXIS_SIGNS["left"]
        np.testing.assert_allclose(
            quat_wxyz_to_rotvec(mapped), expected, atol=1e-8
        )

    def test_default_dex_objective_uses_wrist_to_fingertip_vectors(self) -> None:
        retargeter = ESROBOHandDexVectorRetargeter(
            DEFAULT_ESROBO_URDF_PATH,
            "left",
            _open_hand_points_mm(),
            _dex_args(),
        )
        self.assertEqual(len(retargeter.specs), 5)
        self.assertTrue(all(spec.origin_label == "wrist" for spec in retargeter.specs))

    def test_calibrated_half_curl_maps_to_half_flexion_range(self) -> None:
        points = _open_hand_points_mm()
        current = extract_hand_vector_features(points)
        open_features = {
            f"vector_{finger}_curl": current[f"vector_{finger}_curl"] - 1.0
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        }
        closed_features = {
            f"vector_{finger}_curl": current[f"vector_{finger}_curl"] + 1.0
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        }
        retargeter = ESROBOHandDexVectorRetargeter(
            DEFAULT_ESROBO_URDF_PATH,
            "right",
            points,
            _dex_args(),
            {"open": open_features, "closed": closed_features},
        )
        targets = retargeter._curl_targets(points)
        self.assertEqual(len(targets), 5)
        for index, target in targets.items():
            expected = 0.5 * (retargeter.lower[index] + retargeter.upper[index])
            self.assertAlmostEqual(target, expected)

    def test_legacy_hand_replay_is_upgraded_to_absolute_imu_packet(self) -> None:
        neutral = {
            "left": [1.0, 0.0, 0.0, 0.0],
            "right": [1.0, 0.0, 0.0, 0.0],
        }
        packet = {
            "hand_orientation_deltas": {
                "left": [0.0, 1.0, 0.0, 0.0],
                "right": [0.0, 0.0, 1.0, 0.0],
            }
        }
        replay = _replay_packet(packet, 1, 0, "hand", neutral)
        self.assertTrue(replay["hand_orientation_is_forearm_relative"])
        self.assertEqual(
            replay["hand_orientations_absolute"], packet["hand_orientation_deltas"]
        )
        self.assertEqual(
            set(replay["hand_orientation_sample_monotonic_ns"]), {"left", "right"}
        )

    def test_repaired_quaternion_follows_previous_hemisphere(self) -> None:
        previous = normalize_quat_xyzw([0.2, -0.1, 0.3, -0.92736185])
        repaired, _raw_norm, did_repair = repair_senseglove_imu_w(
            [0.2, -0.1, 0.3, 0.3], "auto", previous
        )
        self.assertTrue(did_repair)
        normalized = normalize_quat_xyzw(repaired)
        assert previous is not None and normalized is not None
        self.assertGreater(float(np.dot(previous, normalized)), 0.0)

    def test_tip_pair_target_retains_human_closure_distance(self) -> None:
        open_points = _open_hand_points_mm()
        dex_args = _dex_args()
        dex_args.dex_vector_set = "dense"
        retargeter = ESROBOHandDexVectorRetargeter(
            DEFAULT_ESROBO_URDF_PATH, "left", open_points, dex_args
        )
        closed_points = [point.copy() for point in open_points]
        closed_points[3] = closed_points[7].copy()
        open_targets = retargeter._target_vectors(
            senseglove_points_with_wrist_m(open_points, "left")
        )
        closed_targets = retargeter._target_vectors(
            senseglove_points_with_wrist_m(closed_points, "left")
        )
        preserved = [
            index
            for index, spec in enumerate(retargeter.specs)
            if spec.preserve_relative_length
        ]
        self.assertTrue(preserved)
        self.assertTrue(
            np.any(
                np.linalg.norm(open_targets[preserved], axis=1)
                != np.linalg.norm(closed_targets[preserved], axis=1)
            )
        )

    def test_dex_target_is_finite_and_inside_joint_limits(self) -> None:
        points = _open_hand_points_mm()
        retargeter = ESROBOHandDexVectorRetargeter(
            DEFAULT_ESROBO_URDF_PATH, "right", points, _dex_args()
        )
        qpos, diagnostics = retargeter.retarget(points)
        qpos_array = np.asarray(qpos)
        self.assertTrue(np.all(np.isfinite(qpos_array)))
        self.assertTrue(np.all(qpos_array >= retargeter.lower))
        self.assertTrue(np.all(qpos_array <= retargeter.upper))
        self.assertTrue(np.isfinite(diagnostics["dex_vector_mean_error_m"]))


if __name__ == "__main__":
    unittest.main()
