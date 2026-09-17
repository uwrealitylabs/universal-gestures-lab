import unittest

import numpy as np

from intensity_scorer import (
    ABDUCTION_IDX,
    ANGLE_IDX,
    CURL_IDX,
    FLEXION_IDX,
    OPPOSITION_IDX,
    extract_features,
)


class ExtractFeaturesTests(unittest.TestCase):
    def test_feature_groups_cover_all_inputs_once(self):
        indices = CURL_IDX + ABDUCTION_IDX + FLEXION_IDX + OPPOSITION_IDX

        self.assertEqual(sorted(indices), list(range(17)))
        self.assertEqual(len(indices), len(set(indices)))

    def test_static_sequence_has_no_motion(self):
        frames = np.ones((3, 17), dtype=np.float32)

        self.assertEqual(extract_features(frames), (0.0, 0.0, 0.0))

    def test_known_changes_use_degrees_and_metres_per_second(self):
        frames = np.zeros((2, 17), dtype=np.float32)
        frames[1, ANGLE_IDX] = 3.0
        frames[1, OPPOSITION_IDX] = 0.01

        angular_speed, angular_accel, opposition_speed = extract_features(frames)

        self.assertAlmostEqual(angular_speed, 90.0, places=4)
        self.assertEqual(angular_accel, 0.0)
        self.assertAlmostEqual(opposition_speed, 0.3, places=6)

    def test_constant_angular_speed_has_no_acceleration(self):
        frames = np.zeros((3, 17), dtype=np.float32)
        frames[1, ANGLE_IDX] = 3.0
        frames[2, ANGLE_IDX] = 6.0

        angular_speed, angular_accel, _ = extract_features(frames)

        self.assertAlmostEqual(angular_speed, 90.0, places=4)
        self.assertAlmostEqual(angular_accel, 0.0)


if __name__ == "__main__":
    unittest.main()
