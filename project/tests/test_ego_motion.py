import unittest

import numpy as np

from core.ego_motion import (EgoMotionProvider, SampledEgoMotion,
                              validate_transform)


class EgoMotionTest(unittest.TestCase):
    def test_identity_provider_has_zero_relative_motion(self):
        provider = EgoMotionProvider()
        np.testing.assert_array_equal(provider.pose_at(12.3), np.eye(4))
        np.testing.assert_array_equal(
            provider.relative_camera_transform(1., 2.), np.eye(4))

    def test_rejects_non_rigid_homogeneous_transform(self):
        with self.assertRaises(ValueError):
            validate_transform(np.diag([2., 1., 1., 1.]))
        with self.assertRaises(ValueError):
            validate_transform(np.eye(3))

    def test_sampled_motion_interpolates_translation_and_computes_relative_pose(self):
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 100.
        provider = SampledEgoMotion([(0., first), (10., second)])
        midpoint = provider.pose_at(5.)
        self.assertEqual(midpoint[0, 3], 50.)
        relative = provider.relative_camera_transform(0., 10.)
        self.assertEqual(relative[0, 3], -100.)


if __name__ == '__main__':
    unittest.main()
