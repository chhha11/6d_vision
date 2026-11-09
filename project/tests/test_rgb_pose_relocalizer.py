import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.rgb_pose_relocalizer import RgbPoseRelocalizer


class RgbPoseRelocalizerTest(unittest.TestCase):
    def setUp(self):
        search = SimpleNamespace(estimate=lambda *_: None, clear=lambda: None)
        initializer = SimpleNamespace(new_search=lambda: search, K=np.eye(3),
                                      dist=np.zeros(5), vertices=np.zeros((8, 3)))
        self.relocalizer = RgbPoseRelocalizer(initializer)
        self.frame = np.zeros((32, 32, 3), np.uint8)
        self.rvec, self.tvec = np.zeros(3), np.array([0., 0., 700.])

    def test_session_reference_is_used_by_both_search_regions(self):
        from unittest.mock import Mock
        shared = self.relocalizer.initializer
        local = SimpleNamespace(new_search=Mock(side_effect=['roi', 'global']),
                                K=shared.K, dist=shared.dist, vertices=shared.vertices)
        shared.with_reference = Mock(return_value=local)
        seeder = object()
        relocalizer = RgbPoseRelocalizer(shared, self.frame,
                                         (self.rvec, self.tvec), seeder)
        shared.with_reference.assert_called_once_with(
            self.frame, self.rvec, self.tvec, seeder)
        self.assertIs(relocalizer.initializer, local)
        self.assertEqual(relocalizer.search, 'roi')
        self.assertEqual(relocalizer.global_search, 'global')
        self.assertIs(self.relocalizer.initializer, shared)

    def confirm(self, pose, now, valid=True, allow_single=False):
        with patch('core.rgb_pose_relocalizer.advance_pose', return_value=pose):
            return self.relocalizer.confirm(pose, self.frame, self.frame, [0, 0, 32, 32],
                                            now, self.rvec, self.tvec, valid,
                                            allow_single_frame=allow_single)

    def test_two_fresh_metric_poses_can_replace_drifted_tracking(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec + [20., 0., 0.])
        self.assertIsNone(self.confirm(pose, 1.))
        self.assertIs(self.confirm(pose, 1.1), pose)

    def test_strong_single_pose_can_restore_after_hard_loss(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec + [20., 0., 0.],
                               points=np.zeros((24, 3)), rms_px=1.2,
                               inlier_ratio=.9, reference_support=2)
        self.assertIs(self.confirm(pose, 1., valid=False, allow_single=True), pose)
        self.assertEqual(self.relocalizer.stats['single_frame_accepted'], 1)

    def test_single_reference_needs_temporal_confirmation_after_loss(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec + [20., 0., 0.],
                               points=np.zeros((24, 3)), rms_px=1.2,
                               inlier_ratio=.9, reference_support=1)
        self.assertIsNone(self.confirm(pose, 1., valid=False, allow_single=True))
        self.assertIs(self.confirm(pose, 1.1, valid=False, allow_single=True), pose)

    def test_weak_single_pose_still_requires_confirmation(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec + [20., 0., 0.],
                               points=np.zeros((12, 3)), rms_px=2.2,
                               inlier_ratio=.6)
        self.assertIsNone(self.confirm(pose, 1., valid=False))
        self.assertIs(self.confirm(pose, 1.1, valid=False), pose)

    def quality_pose(self, shift=3., degrees=0., points=18, support=2, rms=1.5):
        return SimpleNamespace(
            rvec=np.array([0., np.deg2rad(degrees), 0.]),
            tvec=self.tvec + [shift, 0., 0.], points=np.zeros((points, 3)),
            rms_px=rms, inlier_ratio=.8, reference_support=support)

    def test_high_quality_small_drift_corrects_without_second_check(self):
        for shift, degrees, points, support in ((3., 0., 18, 2),
                                                (0., 1., 18, 2),
                                                (3., 0., 24, 1)):
            with self.subTest(shift=shift, degrees=degrees, support=support):
                pose = self.quality_pose(shift, degrees, points, support)
                self.assertIs(self.confirm(pose, 1.), pose)
                self.assertIsNone(self.relocalizer.pending)
        # Only the tracker that installs a pose increments accepted.
        self.assertEqual(self.relocalizer.stats['accepted'], 0)

    def test_tiny_motion_and_weak_small_drift_do_not_replace_good_tracking(self):
        for pose in (self.quality_pose(shift=1., degrees=.2),
                     self.quality_pose(points=18, support=1),
                     self.quality_pose(rms=3.)):
            with self.subTest(pose=pose):
                self.assertIsNone(self.confirm(pose, 1.))
                self.assertIsNone(self.relocalizer.pending)

    def test_large_correction_still_requires_temporal_confirmation(self):
        for shift, degrees in ((21., 0.), (0., 9.)):
            with self.subTest(shift=shift, degrees=degrees):
                self.relocalizer._clear_pending()
                pose = self.quality_pose(shift, degrees)
                self.assertIsNone(self.confirm(pose, 1.))
                self.assertIs(self.confirm(pose, 1.1), pose)

    def test_high_quality_pose_still_requires_current_frame_handoff(self):
        pose = self.quality_pose()
        with patch('core.rgb_pose_relocalizer.advance_pose', return_value=None):
            self.assertIsNone(self.relocalizer.confirm(
                pose, self.frame, self.frame, [0, 0, 32, 32], 1.,
                self.rvec, self.tvec, True))
        self.assertEqual(self.relocalizer.stats['rejected_current'], 1)

    def test_roll_only_change_does_not_replace_good_tracking(self):
        pose = SimpleNamespace(rvec=np.array([0., 0., np.pi / 2]), tvec=self.tvec.copy())
        for now in (1., 1.1, 1.2):
            self.assertIsNone(self.confirm(pose, now))
        self.assertIsNone(self.relocalizer.pending)
        self.assertEqual(self.relocalizer.stats['unchanged'], 3)

    def test_inconsistent_or_expired_confirmations_cannot_replace_pose(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec.copy())
        other = SimpleNamespace(rvec=np.array([0., -.4, 0.]), tvec=self.tvec.copy())
        self.assertIsNone(self.confirm(pose, 1.))
        self.assertIsNone(self.confirm(other, 1.1))
        self.assertIsNone(self.confirm(other, 1.8))
        self.assertIs(self.confirm(other, 1.9), other)

    def test_failed_current_frame_handoff_clears_confirmation(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec.copy())
        self.confirm(pose, 1.)
        with patch('core.rgb_pose_relocalizer.advance_pose', return_value=None):
            self.assertIsNone(self.relocalizer.confirm(
                pose, self.frame, self.frame, [0, 0, 32, 32], 1.1,
                self.rvec, self.tvec, True))
        self.assertIsNone(self.relocalizer.pending)
        self.assertEqual(self.relocalizer.stats['rejected_current'], 1)

    def test_target_disappearance_clears_confirmation(self):
        pose = SimpleNamespace(rvec=np.array([0., .4, 0.]), tvec=self.tvec.copy())
        self.confirm(pose, 1.)
        self.assertIsNone(self.confirm(None, 1.1))
        self.assertIsNone(self.confirm(pose, 1.2))


if __name__ == '__main__':
    unittest.main()
