"""Regression coverage for detector gaps and per-target pose recovery."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.multi_target_live import IndependentPoses, Target, TargetSet, target_record
from core.rgb_global_initializer import RgbPose


class MultiTargetStabilityTest(unittest.TestCase):
    def setUp(self):
        self.first = np.zeros((120, 160, 3), np.uint8)
        self.second = np.full_like(self.first, 1)
        self.third = np.full_like(self.first, 2)
        self.box = np.array([30., 20., 90., 100.], np.float32)
        self.pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0., 0., 700.]))
        self.initializer = SimpleNamespace(
            K=np.eye(3), dist=np.zeros(5),
            vertices=np.zeros((8, 3)))

    def established(self):
        target = Target(1, self.box.copy(), 10., pose=self.pose)
        target.last_good_pose = self.pose
        target.pose_source = self.first
        target.pose_box = self.box.copy()
        target.pose_time = 10.
        target.display_pose = self.pose
        return target

    def service(self):
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        service.previous = self.first
        self.addCleanup(service.close)
        return service

    def test_slow_frame_does_not_expire_continuously_detected_target(self):
        tracker = TargetSet()
        first = tracker.update(self.first, [self.box], 10.)[0]
        with patch('core.multi_target_live.advance_box', return_value=self.box.copy()):
            current = tracker.update(self.second, [self.box], 12.)[0]
        self.assertIs(current, first)
        self.assertEqual(current.id, 1)
        self.assertEqual(current.last_seen, 12.)
        tracker.update(self.third, [], 12.1)
        reentered = tracker.update(self.third, [self.box], 14.)[0]
        self.assertNotEqual(reentered.id, first.id)

    def test_short_detector_gap_keeps_measured_track_but_expires(self):
        target = self.established()
        tracker = TargetSet()
        tracker.targets = {1: target}
        tracker.next_id = 2
        tracker.previous = self.first
        search = target.search = object()
        with patch('core.multi_target_live.advance_box', return_value=self.box.copy()):
            visible = tracker.update(self.second, [], 10.1)
            self.assertEqual([t.id for t in visible], [1])
            self.assertFalse(target.detected)
            self.assertIs(target.search, search)
            self.assertEqual(target.last_seen, 10.)
            self.assertEqual(tracker.update(self.third, [], 10.31), [])
        self.assertIsNone(target.pose)
        self.assertIsNone(target.display_pose)
        self.assertIsNone(target.search)

    def test_failed_box_flow_still_allows_pose_measurement_but_never_holds_without_evidence(self):
        target = self.established()
        tracker = TargetSet()
        tracker.targets = {1: target}
        tracker.previous = self.first
        service = self.service()
        with patch('core.multi_target_live.advance_box', return_value=None), \
                patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                patch('core.multi_target_live.time.monotonic', return_value=10.01):
            visible = tracker.update(self.second, [], 10.01)
            self.assertEqual([t.id for t in visible], [target.id])
            service.update(visible, self.second, schedule=False)
            self.assertIsNone(target.pose)
            self.assertIsNone(target.display_pose)
            self.assertEqual(tracker.update(self.third, [], 10.02), [])
        self.assertIsNone(target.last_good_pose)

    def test_one_failed_pose_holds_display_then_retries_from_measured_source(self):
        target, service = self.established(), self.service()
        target.display_pose = SimpleNamespace(rvec=np.array([0., 0., .7]),
                                             tvec=self.pose.tvec.copy())
        with patch('core.multi_target_live.time.monotonic', return_value=10.05), \
                patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                patch('core.multi_target_live.advance_box', return_value=self.box.copy()), \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            service.update([target], self.second, schedule=False)
        self.assertIsNone(target.pose)
        self.assertTrue(target.display_held)
        self.assertIs(target.display_pose, self.pose)
        record = target_record(target, self.initializer)
        self.assertEqual(record['state'], 'detected')
        self.assertIsNone(record['tvec_mm'])
        self.assertTrue(record['display_visible'])
        recovered = SimpleNamespace(rvec=np.ones(3)*.01, tvec=self.pose.tvec.copy())
        with patch('core.multi_target_live.time.monotonic', return_value=10.10), \
                patch('core.rgb_global_initializer.advance_pose', return_value=recovered) as advance, \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            service.update([target], self.third, schedule=False)
        np.testing.assert_array_equal(advance.call_args.args[1], self.first)
        np.testing.assert_array_equal(advance.call_args.args[2], self.third)
        self.assertIs(target.pose, recovered)
        self.assertIs(target.display_pose, recovered)
        self.assertFalse(target.display_held)
        np.testing.assert_array_equal(target.pose_source, self.third)

    def test_expiry_motion_and_detector_conflict_do_not_hold(self):
        cases = ((10.3, self.box.copy(), True),
                 (10.05, self.box+10, True),
                 (10.05, self.box.copy(), False))
        for now, moved, matched in cases:
            with self.subTest(now=now, moved=moved, matched=matched):
                target, service = self.established(), self.service()
                with patch('core.multi_target_live.time.monotonic', return_value=now), \
                        patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                        patch('core.multi_target_live.advance_box', return_value=moved), \
                        patch('core.multi_target_live.detection_matches_pose', return_value=matched):
                    service.update([target], self.second, schedule=False)
                self.assertIsNone(target.display_pose)
                self.assertFalse(target.display_held)

    def test_hold_is_bounded_by_frame_count_as_well_as_time(self):
        target, service = self.established(), self.service()
        with patch('core.multi_target_live.time.monotonic', return_value=10.05), \
                patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                patch('core.multi_target_live.advance_box', return_value=self.box.copy()), \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            for value in range(1, 8):
                service.update([target], np.full_like(self.first, value), schedule=False)
                self.assertEqual(target.display_held, value <= 6)

    def test_verified_pose_keeps_association_box_separate_from_cad_box(self):
        from core.cad_video_tracker import project

        self.initializer.K = np.array([[530., 0., 80.],
                                       [0., 530., 60.], [0., 0., 1.]])
        self.initializer.vertices = np.array(
            [[-30., -20., -20.], [30., -20., -20.],
             [30., 20., -20.], [-30., 20., -20.],
             [-30., -20., 20.], [30., -20., 20.],
             [30., 20., 20.], [-30., 20., 20.]])
        target, service = self.established(), self.service()
        target.raw_detection_box = self.box.copy()
        target.filtered_box = self.box.copy()
        target.pose_box = self.box.copy()
        recovered = RgbPose(
            np.zeros(3), np.array([0., 0., 700.]),
            self.initializer.vertices.copy(),
            project(self.initializer.vertices, np.zeros(3),
                    np.array([0., 0., 700.]), self.initializer.K, self.initializer.dist),
            'test', 1., 1., 0.)
        with patch('core.rgb_global_initializer.advance_pose', return_value=recovered), \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            service.update([target], self.second, schedule=False, timestamp=10.1)

        np.testing.assert_array_equal(target.box, self.box)
        self.assertIsNotNone(target.pose_projection_box)
        self.assertFalse(np.allclose(target.pose_projection_box, target.box))
        record = target_record(target, self.initializer)
        self.assertEqual(record['state'], 'pose_verified')
        np.testing.assert_allclose(record['association_box_xyxy'], self.box)
        np.testing.assert_allclose(record['filtered_box_xyxy'], self.box)
        self.assertIsNotNone(record['cad_box_xyxy'])

    def test_predicted_pose_is_bounded_and_does_not_refresh_measurement_time(self):
        from core.cad_video_tracker import project

        self.initializer.K = np.array([[530., 0., 80.],
                                       [0., 530., 60.], [0., 0., 1.]])
        self.initializer.vertices = np.array(
            [[-30., -20., -20.], [30., -20., -20.],
             [30., 20., -20.], [-30., 20., -20.],
             [-30., -20., 20.], [30., -20., 20.],
             [30., 20., 20.], [-30., 20., 20.]])
        target, service = self.established(), self.service()
        target.raw_detection_box = self.box.copy()
        target.filtered_box = self.box.copy()
        target.pose_box = self.box.copy()
        target.pose_status = 'pose_verified'
        target.pose = target.last_good_pose = RgbPose(
            np.zeros(3), np.array([0., 0., 700.]),
            self.initializer.vertices.copy(),
            project(self.initializer.vertices, np.zeros(3),
                    np.array([0., 0., 700.]), self.initializer.K, self.initializer.dist),
            'test', 1., 1., 0.)
        target.pose_time = 10.
        # Change width as a horizontal target can do when its edge mask
        # jitters.  This must not be interpreted as depth motion during the
        # short prediction fallback.
        moved = self.box + np.array([5., 0., 10., 0.], np.float32)
        with patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                patch('core.multi_target_live.advance_box', return_value=moved), \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            for index, value in enumerate((10.1, 10.2, 10.3), 1):
                service.update([target], np.full_like(self.first, index),
                               schedule=False, timestamp=value)
                self.assertEqual(target.pose_status, 'pose_predicted')
                self.assertEqual(target.pose_prediction_frames, index)
                self.assertEqual(target_record(target, self.initializer)['state'],
                                 'pose_predicted')
                self.assertEqual(target.pose_time, 10.)
                # Width-only jitter is not a reliable depth cue for a
                # horizontal target, so it must leave the predicted depth
                # unchanged.
                self.assertAlmostEqual(float(target.pose.tvec[2]), 700.)

            service.update([target], np.full_like(self.first, 4),
                           schedule=False, timestamp=10.4)

        self.assertIsNone(target.pose)
        self.assertEqual(target.pose_status, 'detected')
        self.assertEqual(target.pose_prediction_frames, 3)
        self.assertEqual(target.pose_time, 10.)

    def test_predicted_pose_follows_consistent_extent_scale_slowly(self):
        from core.cad_video_tracker import project

        self.initializer.K = np.array([[530., 0., 80.],
                                       [0., 530., 60.], [0., 0., 1.]])
        self.initializer.vertices = np.array(
            [[-30., -20., -20.], [30., -20., -20.],
             [30., 20., -20.], [-30., 20., -20.],
             [-30., -20., 20.], [30., -20., 20.],
             [30., 20., 20.], [-30., 20., 20.]])
        from core.multi_target_live import _predict_pose_from_box

        pose = RgbPose(
            np.zeros(3), np.array([0., 0., 700.]),
            self.initializer.vertices.copy(),
            project(self.initializer.vertices, np.zeros(3),
                    np.array([0., 0., 700.]), self.initializer.K, self.initializer.dist),
            'test', 1., 1., 0.)
        # Both axes grow together, as they do when the target slowly moves
        # closer.  The damped prediction should follow that motion.
        moved = self.box + np.array([5., 0., 10., 10.], np.float32)
        predicted = _predict_pose_from_box(pose, self.box, moved, self.initializer.K)

        self.assertIsNotNone(predicted)
        self.assertLess(float(predicted.tvec[2]), float(pose.tvec[2]))
        self.assertGreater(float(predicted.tvec[2]), 600.)

    def test_invalid_calibration_clears_held_contour_too(self):
        target, service = self.established(), self.service()
        service.image_size = (640, 480)
        service.update([target], self.second, schedule=False)
        self.assertIsNone(target.pose)
        self.assertIsNone(target.display_pose)
        self.assertIsNone(target.last_good_pose)


if __name__ == '__main__':
    unittest.main()
