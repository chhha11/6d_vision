"""Motion regressions with real LK/PnP and bounded asynchronous observations."""
import unittest
from dataclasses import replace
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.background_detector import BackgroundDetector
from core.cad_video_tracker import project
from core.multi_target_live import (IndependentPoses, TargetSet, lock_long_axis_twist,
                                    target_record)
from core.multi_target_tracking import replenish_pose
from core.rgb_global_initializer import solve_correspondences
from core.rgb_pose_recovery import (directed_long_axis_jump_deg,
                                    pose_has_continuous_long_axis)
from scipy.spatial.transform import Rotation


class MultiTargetMotionTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[530., 0, 320.], [0, 530., 240.], [0, 0, 1.]])
        self.dist = np.zeros(5)
        self.points = np.array([[x, y, ((x+y) % 30)]
                                for x in (-60., -30., 0., 30., 60.)
                                for y in (-40., 0., 40.)])
        pixels = project(self.points, np.array([.2, -.15, .1]),
                         np.array([20., -15., 600.]), self.K, self.dist)
        self.pose = solve_correspondences(self.points, pixels, self.K, self.dist,
                                          self.points, None)
        self.assertIsNotNone(self.pose)
        self.source = np.random.default_rng(49).integers(0, 255, (480, 640, 3), np.uint8)
        self.box = np.r_[pixels.min(0), pixels.max(0)].astype(np.float32)
        self.initializer = SimpleNamespace(K=self.K, dist=self.dist, vertices=self.points)

    def test_display_pose_ignores_rotation_about_model_long_axis(self):
        base_rotation = Rotation.from_rotvec(self.pose.rvec.reshape(3))
        twist_rotation = base_rotation * Rotation.from_euler('z', 73., degrees=True)
        measured = replace(self.pose, rvec=twist_rotation.as_rotvec())
        displayed = lock_long_axis_twist(measured, self.pose)

        np.testing.assert_allclose(displayed.rvec, self.pose.rvec, atol=1e-10)
        np.testing.assert_allclose(displayed.tvec, measured.tvec)
        # The metric measurement is retained for tracking and diagnostics.
        self.assertFalse(np.allclose(measured.rvec, self.pose.rvec))

    def test_display_pose_keeps_long_axis_direction_when_object_moves(self):
        base_rotation = Rotation.from_rotvec(self.pose.rvec.reshape(3))
        measured_rotation = (base_rotation *
                             Rotation.from_euler('y', 18., degrees=True) *
                             Rotation.from_euler('z', 41., degrees=True))
        measured = replace(self.pose, rvec=measured_rotation.as_rotvec())
        displayed = lock_long_axis_twist(measured, self.pose)
        displayed_rotation = Rotation.from_rotvec(displayed.rvec.reshape(3))

        np.testing.assert_allclose(displayed_rotation.as_matrix()[:, 2],
                                   measured_rotation.as_matrix()[:, 2], atol=1e-10)
        relative = base_rotation.inv() * displayed_rotation
        # The remaining rotation is the minimum swing from the old long axis;
        # its rotation axis is perpendicular to that axis, so it has no roll.
        np.testing.assert_allclose(relative.as_rotvec()[2], 0., atol=1e-10)

    def test_display_pose_unwraps_reversed_long_axis_without_flipping_xy(self):
        base_rotation = Rotation.from_rotvec(self.pose.rvec.reshape(3))
        # A symmetric/elongated target can produce an equivalent PnP solution
        # with its local long axis reversed. Treat that as a sign ambiguity for
        # display, rather than as a real 180-degree object motion.
        measured_rotation = (base_rotation *
                             Rotation.from_euler('y', 180., degrees=True))
        measured = replace(self.pose, rvec=measured_rotation.as_rotvec())

        displayed = lock_long_axis_twist(measured, self.pose)
        displayed_rotation = Rotation.from_rotvec(displayed.rvec.reshape(3))

        np.testing.assert_allclose(displayed_rotation.as_matrix(),
                                   base_rotation.as_matrix(), atol=1e-10)

    def test_directed_long_axis_gate_rejects_the_opposite_pnp_branch(self):
        # The object is viewed broadside, so a 180-degree branch produces the
        # same elongated silhouette while reversing the model +Z direction.
        base_rotation = Rotation.from_euler('y', 90., degrees=True)
        opposite_rotation = base_rotation * Rotation.from_euler('x', 180., degrees=True)
        tvec = np.array([0., 0., 600.])
        jump = directed_long_axis_jump_deg(
            base_rotation.as_rotvec(), tvec,
            opposite_rotation.as_rotvec(), tvec, self.K, self.dist)

        self.assertAlmostEqual(jump, 180., delta=1e-6)
        self.assertFalse(pose_has_continuous_long_axis(
            replace(self.pose, rvec=base_rotation.as_rotvec(), tvec=tvec),
            replace(self.pose, rvec=opposite_rotation.as_rotvec(), tvec=tvec),
            self.K, self.dist))

    def test_display_pose_does_not_reintroduce_accumulated_roll(self):
        base_rotation = Rotation.from_rotvec(self.pose.rvec.reshape(3))
        first_measured_rotation = (base_rotation *
                                   Rotation.from_euler('z', 67., degrees=True))
        first_measured = replace(
            self.pose, rvec=first_measured_rotation.as_rotvec())
        first_displayed = lock_long_axis_twist(first_measured, self.pose)

        second_measured_rotation = (first_measured_rotation *
                                    Rotation.from_euler('y', 18., degrees=True) *
                                    Rotation.from_euler('z', 23., degrees=True))
        second_measured = replace(
            self.pose, rvec=second_measured_rotation.as_rotvec())
        second_displayed = lock_long_axis_twist(second_measured, first_displayed)
        second_displayed_rotation = Rotation.from_rotvec(
            second_displayed.rvec.reshape(3))
        relative = (Rotation.from_rotvec(first_displayed.rvec.reshape(3)).inv() *
                    second_displayed_rotation)

        np.testing.assert_allclose(
            second_displayed_rotation.as_matrix()[:, 2],
            second_measured_rotation.as_matrix()[:, 2], atol=1e-10)
        np.testing.assert_allclose(relative.as_rotvec()[2], 0., atol=1e-10)

    def test_measured_motion_survives_detector_and_box_flow_gap_then_hides_on_removal(self):
        tracker = TargetSet()
        target = tracker.update(self.source, [self.box], 10.)[0]
        target.pose = target.last_good_pose = target.display_pose = self.pose
        target.pose_source, target.pose_box, target.pose_time = self.source, self.box.copy(), 10.
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        # A full second without a detector hit exceeds the previous .3s/6-frame
        # cutoff. Real image correspondences, not a held display, support motion.
        with patch('core.multi_target_live.advance_box', return_value=None):
            for index in range(1, 31):
                frame = cv2.warpAffine(self.source,
                    np.array([[1., 0., 2.*index], [0., 1., index]]), (640, 480))
                now = 10. + index/30.
                visible = tracker.update(frame, [], now)
                with patch('core.multi_target_live.time.monotonic', return_value=now):
                    service.update(visible, frame, schedule=False)
                self.assertEqual([t.id for t in visible], [target.id])
                self.assertIsNotNone(target.pose, f'frame {index}')
                self.assertFalse(target.detected)
                self.assertFalse(target.display_held)
                self.assertEqual(target.last_seen, 10.)
            actual = project(self.points, target.pose.rvec, target.pose.tvec, self.K, self.dist)
            expected = project(self.points, self.pose.rvec, self.pose.tvec, self.K, self.dist) + [60, 30]
            self.assertLess(np.linalg.norm(actual-expected, axis=1).mean(), 1.)
            self.assertGreater(np.linalg.norm(target.pose.tvec-self.pose.tvec), 30.)
            self.assertAlmostEqual(target.box[0]-self.box[0], 60., delta=2.)
            blank = np.zeros_like(self.source)
            visible = tracker.update(blank, [], 11.04)
            with patch('core.multi_target_live.time.monotonic', return_value=11.04):
                service.update(visible, blank, schedule=False)
            self.assertIsNone(target.pose)
            self.assertIsNone(target.display_pose)
            self.assertIsNone(target_record(target, self.initializer)['tvec_mm'])
            self.assertEqual(tracker.update(blank, [], 11.08), [])

    def test_update_rejects_opposite_long_axis_branch_and_keeps_last_good_pose(self):
        tracker = TargetSet()
        target = tracker.update(self.source, [self.box], 10.)[0]
        base_rotation = Rotation.from_euler('y', 90., degrees=True)
        base_pose = replace(self.pose, rvec=base_rotation.as_rotvec(),
                            tvec=np.array([0., 0., 600.]))
        target.pose = target.last_good_pose = base_pose
        target.pose_source, target.pose_box, target.pose_time = (
            self.source, self.box.copy(), 10.)
        opposite_rotation = base_rotation * Rotation.from_euler('x', 180., degrees=True)
        opposite = replace(base_pose, rvec=opposite_rotation.as_rotvec())
        shifted = cv2.warpAffine(self.source,
                                 np.array([[1., 0., 2.], [0., 1., 1.]]),
                                 (640, 480))
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        with patch('core.rgb_global_initializer.advance_pose', return_value=opposite), \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            service.update([target], shifted, schedule=False, timestamp=10.03)
        self.assertIsNone(target.pose)
        self.assertIs(target.last_good_pose, base_pose)
        self.assertEqual(target.tracking_failure, 'long_axis_branch')
        self.assertAlmostEqual(target.long_axis_jump_deg, 180., delta=1e-6)
        self.assertEqual(service.stats['long_axis_branch'], 1)

    def test_temporal_pose_update_keeps_last_measured_box_as_geometry_guard(self):
        tracker = TargetSet()
        target = tracker.update(self.source, [self.box], 10.)[0]
        target.pose = target.last_good_pose = self.pose
        target.pose_source, target.pose_box, target.pose_time = (
            self.source, self.box.copy(), 10.)
        shifted = cv2.warpAffine(self.source,
                                  np.array([[1., 0., 2.], [0., 1., 1.]]),
                                  (640, 480))
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        with patch('core.rgb_global_initializer.advance_pose', return_value=self.pose) as advance, \
                patch('core.multi_target_live.detection_matches_pose', return_value=True):
            service.update([target], shifted, schedule=False, timestamp=10.03)
        self.assertIsNotNone(advance.call_args)
        np.testing.assert_array_equal(advance.call_args.args[6], self.box)

    def test_oversized_detector_box_cannot_replace_measured_track(self):
        tracker = TargetSet()
        target = tracker.update(self.source, [self.box], 10.)[0]
        target.pose = target.last_good_pose = self.pose
        target.pose_source, target.pose_box, target.pose_time = self.source, self.box.copy(), 10.
        center = (self.box[:2]+self.box[2:])*.5
        size = self.box[2:]-self.box[:2]
        merged = np.r_[center-size, center+size]
        visible = tracker.update(self.source, [merged], 10.05)
        self.assertIn(target.id, [t.id for t in visible])
        self.assertFalse(target.detected)
        np.testing.assert_allclose(target.box, self.box, atol=1.)
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        service.update(visible, self.source, schedule=False, timestamp=10.05)
        self.assertIsNotNone(target.pose)
        self.assertEqual(sum(t.pose is not None for t in visible), 1)

    def test_capture_clock_controls_recovery_not_slow_replay_wall_clock(self):
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        tracker = TargetSet()
        target = tracker.update(self.source, [self.box], 1.)[0]
        target.pose = None
        target.last_good_pose = self.pose
        target.pose_source, target.pose_box, target.pose_time = self.source, self.box.copy(), 1.
        shifted = cv2.warpAffine(self.source, np.array([[1., 0., 2.], [0., 1., 1.]]), (640, 480))
        with patch('core.multi_target_live.time.monotonic', return_value=1000.):
            service.update([target], shifted, schedule=False, timestamp=1.03)
        self.assertIsNotNone(target.pose)
        self.assertEqual(target.pose_time, 1.03)

    def test_replenish_preserves_measurement_and_excludes_duplicate_points(self):
        extra_pixels = np.array([self.pose.pixels[0], [400., 350.], [410., 350.], [420., 350.]])
        extra_points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [np.nan, 0, 0]])
        self.initializer.tracking_seeder = SimpleNamespace(seed=lambda *a: (extra_points, extra_pixels))
        refreshed = replenish_pose(self.pose, self.source, self.initializer)
        self.assertEqual(len(refreshed.points), len(self.pose.points)+2)
        np.testing.assert_array_equal(refreshed.points[:len(self.pose.points)], self.pose.points)
        np.testing.assert_array_equal(refreshed.rvec, self.pose.rvec)
        np.testing.assert_array_equal(refreshed.tvec, self.pose.tvec)
        self.assertEqual(refreshed.reference_support, self.pose.reference_support)
        self.assertEqual(refreshed.rms_px, self.pose.rms_px)
        self.assertEqual(len(self.pose.points), 15)

    def test_background_boxes_advance_from_adjacent_frame_without_renewing_age(self):
        detector = BackgroundDetector(lambda: object(), lambda *_: [], multiple=True, max_age_s=1.)
        self.addCleanup(detector.close)
        detector.loading.result(timeout=2)
        detector.pending = Future()  # No detector refresh during this interval.
        detector.observation = (self.source, [self.box], 10.)
        second = np.full_like(self.source, 2)
        third = np.full_like(self.source, 3)
        with patch('core.background_detector.advance_box', return_value=self.box+1) as flow:
            with patch('core.background_detector.time.monotonic', return_value=10.2):
                detector.get_boxes(second)
            with patch('core.background_detector.time.monotonic', return_value=10.4):
                detector.get_boxes(third)
            np.testing.assert_array_equal(flow.call_args.args[0], second)
            np.testing.assert_array_equal(flow.call_args.args[1], third)
            self.assertEqual(detector.observation[2], 10.)
            with patch('core.background_detector.time.monotonic', return_value=11.01):
                self.assertEqual(detector.get_boxes(third), [])
            self.assertEqual(flow.call_count, 2)
        detector.pending = None

    def test_recovery_slots_alternate_with_normal_round_robin(self):
        calls = []
        def new_search():
            return SimpleNamespace(estimate=lambda frame, box, **kw: calls.append(box[0]))
        self.initializer.new_search = new_search
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        tracker = TargetSet()
        targets = tracker.update(self.source, [[10, 20, 50, 80], [100, 20, 140, 80],
                                               [200, 20, 240, 80]], 10.)
        targets[-1].last_good_pose = self.pose
        # Old measurement cannot be displayed, but its lost target gets recovery
        # priority. Normal candidates still receive half of the search slots.
        for _ in range(4):
            service.update(targets, self.source)
            service.pending.result(timeout=2)
        self.assertEqual(calls, [200., 10., 200., 100.])


if __name__ == '__main__':
    unittest.main()
