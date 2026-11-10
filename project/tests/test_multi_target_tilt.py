"""Regression coverage for silhouette changes and actual perspective rotation."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import cv2
import numpy as np

from core.cad_video_tracker import project
from core.multi_target_live import TargetSet, IndependentPoses
from core.multi_target_tracking import needs_replenishment, replenish_pose
from core.rgb_global_initializer import solve_correspondences, advance_pose


class MultiTargetTiltTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[530., 0, 320.], [0, 530., 240.], [0, 0, 1.]])
        self.dist = np.zeros(5)
        self.points = np.array([[x, y, 0.] for x in np.linspace(-65, 65, 7)
                                for y in np.linspace(-45, 45, 5)])
        self.rvec = np.array([.05, .1, .05])
        self.tvec = np.array([0., 0., 500.])
        pixels = project(self.points, self.rvec, self.tvec, self.K, self.dist)
        self.pose = solve_correspondences(self.points, pixels, self.K, self.dist, self.points, None)
        self.assertIsNotNone(self.pose)
        self.frame = np.random.default_rng(123).integers(0, 256, (480, 640, 3), np.uint8)
        self.box = np.r_[pixels.min(0), pixels.max(0)].astype(np.float32)

    def test_measured_track_wins_over_duplicate_with_higher_iou(self):
        tracker = TargetSet()
        measured, duplicate = tracker.update(self.frame, [self.box, self.box + [12, 0, 12, 0]], 1.)
        measured.pose = measured.last_good_pose = self.pose
        measured.pose_source, measured.pose_time = self.frame, 1.
        observed = duplicate.box.copy()
        with patch('core.multi_target_live.advance_box', side_effect=lambda a, b, box: box):
            visible = tracker.update(self.frame, [observed], 1.04)
        self.assertEqual([t.id for t in visible], [measured.id])
        self.assertIs(measured.pose, self.pose)
        self.assertTrue(measured.detected)
        self.assertFalse(duplicate.detected)

    def test_measured_tracks_still_match_one_to_one(self):
        tracker = TargetSet()
        first, second, duplicate = tracker.update(
            self.frame, [self.box, self.box + [200, 0, 200, 0], self.box + [10, 0, 10, 0]], 1.)
        for target in [first, second]:
            target.pose = target.last_good_pose = self.pose
            target.pose_source, target.pose_time = self.frame, 1.
        boxes = [second.box.copy(), duplicate.box.copy()]
        with patch('core.multi_target_live.advance_box', side_effect=lambda a, b, box: box):
            visible = tracker.update(self.frame, boxes, 1.04)
        self.assertEqual({t.id for t in visible}, {first.id, second.id})
        np.testing.assert_array_equal(first.box, boxes[1])
        np.testing.assert_array_equal(second.box, boxes[0])

    def test_rotation_and_attrition_refresh_before_fixed_interval(self):
        steady = replace(self.pose, points=np.tile(self.pose.points, (2, 1)))
        self.assertFalse(needs_replenishment(steady, steady, steady.rvec, 1))
        rotated = replace(steady, rvec=np.array([.05, .4, .05]))
        self.assertTrue(needs_replenishment(rotated, steady, steady.rvec, 1))
        reduced = replace(steady, points=steady.points[:40])
        self.assertTrue(needs_replenishment(reduced, steady, steady.rvec, 1))
        few = replace(steady, points=steady.points[:20])
        self.assertTrue(needs_replenishment(few, few, few.rvec, 1))

    def test_service_refreshes_after_one_frame_but_never_on_failed_pose(self):
        initial = replace(self.pose, points=np.tile(self.pose.points, (2, 1)),
                          pixels=np.tile(self.pose.pixels, (2, 1)))
        reduced = replace(self.pose, points=self.pose.points[:20], pixels=self.pose.pixels[:20])
        initializer = SimpleNamespace(K=self.K, dist=self.dist, vertices=self.points,
                                      tracking_seeder=object())
        service = IndependentPoses(lambda: initializer)
        service.loading.result(timeout=2)
        self.addCleanup(service.close)
        target = TargetSet().update(self.frame, [self.box], 1.)[0]
        target.pose = target.last_good_pose = initial
        target.pose_source = self.frame
        target.pose_time = 1.
        target.seeded_at = target.tracked_frames = 1
        target.seeded_rvec = initial.rvec
        current = np.roll(self.frame, 1, axis=1)
        with patch('core.rgb_global_initializer.advance_pose', return_value=reduced), \
                patch('core.multi_target_live.replenish_pose', return_value=reduced) as seed:
            service.update([target], current, schedule=False, timestamp=1.04)
        seed.assert_called_once()
        with patch('core.rgb_global_initializer.advance_pose', return_value=None), \
                patch('core.multi_target_live.replenish_pose') as seed:
            service.update([target], np.roll(current, 1, axis=1), schedule=False, timestamp=1.08)
        seed.assert_not_called()

    def test_planar_tracking_uses_current_measurements_without_pose_prior(self):
        with patch('core.rgb_global_initializer.solve_correspondences', return_value=self.pose) as solve:
            result = advance_pose(self.pose, self.frame, self.frame, self.K, self.dist,
                                  self.points, None, retry_pnp=True)
        self.assertIsNotNone(result)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.kwargs['pnp_method'], cv2.SOLVEPNP_ITERATIVE)
        self.assertIs(solve.call_args.kwargs['prior_pose'], self.pose)
        np.testing.assert_allclose(result.motion_shift_px, [0., 0.], atol=1e-6)
        self.assertNotIn('initial_pose', solve.call_args.kwargs)
        self.assertIsNone(solve_correspondences(
            self.points[:11], self.pose.pixels[:11], self.K, self.dist,
            self.points, None, pnp_method=cv2.SOLVEPNP_ITERATIVE))
        # A blank current image cannot be accepted by keeping the prior pose.
        self.assertIsNone(advance_pose(self.pose, self.frame, np.zeros_like(self.frame),
                                       self.K, self.dist, self.points, None, retry_pnp=True))

    def test_real_lk_pnp_tracks_out_of_plane_tilt(self):
        # A textured physical plane is rendered at known 3D rotations. This is
        # perspective foreshortening, not a rotated axis-aligned screen box.
        plane = np.array([[-80., -60., 0], [80., -60., 0], [80., 60., 0], [-80., 60., 0]])
        corners = project(plane, self.rvec, self.tvec, self.K, self.dist).astype(np.float32)
        K, dist = self.K, self.dist
        class PlaneSeeder:
            def seed(self, gray, vertices, rvec, tvec, count):
                projected = project(plane, rvec, tvec, K, dist).astype(np.float32)
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, np.rint(projected).astype(np.int32), 255)
                mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
                pixels = cv2.goodFeaturesToTrack(gray, count, .008, 5, mask=mask)
                if pixels is None:
                    return np.empty((0, 3)), np.empty((0, 2))
                inverse = cv2.getPerspectiveTransform(projected, plane[:, :2].astype(np.float32))
                xy = cv2.perspectiveTransform(pixels, inverse).reshape(-1, 2)
                return np.c_[xy, np.zeros(len(xy))], pixels.reshape(-1, 2)
        initializer = SimpleNamespace(vertices=self.points, tracking_seeder=PlaneSeeder())
        pose = replenish_pose(self.pose, self.frame, initializer)
        source, seeded_rvec, age = self.frame, pose.rvec, 0
        for angle in np.linspace(.12, .95, 35):
            rotation = np.array([.05, angle, .05])
            destination = project(plane, rotation, self.tvec, self.K, self.dist).astype(np.float32)
            warp = cv2.getPerspectiveTransform(corners, destination)
            current = cv2.warpPerspective(self.frame, warp, (640, 480))
            previous = pose
            pose = advance_pose(pose, source, current, self.K, self.dist, self.points, None, retry_pnp=True)
            self.assertIsNotNone(pose, f'tilt angle {angle}')
            age += 1
            if needs_replenishment(pose, previous, seeded_rvec, age):
                pose = replenish_pose(pose, current, initializer)
                seeded_rvec, age = pose.rvec, 0
            source = current
        actual = project(self.points, pose.rvec, pose.tvec, self.K, self.dist)
        expected = project(self.points, rotation, self.tvec, self.K, self.dist)
        self.assertLess(np.linalg.norm(actual - expected, axis=1).mean(), 1.)
        self.assertGreater(np.linalg.norm(pose.rvec - self.rvec), .7)


if __name__ == '__main__':
    unittest.main()
