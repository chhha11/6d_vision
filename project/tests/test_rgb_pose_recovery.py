import unittest
from dataclasses import replace

import cv2
import numpy as np

from core import cad_video_tracker as tracker
from core.reference_pose_corrector import (CONTROL_POINTS, axis_angle,
                                            make_projection_mask, project_points)
from core.rgb_pose_recovery import (RgbPoseRecovery, axis_angle_difference,
                                    axis_matches_pose, observe_long_axis)


class RgbPoseRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = cv2.imdecode(np.fromfile(tracker.DEFAULT_STATIC_REFERENCE_IMAGE,
                                           np.uint8), cv2.IMREAD_COLOR)
        cls.rvec, cls.tvec = tracker.load_pose(tracker.DEFAULT_POSE)
        cls.K, cls.dist, _ = tracker.load_calibration(tracker.DEFAULT_INTRINSICS)
        cls.vertices, _, _ = tracker.make_edge_points(tracker.DEFAULT_MODEL, 700, 3)
        cls.mask = make_projection_mask(cls.frame.shape, cls.vertices, cls.rvec,
                                        cls.tvec, cls.K, cls.dist, erode_px=0)
        cls.background = cv2.inpaint(cls.frame, cls.mask, 5, cv2.INPAINT_TELEA)
        cls.center = project_points(CONTROL_POINTS[:1], cls.rvec, cls.tvec,
                                     cls.K, cls.dist)[0]

    def recovery(self):
        return RgbPoseRecovery(self.frame, self.rvec, self.tvec,
                               self.vertices, self.K, self.dist)

    def moved_object(self, angle):
        matrix = cv2.getRotationMatrix2D(tuple(self.center), angle, 1.0)
        size = self.frame.shape[1::-1]
        moved = cv2.warpAffine(self.frame, matrix, size)
        mask = cv2.warpAffine(self.mask, matrix, size)
        scene = self.background.copy()
        scene[mask > 0] = moved[mask > 0]
        return scene, matrix

    def test_object_rotates_45_and_90_degrees_while_background_stays_fixed(self):
        for degrees in (45, 90):
            with self.subTest(degrees=degrees):
                recovery = self.recovery()
                frame, affine = self.moved_object(degrees)
                observation = recovery.observe(frame)
                proposal = recovery.proposal(observation, self.rvec, self.tvec)
                self.assertIsNotNone(proposal)
                self.assertTrue(recovery.fast_recovery_ready(proposal))
                self.assertTrue(recovery.consistent(proposal, proposal))
                rvec, tvec = recovery.recover(proposal)
                actual = project_points(CONTROL_POINTS, rvec, tvec, self.K, self.dist)
                expected = cv2.transform(recovery.corrector.reference_controls[None],
                                         affine)[0]
                self.assertLess(abs(axis_angle_difference(axis_angle(actual),
                                                           axis_angle(expected))), 1.0)
                self.assertLess(np.linalg.norm(actual[0] - expected[0]), 2.0)
                self.assertLess(abs(tvec[2] - self.tvec[2]), 5.0)

    def test_unchanged_and_removed_object_cannot_trigger_correction(self):
        recovery = self.recovery()
        for frame in (self.frame, self.background, np.zeros_like(self.frame)):
            observation = recovery.observe(frame)
            self.assertIsNone(recovery.proposal(observation, self.rvec, self.tvec))

    def test_observation_box_follows_independent_reference_match(self):
        recovery = self.recovery()
        frame, _ = self.moved_object(45)
        observation = recovery.observe(frame)
        box = recovery.observation_box(observation, frame.shape)
        self.assertIsNotNone(box)
        self.assertGreater(box[2], box[0])
        self.assertGreater(box[3], box[1])
        self.assertTrue(np.all(observation.control_points[0] >= box[:2]))
        self.assertTrue(np.all(observation.control_points[0] <= box[2:]))

    def test_strong_appearance_verifies_pose_but_rejects_drift_or_removal(self):
        recovery = self.recovery()
        observation = recovery.observe(self.frame)
        self.assertTrue(recovery.verifies_pose(observation, self.rvec, self.tvec))
        self.assertFalse(recovery.verifies_pose(
            replace(observation, inliers=19), self.rvec, self.tvec))
        self.assertFalse(recovery.verifies_pose(
            replace(observation, spread_px2=20.), self.rvec, self.tvec))
        self.assertFalse(recovery.verifies_pose(
            observation, self.rvec, self.tvec + [25., 0., 0.]))
        self.assertFalse(recovery.verifies_pose(
            observation, self.rvec, self.tvec * 1.2))
        self.assertFalse(recovery.verifies_pose(
            recovery.observe(self.background), self.rvec, self.tvec))

    def test_repeated_small_patch_and_conflicting_detector_box_are_rejected(self):
        recovery = self.recovery()
        frame, _ = self.moved_object(90)
        observation = recovery.observe(frame)
        self.assertIsNotNone(observation)
        patch = replace(observation, spread_px2=20.0)
        self.assertIsNone(recovery.proposal(patch, self.rvec, self.tvec))
        self.assertIsNone(recovery.proposal(observation, self.rvec, self.tvec,
                                            np.array([0, 0, 30, 30])))

    def test_axis_is_undirected_across_180_degree_boundary(self):
        self.assertAlmostEqual(axis_angle_difference(179, -1), 0)
        self.assertAlmostEqual(axis_angle_difference(-89, 89), 2)

    def test_independent_image_axis_rejects_diagonal_pose_on_upright_target(self):
        frame = np.full((240, 320, 3), 180, np.uint8)
        for x in (125, 140, 160, 180):
            cv2.line(frame, (x, 50), (x, 195), (20, 20, 20), 4)
        observed = observe_long_axis(frame, [115, 40, 190, 205])
        self.assertIsNotNone(observed)
        K = np.array([[500., 0, 160], [0, 500., 120], [0, 0, 1]])
        upright = np.array([np.pi / 2, 0, 0])
        tilt, _ = cv2.Rodrigues(np.array([0., 0, np.pi / 4]))
        rotation, _ = cv2.Rodrigues(upright)
        diagonal, _ = cv2.Rodrigues(tilt @ rotation)
        self.assertTrue(axis_matches_pose(observed, upright, np.array([0., 0, 700]), K, np.zeros(5)))
        self.assertFalse(axis_matches_pose(observed, diagonal, np.array([0., 0, 700]), K, np.zeros(5)))
        self.assertIsNone(observe_long_axis(np.zeros_like(frame), [115, 40, 190, 205]))


if __name__ == '__main__':
    unittest.main()
