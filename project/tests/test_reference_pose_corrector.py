import unittest
import cv2
import numpy as np

from core import cad_video_tracker as tracker
from core.cad_pose_overlay import load_calibration
from core.paths import DATA_ROOT
from core.reference_pose_corrector import ReferencePoseCorrector


ROOT = DATA_ROOT


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


class ReferencePoseCorrectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = read_image(tracker.DEFAULT_STATIC_REFERENCE_IMAGE)
        cls.live_image = read_image(ROOT / 'samples/energy_unit/akaze_scale_regression.png')
        cls.K, cls.dist, _ = load_calibration(tracker.DEFAULT_INTRINSICS)
        cls.rvec, cls.tvec = tracker.load_pose(tracker.DEFAULT_STATIC_REFERENCE_POSE)
        cls.vertices, _, _ = tracker.make_edge_points(tracker.DEFAULT_MODEL, 700, 3)
        cls.corrector = ReferencePoseCorrector(
            cls.image, cls.rvec, cls.tvec, cls.vertices, cls.K, cls.dist)

    def test_recovers_confirmed_view_affine_motion(self):
        matrix = cv2.getRotationMatrix2D((320, 240), 5, .95)
        matrix[:, 2] += [12, -7]
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpAffine(gray, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
        observation = self.corrector.observe(warped)
        self.assertIsNotNone(observation)
        self.assertGreaterEqual(observation.inliers, 40)
        self.assertGreaterEqual(observation.inlier_ratio, .7)
        expected = cv2.transform(self.corrector.reference_controls[None], matrix)[0]
        error = np.linalg.norm(observation.control_points - expected, axis=1)
        self.assertLess(float(error.max()), 1.5)

    def test_rejects_unconfirmed_view_by_confidence_gate(self):
        names = [
            'frame_20261029_153640_623299.png',
            'frame_20261029_153706_573194.png',
            'frame_20261029_153714_109307.png',
            'frame_20261029_153748_158035.png',
        ]
        for name in names:
            with self.subTest(name=name):
                other = read_image(ROOT / 'samples/energy_unit' / name)
                search_mask = tracker.make_search_mask(
                    other.shape, self.vertices, self.rvec, self.tvec,
                    self.K, self.dist, margin=80)
                observation = self.corrector.observe(
                    cv2.cvtColor(other, cv2.COLOR_BGR2GRAY), search_mask)
                self.assertTrue(observation is None or
                                observation.inliers < tracker.STATIC_MIN_MATCH_INLIERS)

    def test_accepts_live_view_with_large_translation_and_scale_change(self):
        self.assertIsNotNone(self.live_image)
        search_mask = tracker.make_search_mask(
            self.live_image.shape, self.vertices, self.rvec, self.tvec,
            self.K, self.dist, margin=80)
        observation = self.corrector.observe(
            cv2.cvtColor(self.live_image, cv2.COLOR_BGR2GRAY), search_mask)
        self.assertIsNotNone(observation)
        self.assertGreaterEqual(observation.inliers, tracker.STATIC_MIN_MATCH_INLIERS)
        self.assertGreaterEqual(observation.inlier_ratio, tracker.STATIC_MIN_INLIER_RATIO)
        self.assertGreaterEqual(observation.spread_px2, tracker.STATIC_MIN_SPREAD_PX2)
        center, angle, scale = self.corrector.residual(observation, self.rvec, self.tvec)
        self.assertLess(np.linalg.norm(center), tracker.STATIC_MAX_CENTER_ERROR_PX)
        self.assertLess(abs(angle), tracker.STATIC_MAX_ANGLE_ERROR_DEG)
        self.assertLess(abs(scale - 1.0), tracker.STATIC_MAX_SCALE_ERROR)

        corrected_rvec, corrected_tvec = self.rvec.copy(), self.tvec.copy()
        for _ in range(20):
            corrected_rvec, corrected_tvec, _ = self.corrector.apply(
                observation, corrected_rvec, corrected_tvec, gain=.25,
                max_center_step_px=5.0, max_angle_step_deg=1.0,
                max_depth_step_mm=5.0)
        final_center, final_angle, final_scale = self.corrector.residual(
            observation, corrected_rvec, corrected_tvec)
        self.assertLess(np.linalg.norm(final_center), 1.0)
        self.assertLess(abs(final_angle), .1)
        self.assertLess(abs(final_scale - 1.0), .01)

    def test_correction_steps_are_bounded_and_reduce_residual(self):
        matrix = cv2.getRotationMatrix2D((320, 240), 3, .97)
        matrix[:, 2] += [10, -5]
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        warped = cv2.warpAffine(gray, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
        observation = self.corrector.observe(warped)
        before = self.corrector.residual(observation, self.rvec, self.tvec)
        rvec, tvec, diagnostics = self.corrector.apply(
            observation, self.rvec, self.tvec, gain=.25)
        after = self.corrector.residual(observation, rvec, tvec)
        self.assertLess(np.linalg.norm(after[0]), np.linalg.norm(before[0]))
        self.assertLessEqual(abs(diagnostics['center_step_px']), 2.0)
        self.assertLessEqual(abs(diagnostics['angle_step_deg']), .5)
        self.assertLessEqual(abs(diagnostics['depth_step_mm']), 1.0)

    def test_lock_anchor_detects_slow_rotation_without_accumulating_static_noise(self):
        center = np.array([320.0, 240.0])
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        anchor = center + np.column_stack((np.cos(angles), np.sin(angles))) * 100.0

        small_rotation = cv2.getRotationMatrix2D(tuple(center), .1, 1.0)
        first_step = cv2.transform(anchor.astype(np.float32)[None], small_rotation)[0]
        shift, motion = tracker.anchor_motion(anchor, first_step)
        self.assertLess(np.linalg.norm(shift), .05)
        self.assertLess(motion, tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX)

        accumulated_rotation = cv2.getRotationMatrix2D(tuple(center), 1.0, 1.0)
        rotated = cv2.transform(anchor.astype(np.float32)[None], accumulated_rotation)[0]
        shift, motion = tracker.anchor_motion(anchor, rotated)
        self.assertLess(np.linalg.norm(shift), .05)
        self.assertGreater(motion, tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX)

        noisy_static = anchor + np.tile([.08, -.06], (len(anchor), 1))
        _, motion = tracker.anchor_motion(anchor, noisy_static)
        self.assertLess(motion, tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX)


if __name__ == '__main__':
    unittest.main()
