"""Live initialization, delayed results and independent reference support."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from core.background_detector import BackgroundDetector, advance_box, box_corners
from core.frame_trail import FrameTrail
from core.multi_target_live import IndependentPoses, Target, run_multi_target
from core import live_depth_6dof as live
from core.multi_target_tracking import advance_pose_trail, reference_improves_pose
from core.rgb_global_initializer import ReferenceView, RgbGlobalInitializer, solve_correspondences


class DeferredExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, *args, **kwargs):
        future = Future()
        self.calls.append((future, args, kwargs))
        return future

    def shutdown(self, **kwargs):
        for future, *_ in self.calls:
            future.cancel()


class HandheldInitializationTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[530., 0, 160.], [0, 530., 120.], [0, 0, 1.]])
        self.points = np.array([[x, y, (i % 3) * 10.] for i, (x, y) in enumerate(
            [(x, y) for x in [-50., 0., 50.] for y in [-40., 0., 40.]] +
            [(-30., -20.), (-30., 20.), (30., -20.), (30., 20.)])])
        self.rvec, self.tvec, self.dist = np.array([.2, -.1, .1]), np.array([0., 0., 500.]), np.zeros(5)
        self.uv = cv2.projectPoints(self.points, self.rvec, self.tvec, self.K, self.dist)[0].reshape(-1, 2)
        self.pose = solve_correspondences(self.points, self.uv, self.K, self.dist, self.points, None)
        self.assertIsNotNone(self.pose)
        self.box = np.r_[self.uv.min(0), self.uv.max(0)]
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.initializer = SimpleNamespace(K=self.K, dist=self.dist, vertices=self.points,
            new_search=lambda: SimpleNamespace(estimate=lambda *a, **k: None))

    def service(self):
        service = IndependentPoses(lambda: self.initializer)
        service.loading.result(timeout=2)
        service.executor.shutdown(wait=True)
        service.executor = DeferredExecutor()
        self.addCleanup(service.close)
        return service

    def test_pose_handoff_uses_each_intermediate_frame(self):
        service = self.service()
        target = Target(1, self.box.copy(), 0.)
        service.update([target], self.frame)
        for value in [1, 2]:
            service.update([target], np.full_like(self.frame, value), schedule=False)
        service.pending.set_result(self.pose)
        calls = []
        def track(pose, first, second, *args, **kwargs):
            calls.append((int(first[0, 0, 0]), int(second[0, 0, 0])))
            return pose
        with patch('core.rgb_global_initializer.advance_pose', side_effect=track):
            service.update([target], np.full_like(self.frame, 3), schedule=False)
        self.assertEqual(calls, [(0, 1), (1, 2), (2, 3)])
        self.assertIsNotNone(target.pose)
        self.assertEqual(service.stats['accepted'], 1)

    def test_failed_intermediate_frame_never_reuses_old_pose(self):
        with patch('core.rgb_global_initializer.advance_pose', side_effect=[self.pose, None]) as track:
            pose = advance_pose_trail(self.pose, [self.frame + i for i in range(4)], self.initializer)
        self.assertIsNone(pose)
        self.assertEqual(track.call_count, 2)

    def test_expired_pose_is_rejected_even_when_image_is_unchanged(self):
        service = self.service()
        target = Target(1, self.box.copy(), 0.)
        with patch('core.multi_target_live.time.monotonic', return_value=10.):
            service.update([target], self.frame)
        service.pending.set_result(self.pose)
        with patch('core.multi_target_live.time.monotonic', return_value=11.1):
            service.update([target], self.frame, schedule=False)
        self.assertIsNone(target.pose)
        self.assertEqual(target.initialization_failure, 'expired')

    def test_old_result_cannot_replace_current_measurement(self):
        service = self.service()
        target = Target(1, self.box.copy(), 0.)
        service.update([target], self.frame)
        old = replace(self.pose, tvec=self.pose.tvec + [8., 0., 0.])
        service.pending.set_result(old)
        target.pose = target.last_good_pose = self.pose
        target.pose_source, target.pose_time = self.frame.copy(), 10.
        service.update([target], self.frame, schedule=False, timestamp=10.)
        self.assertIs(target.pose, self.pose)
        self.assertEqual(service.stats['superseded'], 1)

    def test_search_capacity_goes_to_uninitialized_targets(self):
        service = self.service()
        healthy = Target(1, self.box.copy(), 10., pose=self.pose, last_good_pose=self.pose,
                         pose_source=self.frame, pose_time=10.)
        waiting = Target(2, self.box.copy(), 10.)
        service.update([healthy, waiting], self.frame, timestamp=10.)
        self.assertEqual(service.request[0], 2)
        service.pending.set_result(None)
        service.update([healthy], self.frame, timestamp=10.)
        self.assertIsNone(service.pending)

    def test_reference_audit_has_bounded_slot_and_requires_independent_improvement(self):
        service = self.service()
        drifted = replace(self.pose, tvec=self.pose.tvec + [8., 0., 0.])
        target = Target(1, self.box.copy(), 10., pose=drifted, last_good_pose=drifted,
                        pose_source=self.frame, pose_time=10., tracked_frames=40)
        waiting = Target(2, self.box.copy(), 10.)
        service.search_round = 3
        with patch('core.multi_target_live.time.monotonic', return_value=10.):
            service.update([target, waiting], self.frame, timestamp=10.)
            self.assertEqual(service.request[0], 1)
            self.assertTrue(service.pending_refresh)
            service.pending.set_result(self.pose)
            service.update([target, waiting], self.frame, schedule=False, timestamp=10.)
        self.assertIs(target.pose, self.pose)
        self.assertEqual(service.stats['corrected'], 1)
        self.assertFalse(reference_improves_pose(self.pose, drifted, self.initializer))
        self.assertFalse(reference_improves_pose(self.pose, self.pose, self.initializer))
        # The audit cannot immediately take the next waiting target's slot.
        service.update([target, waiting], self.frame, timestamp=10.)
        self.assertEqual(service.request[0], 2)

    def test_failed_audit_preserves_measured_pose(self):
        service = self.service()
        target = Target(1, self.box.copy(), 10., pose=self.pose, last_good_pose=self.pose,
                        pose_source=self.frame, pose_time=10., tracked_frames=40)
        service.update([target], self.frame, timestamp=10.)
        self.assertTrue(service.pending_refresh)
        service.pending.set_result(None)
        service.update([target], self.frame, schedule=False, timestamp=10.)
        self.assertIs(target.pose, self.pose)
        self.assertEqual(service.stats['corrected'], 0)

    def test_audit_handoff_never_seeds_its_own_validation_evidence(self):
        with patch('core.rgb_global_initializer.advance_pose', return_value=self.pose), \
                patch('core.multi_target_tracking.replenish_pose') as seed:
            pose = advance_pose_trail(self.pose, [self.frame + i for i in range(4)],
                                      self.initializer, replenish=False)
        self.assertIs(pose, self.pose)
        seed.assert_not_called()

    def test_history_overflow_discards_result(self):
        service = self.service()
        target = Target(1, self.box.copy(), 0.)
        service.update([target], self.frame)
        for value in range(1, 35):
            service.update([target], self.frame + value, schedule=False)
        service.pending.set_result(self.pose)
        service.update([target], self.frame + 35, schedule=False)
        self.assertIsNone(target.pose)
        self.assertEqual(service.stats['history_lost'], 1)

    def test_cross_family_support_counts_independent_references_only(self):
        rng = np.random.default_rng(123)
        akaze = rng.integers(0, 256, (13, 61), np.uint8)
        sift = rng.random((13, 128)).astype(np.float32)
        keys = [cv2.KeyPoint(float(x), float(y), 4.) for x, y in self.uv]
        for same_reference, shifted in [(False, False), (True, False), (False, True)]:
            with self.subTest(same_reference=same_reference, shifted=shifted):
                points = self.points + ([40., 0., 0.] if shifted else [0., 0., 0.])
                views = [ReferenceView('a', self.points, akaze),
                         ReferenceView('a' if same_reference else 'b', points, sift, 'sift')]
                init = RgbGlobalInitializer(self.K, self.dist, self.points, views)
                init.detector = SimpleNamespace(detectAndCompute=lambda *_: (keys, akaze))
                init.detectors['sift'] = SimpleNamespace(detectAndCompute=lambda *_: (keys, sift))
                # Default single-target behavior still requires support within a family.
                self.assertIsNone(init.estimate(self.frame, [0, 0, 320, 240]))
                search = init.new_search()
                search.cross_family_support = True
                pose = search.estimate(self.frame, [0, 0, 320, 240])
                if same_reference or shifted:
                    self.assertIsNone(pose)
                else:
                    self.assertIsNotNone(pose)
                    self.assertEqual(pose.reference_support, 2)
                    np.testing.assert_allclose(pose.tvec, self.tvec, atol=.01)

    def test_delayed_detection_advances_through_pending_frames_and_keeps_age(self):
        detector = BackgroundDetector(lambda: object(), lambda *_: [], multiple=True)
        detector.loading.result(timeout=2)
        detector.executor.shutdown(wait=True)
        detector.executor = DeferredExecutor()
        self.addCleanup(detector.close)
        with patch('core.background_detector.time.monotonic', return_value=10.):
            detector.get_boxes(self.frame)
        detector.get_boxes(self.frame + 1)
        detector.get_boxes(self.frame + 2)
        detector.pending.set_result([self.box])
        calls = []
        def flow(first, second, box, **kwargs):
            a, b = int(first[0, 0, 0]), int(second[0, 0, 0])
            calls.append((a, b))
            return box + np.array([b-a, 0, b-a, 0])
        with patch('core.background_detector.time.monotonic', return_value=10.2), \
                patch('core.background_detector.advance_box', side_effect=flow):
            boxes = detector.get_boxes(self.frame + 3)
        self.assertEqual(calls[:3], [(0, 1), (1, 2), (2, 3)])
        np.testing.assert_allclose(boxes[0], self.box + [3, 0, 3, 0])
        self.assertEqual(detector.observation[2], 10.)
        with patch('core.background_detector.time.monotonic', return_value=11.1):
            self.assertEqual(detector.get_boxes(self.frame + 4), [])

    def test_live_diagnostics_record_raw_frames_and_honor_disable(self):
        for enabled in [True, False]:
            with self.subTest(enabled=enabled), TemporaryDirectory() as directory:
                frames = iter([(True, self.frame.copy()), (True, self.frame + 20)])
                cap = SimpleNamespace(read=lambda: next(frames), release=lambda: None)
                detector = SimpleNamespace(get_boxes=lambda _: [], error=None, close=lambda: None)
                flags = ['--multi-target', '--headless', '--no-rgb-first', '--no-manual-start',
                         '--max-frames', '2', '--output-dir', directory]
                if not enabled:
                    flags.append('--no-record-diagnostics')
                args = live.parse_args(flags)
                with patch.object(live, 'resolve_camera_index', return_value=(0, 'fixture')), \
                        patch.object(live, 'open_camera', return_value=cap), \
                        patch.object(live, 'load_calibration', return_value=(self.K, self.dist, (320, 240))), \
                        patch('core.multi_target_live.BackgroundDetector', return_value=detector), \
                        patch('core.multi_target_live.draw_targets', return_value=self.frame + 255):
                    self.assertEqual(run_multi_target(args), 0)
                root = Path(directory) / 'diagnostics'
                self.assertEqual(root.exists(), enabled)
                if enabled:
                    metadata = json.loads((root / 'metadata.json').read_text())
                    self.assertEqual(metadata['recorded_frames'], 2)
                    self.assertGreater(metadata['frame_timestamps_s'][1], 0.)
                    rows = [json.loads(line) for line in (root / 'poses.jsonl').read_text().splitlines()]
                    self.assertEqual([r['video_frame_index'] for r in rows], [0, 1])
                    self.assertIn('pose_search_stats', rows[0])
                    recorded = cv2.VideoCapture(str(root / 'rgb.avi'))
                    try:
                        ok, frame = recorded.read()
                        self.assertTrue(ok)
                        self.assertLess(float(frame.mean()), 1.)  # Raw, not rendered white canvas.
                    finally:
                        recorded.release()

    def test_repeated_rotation_does_not_inflate_detector_box(self):
        source = np.zeros((360, 480, 3), np.uint8)
        source[100:260, 200:280] = np.random.default_rng(2).integers(30, 240, (160, 80, 3), np.uint8)
        original = source.copy()
        box = np.array([200, 100, 280, 260], np.float32)
        corners = box_corners(box)
        initial_corners = corners.copy()
        for angle in range(3, 61, 3):
            matrix = cv2.getRotationMatrix2D((240, 180), angle, 1.)
            frame = cv2.warpAffine(original, matrix, (480, 360))
            box = advance_box(source, frame, box, corners=corners)
            self.assertIsNotNone(box)
            source = frame
        expected_corners = cv2.transform(initial_corners[None], matrix)[0]
        expected = np.r_[expected_corners.min(0), expected_corners.max(0)]
        np.testing.assert_allclose(box, expected, atol=4.)
        saved = corners.copy()
        self.assertIsNone(advance_box(source, np.zeros_like(source), box, corners=corners))
        np.testing.assert_array_equal(corners, saved)

    def test_frame_trail_bounds_memory_and_rejects_size_change(self):
        trail = FrameTrail(self.frame, max_frames=2)
        trail.append(self.frame.copy())
        self.assertEqual(len(trail.frames), 1)
        trail.append(self.frame + 1)
        trail.append(self.frame + 2)
        self.assertFalse(trail.complete)
        self.assertEqual(trail.frames, [])
        trail.append(self.frame + 3)
        resized = FrameTrail(self.frame)
        resized.append(np.zeros((100, 100, 3), np.uint8))
        self.assertFalse(resized.complete)


if __name__ == '__main__':
    unittest.main()
