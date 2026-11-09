import itertools
import csv
import json
import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core import cad_video_tracker as tracker


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


class FakeCapture:
    def __init__(self, frame_count, frame_factory):
        self.frame_count = frame_count
        self.frame_factory = frame_factory
        self.index = 0

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 20.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.frame_count)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.index * 50.0
        return 0.0

    def read(self):
        if self.index >= self.frame_count:
            return False, None
        frame = self.frame_factory(self.index)
        self.index += 1
        return True, frame

    def release(self):
        pass


class CadBoxDisplayTest(unittest.TestCase):
    def test_pose_display_holds_brief_invalid_frames_and_recovers(self):
        state = tracker.PoseDisplayState(grace_s=1.0, max_hold_frames=3)
        rvec = np.zeros(3)
        tvec = np.array([0., 0., 700.])
        self.assertEqual(state.update('TRACKING', 0.0, rvec, tvec)[0], 'TRACKING')
        self.assertTrue(state.update('VERIFYING POSE', .1, rvec, tvec)[2])
        self.assertTrue(state.update('SEEDED', .2, rvec, tvec)[2])
        display_state, pose, held = state.update('TRACKING', .3, rvec, tvec)
        self.assertEqual(display_state, 'TRACKING')
        self.assertFalse(held)
        self.assertIsNotNone(pose)

    def test_pose_display_hides_after_bounded_loss(self):
        state = tracker.PoseDisplayState(grace_s=1.0, max_hold_frames=2)
        rvec = np.zeros(3)
        tvec = np.array([0., 0., 700.])
        state.update('TRACKING', 0.0, rvec, tvec)
        state.update('FLOW LOST', .1, rvec, tvec)
        state.update('FLOW LOST', .2, rvec, tvec)
        display_state, pose, held = state.update('FLOW LOST', .3, rvec, tvec)
        self.assertEqual(display_state, 'RECOVERING')
        self.assertIsNone(pose)
        self.assertFalse(held)

    def test_pose_display_expires_by_time_and_confirms_return(self):
        state = tracker.PoseDisplayState()
        rvec, tvec = np.zeros(3), np.array([0., 0., 700.])
        state.update('TRACKING', 0., rvec, tvec)
        self.assertIsNone(state.update('SEEDED', .26, rvec, tvec)[1])
        self.assertIsNone(state.update('TRACKING', .28, rvec, tvec)[1])
        self.assertIsNotNone(state.update('TRACKING', .30, rvec, tvec)[1])
        # Confirmed loss never displays the old pose during the grace period.
        label, pose, held = state.update('TARGET LOST', .31, rvec, tvec)
        self.assertEqual(label, 'SEARCHING')
        self.assertIsNone(pose)
        self.assertFalse(held)
        self.assertIsNone(state.update('SEEDED', .32, rvec, tvec)[1])

    def test_brief_display_hold_does_not_publish_a_measured_pose(self):
        state = tracker.PoseDisplayState()
        rvec, tvec = np.zeros(3), np.array([0., 0., 700.])
        state.update('TRACKING', 0., rvec, tvec)
        self.assertIsNotNone(state.update('VERIFYING POSE', .05, rvec, tvec)[1])
        self.assertFalse(tracker.make_xyz_record(
            1, .05, 'VERIFYING POSE', rvec, tvec)['valid'])

    def test_detector_verification_latches_only_repeated_conflicts(self):
        verification = tracker.PoseVerification()
        verification.update(False)
        self.assertFalse(verification.latched)
        verification.update(False)
        self.assertTrue(verification.latched)
        verification.update(True)
        self.assertTrue(verification.latched)
        verification.update(True)
        self.assertFalse(verification.latched)

    def test_target_check_interval_prioritizes_motion_and_lost_pose(self):
        interval = tracker.target_check_interval
        self.assertEqual(interval(8, 'TRACKING', 0.1, 0.3), 8)
        self.assertEqual(interval(8, 'TRACKING', 1.2, 3.2), 4)
        self.assertEqual(interval(8, 'TRACKING', 0.1, 0.3, conflicts=1), 2)
        self.assertEqual(interval(8, 'FLOW LOST', 0., 0.), 2)
        self.assertEqual(interval(1, 'FLOW LOST', 0., 0.), 1)

    def test_inconclusive_checks_do_not_clear_a_conflict(self):
        verification = tracker.PoseVerification()
        verification.update(False)
        verification.update(None)
        self.assertEqual(verification.conflicts, 1)
        verification.update(False)
        verification.update(None)
        self.assertTrue(verification.latched)

    def test_detector_associates_existing_target_before_confidence(self):
        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def cpu(self):
                return self

            def numpy(self):
                return self.array

            def argmax(self):
                return self.array.argmax()

        class Boxes:
            conf = Tensor([.90, .25])
            xyxy = Tensor([[0, 0, 70, 80], [310, 205, 395, 348]])

            def __len__(self):
                return 2

        detector = SimpleNamespace(predict=lambda *_a, **_kw: [SimpleNamespace(boxes=Boxes())])
        frame = np.zeros((480, 640, 3), np.uint8)
        np.testing.assert_array_equal(tracker.detect_target_box(
            detector, frame, expected_box=[320, 208, 392, 329]), [310, 205, 395, 348])

    def test_fast_detector_uses_single_lower_cost_inference(self):
        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)

            def cpu(self):
                return self

            def numpy(self):
                return self.array

            def argmax(self):
                return self.array.argmax()

        class Boxes:
            conf = Tensor([.9])
            xyxy = Tensor([[100, 100, 200, 220]])

            def __len__(self):
                return 1

        calls = []
        detector = SimpleNamespace(
            predict=lambda *_a, **kwargs: (calls.append(kwargs) or
                                            [SimpleNamespace(boxes=Boxes())]))
        frame = np.zeros((480, 640, 3), np.uint8)
        tracker.detect_target_box(detector, frame, fast=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['imgsz'], 512)

    def test_lost_pose_search_can_use_projected_fallback_box(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        expected = np.array([20, 30, 80, 100], np.float32)
        seen = []

        class Search:
            def estimate(self, _frame, box):
                seen.append(np.asarray(box))
                return None

        result = tracker.check_target(frame, None, None, 0, expected, Search(),
                                      expected, True)
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(seen[0], expected)

    def test_hard_loss_search_uses_full_frame_without_detector_box(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        seen = []

        class Search:
            def estimate(self, _frame, box, validation_box=None):
                seen.append((np.asarray(box), validation_box))
                return None

        result = tracker.check_target(frame, None, None, 0, [20, 30, 80, 100],
                                      Search(), [20, 30, 80, 100], True,
                                      global_search=True)
        self.assertEqual(len(result), 5)
        np.testing.assert_array_equal(seen[0][0], [0, 0, 160, 120])
        self.assertIsNone(seen[0][1])

    def test_hard_loss_search_does_not_validate_against_detector_box(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        seen = []

        class Search:
            def estimate(self, _frame, box, validation_box=None):
                seen.append((np.asarray(box), validation_box))
                return None

        class Detector:
            def predict(self, *_args, **_kwargs):
                class Tensor:
                    def cpu(self):
                        return self

                    def numpy(self):
                        return np.array([[20, 30, 80, 100]], np.float32)

                    def argmax(self):
                        return SimpleNamespace(item=lambda: 0)

                class Boxes:
                    xyxy = Tensor()
                    conf = Tensor()

                    def __len__(self):
                        return 1

                return [SimpleNamespace(boxes=Boxes())]

        tracker.check_target(frame, Detector(), None, 0, [20, 30, 80, 100], Search(),
                             [20, 30, 80, 100], True, global_search=True)
        np.testing.assert_array_equal(seen[0][0], [0, 0, 160, 120])
        self.assertIsNone(seen[0][1])

    def test_hard_loss_metric_pose_skips_detector_latency(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        pose = object()
        seen = []

        class Search:
            def estimate(self, _frame, box, validation_box=None):
                seen.append((box, validation_box))
                return pose

        detector = SimpleNamespace(predict=lambda *_a, **_k:
                                   self.fail('metric pose must not wait for YOLO'))
        result = tracker.check_target(frame, detector, None, 0, None, Search(),
                                      global_search=True)
        np.testing.assert_array_equal(seen[0][0], [0, 0, 160, 120])
        self.assertIsNone(seen[0][1])
        self.assertIs(result[3], pose)
        self.assertIsNone(result[0])
        self.assertIsNone(result[4])

    def test_hard_loss_without_pose_falls_back_to_detector_once(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        search = SimpleNamespace(estimate=lambda *_a, **_k: None)
        with patch.object(tracker, 'detect_target_box', return_value=None) as detect:
            result = tracker.check_target(frame, object(), None, 0, None, search,
                                          global_search=True)
        detect.assert_called_once()
        self.assertEqual(len(result), 5)
        self.assertIsNone(result[3])

    def test_hard_loss_failed_roi_searches_full_frame_in_same_check(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        stale_box = np.array([20, 30, 80, 100], np.float32)
        detector = SimpleNamespace(get_box=lambda *_a, **_k: stale_box)
        pose = object()
        for global_pose in (pose, None):
            with self.subTest(global_pose=global_pose):
                search = SimpleNamespace(estimate=Mock(side_effect=[None, global_pose]))
                result = tracker.check_target(
                    frame, detector, None, 0, stale_box, search,
                    global_search=True, prefer_roi=True)
                self.assertEqual(search.estimate.call_count, 2)
                roi, full = search.estimate.call_args_list
                np.testing.assert_array_equal(roi.args[1], stale_box)
                np.testing.assert_array_equal(roi.kwargs['validation_box'], stale_box)
                np.testing.assert_array_equal(full.args[1], [0, 0, 160, 120])
                self.assertIsNone(full.kwargs['validation_box'])
                self.assertIs(result[3], global_pose)
                if global_pose is not None:
                    # A failed ROI is not evidence against the global pose,
                    # and the search extent is not a target validation box.
                    self.assertIsNone(result[0])
                    self.assertIsNone(result[4])
                else:
                    np.testing.assert_array_equal(result[4], [0, 0, 160, 120])

    def test_hard_loss_successful_roi_skips_full_frame(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        box = np.array([20, 30, 80, 100], np.float32)
        detector = SimpleNamespace(get_box=lambda *_a, **_k: box)
        pose = object()
        search = SimpleNamespace(estimate=Mock(return_value=pose))
        result = tracker.check_target(frame, detector, None, 0, box, search,
                                      global_search=True, prefer_roi=True)
        search.estimate.assert_called_once()
        self.assertIs(result[3], pose)
        np.testing.assert_array_equal(result[4], box)

    def test_xyz_record_is_gated_by_tracking_status(self):
        rvec = np.array([0.1, -0.2, 0.3])
        center = np.array([12.5, -8.0, 700.0])
        valid = tracker.make_xyz_record(7, 0.35, 'TRACKING', rvec, center)
        self.assertEqual(valid['center_xyz_mm'], [12.5, -8.0, 700.0])
        self.assertEqual(valid['rvec_rad'], [0.1, -0.2, 0.3])
        self.assertEqual(valid['coordinate_frame'], 'rgb_camera_optical')
        self.assertTrue(valid['valid'])
        invalid = tracker.make_xyz_record(8, 0.4, 'FLOW LOST', rvec, center)
        self.assertFalse(invalid['valid'])
        self.assertIsNone(invalid['center_xyz_mm'])
        self.assertIsNone(invalid['rvec_rad'])

    def test_status_overlay_omits_stale_xyz(self):
        frame = np.zeros((120, 640, 3), dtype=np.uint8)
        with patch.object(cv2, 'putText', wraps=cv2.putText) as put_text:
            tracker.draw_status(frame, np.zeros_like(frame), 'FLOW LOST', 0, 0, 20,
                                center_xyz_mm=None)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertTrue(any('XYZ' in label and 'unavailable' in label for label in labels))
        self.assertFalse(any('700' in label for label in labels))

    def test_box_uses_exact_model_bounds(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        self.assertEqual(corners.shape, (8, 3))
        self.assertEqual(len(np.unique(corners, axis=0)), 8)
        np.testing.assert_allclose(corners.min(axis=0), [-47.5, -47.4341774, -75], atol=1e-5)
        np.testing.assert_allclose(corners.max(axis=0), [47.5, 47.5, 75], atol=1e-5)

    def test_box_renders_only_twelve_edges_and_hides_invalid_pose(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        K = np.array([[200, 0, 100], [0, 200, 100], [0, 0, 1]], np.float64)
        dist = np.zeros(5, np.float64)
        with patch.object(cv2, 'line', wraps=cv2.line) as draw_line:
            layer = tracker.render_box_layer(
                (200, 200, 3), corners, K, dist,
                np.zeros(3), np.array([0, 0, 500], np.float64))
        self.assertEqual(draw_line.call_count, 12)
        self.assertGreater(np.count_nonzero(layer), 0)
        self.assertFalse(layer[100, 100].any())
        hidden = tracker.render_box_layer(
            (200, 200, 3), corners, K, dist,
            np.zeros(3), np.array([0, 0, 10], np.float64))
        self.assertFalse(hidden.any())

    def test_display_offset_moves_box_pixels_without_changing_shape(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        K = np.array([[200, 0, 100], [0, 200, 100], [0, 0, 1]], np.float64)
        dist = np.zeros(5, np.float64)
        pose = np.array([0, 0, 500], np.float64)
        baseline = tracker.render_box_layer(
            (200, 200, 3), corners, K, dist, np.zeros(3), pose, 0)
        shifted = tracker.render_box_layer(
            (200, 200, 3), corners, K, dist, np.zeros(3), pose, 9)
        np.testing.assert_array_equal(baseline[:-9], shifted[9:])
        self.assertFalse(shifted[:9].any())

    def test_side_pose_uses_unshifted_box_by_default(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        K = np.array([[400, 0, 160], [0, 400, 120], [0, 0, 1]], np.float64)
        # A horizontal/side-mounted pose makes a fixed screen Y compensation
        # visible as a direct gap between the model and the object.
        rvec = np.array([1.8957456254, 0.6460221359, 1.7851894111])
        tvec = np.array([18.2794577, -3.0712642, 514.3849136])
        default = tracker.render_box_layer(
            (240, 320, 3), corners, K, np.zeros(5), rvec, tvec)
        unshifted = tracker.render_box_layer(
            (240, 320, 3), corners, K, np.zeros(5), rvec, tvec, 0)
        np.testing.assert_array_equal(default, unshifted)

    def test_live_box_uses_measured_roll_instead_of_a_level_surrogate(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        K = np.array([[400, 0, 160], [0, 400, 120], [0, 0, 1]], np.float64)
        rvec = np.array([0., 0., np.deg2rad(35.)])
        tvec = np.array([0., 0., 500.])
        measured = tracker.render_box_layer((240, 320, 3), corners, K, np.zeros(5),
                                            rvec, tvec, tracker.BOX_DISPLAY_Y_OFFSET_PX, neutralize_roll=False)
        level = tracker.render_box_layer((240, 320, 3), corners, K, np.zeros(5),
                                         rvec, tvec, tracker.BOX_DISPLAY_Y_OFFSET_PX, neutralize_roll=True)
        default = tracker.render_box_layer((240, 320, 3), corners, K, np.zeros(5),
                                           rvec, tvec)
        np.testing.assert_array_equal(default, measured)
        self.assertGreater(np.count_nonzero(measured), 0)
        self.assertFalse(np.array_equal(measured, level))

    def test_tracking_support_rejects_a_rail_despite_low_reprojection_error(self):
        K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], np.float64)
        vertices = np.array([[x, y, z] for x in (-60., 60.)
                             for y in (-40., 40.) for z in (-75., 75.)])
        full = np.array([[x, y, 0.] for x in (-50., -25., 0., 25., 50.)
                         for y in (-30., 0., 30.)])
        rail = np.array([[x, 0., 0.] for x in np.linspace(-50., 50., 24)])
        tvec = np.array([0., 0., 500.])
        pixels = tracker.project(full, np.zeros(3), tvec, K, np.zeros(5))
        rail_pixels = tracker.project(rail, np.zeros(3), tvec, K, np.zeros(5))
        self.assertTrue(tracker.tracking_support_quality(
            full, pixels, vertices, np.zeros(3), tvec, K, np.zeros(5)))
        self.assertFalse(tracker.tracking_support_quality(
            rail, rail_pixels, vertices, np.zeros(3), tvec, K, np.zeros(5)))

    def test_box_ignores_roll_and_keeps_horizontal_view_level(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        base, _ = cv2.Rodrigues(np.array([0, np.pi / 2, 0], np.float64))
        center = np.array([0, 0, 800], np.float64)
        points = None
        for degrees in (0, 60, 120, 180):
            roll, _ = cv2.Rodrigues(np.array([0, 0, np.deg2rad(degrees)], np.float64))
            rvec, _ = cv2.Rodrigues(base @ roll)
            actual = tracker.roll_neutral_box_points(corners, rvec.reshape(3), center)
            if points is None:
                points = actual
            else:
                np.testing.assert_allclose(actual, points, atol=1e-8)
        np.testing.assert_allclose(points[4, 1:], points[0, 1:], atol=1e-8)

    def test_box_remains_finite_when_long_axis_points_up(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        rvec = np.array([np.pi / 2, 0, 0], np.float64)
        points = tracker.roll_neutral_box_points(
            corners, rvec, np.array([0, 0, 800], np.float64))
        self.assertTrue(np.isfinite(points).all())
        self.assertGreater(points[:, 2].min(), 0)

    def test_box_cross_section_is_continuous_through_upright_orientation(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        center = np.array([0., 0., 700.])
        previous = None
        for depth_component in np.linspace(.15, .25, 41):
            axis = np.array([.05, -np.sqrt(1. - .05**2 - depth_component**2), depth_component])
            cross = np.cross([0., 0., 1.], axis)
            rvec = cross / np.linalg.norm(cross) * np.arccos(axis[2])
            points = tracker.roll_neutral_box_points(corners, rvec, center)
            if previous is not None:
                self.assertLess(np.linalg.norm(points - previous, axis=1).max(), 1.)
            previous = points

    def test_box_fixed_transverse_edge_does_not_roll_as_long_axis_rotates(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        for angle in np.linspace(0, np.pi, 37):
            axis = np.array([np.cos(angle) * .95, np.sin(angle) * .95,
                             np.sqrt(1. - .95**2)])
            cross = np.cross([0., 0., 1.], axis)
            rvec = cross / np.linalg.norm(cross) * np.arccos(axis[2])
            points = tracker.roll_neutral_box_points(corners, rvec, np.array([0., 0., 700.]))
            edge = points[1] - points[0]
            self.assertAlmostEqual(edge[2], 0., places=8)
            self.assertAlmostEqual(float(edge @ axis), 0., places=8)

    def test_box_follows_yaw_foreshortening_and_pitch_slope(self):
        corners = tracker.load_model_box(tracker.DEFAULT_MODEL)
        K = np.array([[530, 0, 320], [0, 530, 240], [0, 0, 1]], np.float64)
        base, _ = cv2.Rodrigues(np.array([0, np.pi / 2, 0], np.float64))
        center = np.array([0, 0, 800], np.float64)

        def projected_axis(rotation):
            rvec, _ = cv2.Rodrigues(rotation)
            box = tracker.roll_neutral_box_points(corners, rvec.reshape(3), center)
            ends = np.stack((box[:4].mean(axis=0), box[4:].mean(axis=0)))
            uv, _ = cv2.projectPoints(ends, np.zeros(3), np.zeros(3), K, np.zeros(5))
            return uv.reshape(2, 2)[1] - uv.reshape(2, 2)[0], ends

        straight, _ = projected_axis(base)
        self.assertAlmostEqual(float(straight[1]), 0.0, places=6)
        for degrees in (-30, 30):
            yaw, _ = cv2.Rodrigues(np.array([0, np.deg2rad(degrees), 0]))
            axis, ends = projected_axis(yaw @ base)
            self.assertLess(np.linalg.norm(axis), np.linalg.norm(straight) * 0.95)
            self.assertGreater(abs(float(ends[1, 2] - ends[0, 2])), 70.0)
            self.assertAlmostEqual(float(axis[1]), 0.0, places=6)
            pitch, _ = cv2.Rodrigues(np.array([0, 0, np.deg2rad(degrees)]))
            tilted, _ = projected_axis(pitch @ base)
            self.assertAlmostEqual(
                float(np.rad2deg(np.arctan2(tilted[1], tilted[0]))),
                degrees, delta=0.5)

    def test_detector_box_must_overlap_projected_pose(self):
        vertices = np.array([[-50, -50, 0], [50, 50, 0]], np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
        pose = np.array([0, 0, 500], np.float64)
        self.assertTrue(tracker.detection_matches_pose(
            [260, 180, 380, 300], vertices, np.zeros(3), pose, K, np.zeros(5)))
        self.assertFalse(tracker.detection_matches_pose(
            [20, 20, 120, 120], vertices, np.zeros(3), pose, K, np.zeros(5)))

    def test_unassociated_detection_is_not_a_pose_veto(self):
        vertices = np.array([[-50, -50, 0], [50, 50, 0]], np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
        pose = np.array([0., 0., 500.])
        frame = np.zeros((480, 640, 3), np.uint8)
        # A nearby distractor touches the target but has no shared area.
        box = [370, 190, 510, 290]
        self.assertFalse(tracker.detection_associates_pose(
            box, vertices, np.zeros(3), pose, K, np.zeros(5), frame.shape))
        with patch.object(tracker, 'observe_long_axis') as observe:
            self.assertTrue(tracker.current_detection_agrees(
                frame, box, vertices, np.zeros(3), pose, K, np.zeros(5)))
        observe.assert_not_called()
        # Association does not loosen the existing overlapping-scale veto.
        self.assertFalse(tracker.current_detection_agrees(
            frame, [240, 160, 400, 320], vertices, np.zeros(3), pose, K, np.zeros(5)))
        with patch.object(tracker, 'axis_matches_pose', return_value=False):
            self.assertFalse(tracker.current_detection_agrees(
                frame, [270, 190, 370, 290], vertices, np.zeros(3), pose, K, np.zeros(5)))

    def test_detector_box_can_be_clipped_at_image_edge_when_projection_is_clipped(self):
        vertices = np.array([[-50, -50, 0], [50, 50, 0]], np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
        # The full projection is x=-50..50; only its right half is visible.
        self.assertTrue(tracker.detection_matches_pose(
            [0, 190, 50, 290], vertices, np.zeros(3),
            np.array([-320., 0., 500.]), K, np.zeros(5), image_shape=(480, 640, 3)))
        self.assertFalse(tracker.detection_matches_pose(
            [0, 190, 50, 290], vertices, np.zeros(3),
            np.array([-320., 0., 500.]), K, np.zeros(5)))
        self.assertFalse(tracker.detection_matches_pose(
            [0, 50, 50, 150], vertices, np.zeros(3),
            np.array([-320., 0., 500.]), K, np.zeros(5), image_shape=(480, 640, 3)))

    def test_crossing_horizontal_and_vertical_boxes_do_not_verify_a_pose(self):
        vertices = np.array([[-30, -65, 0], [30, 65, 0]], np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
        pose = np.array([0, 0, 500], np.float64)
        self.assertFalse(tracker.detection_matches_pose(
            [255, 210, 385, 270], vertices, np.zeros(3), pose, K, np.zeros(5)))

    def test_detector_rejects_wrong_scale_and_offset_even_when_boxes_overlap(self):
        vertices = np.array([[-50, -50, 0], [50, 50, 0]], np.float64)
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], np.float64)
        for box in ([240, 160, 400, 320], [305, 190, 405, 290]):
            self.assertFalse(tracker.detection_matches_pose(
                box, vertices, np.zeros(3), np.array([0, 0, 500.]), K, np.zeros(5)))

    def test_detector_rotated_boxes_map_back_to_camera_coordinates(self):
        original = np.array([100, 200, 250, 280], np.float32)
        rotated = ([200, 390, 280, 540], [390, 200, 540, 280],
                   [200, 100, 280, 250])
        for turn, box in enumerate(rotated, 1):
            np.testing.assert_allclose(
                tracker.restore_rotated_box(box, (480, 640, 3), turn), original)

    def test_long_axis_change_ignores_endpoint_symmetry(self):
        base = np.array([0, np.pi / 2, 0], np.float64)
        turn, _ = cv2.Rodrigues(np.array([0, np.deg2rad(20), 0]))
        base_rotation, _ = cv2.Rodrigues(base)
        moved, _ = cv2.Rodrigues(turn @ base_rotation)
        self.assertAlmostEqual(
            tracker.axis_line_change_deg(base, moved.reshape(3)), 20, delta=.01)
        half_turn, _ = cv2.Rodrigues(np.array([0, np.pi, 0]))
        flipped, _ = cv2.Rodrigues(half_turn @ base_rotation)
        self.assertAlmostEqual(
            tracker.axis_line_change_deg(base, flipped.reshape(3)), 0, delta=.01)

    def test_held_pose_is_visible_but_not_reported_as_measured(self):
        self.assertIn('TRACKING HOLD', tracker.VISIBLE_POSE_STATES)
        self.assertNotIn('TRACKING HOLD', tracker.DISPLAY_POSE_STATES)
        record = tracker.make_xyz_record(10, .5, 'TRACKING HOLD',
                                         np.zeros(3), np.array([0, 0, 700]))
        self.assertFalse(record['valid'])


class CadVideoTrackerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = read_image(tracker.DEFAULT_STATIC_REFERENCE_IMAGE)
        cls.live_view = read_image(
            tracker.DATA_ROOT / 'samples/energy_unit/akaze_scale_regression.png')
        cls.other_view = read_image(
            tracker.DATA_ROOT / 'samples/energy_unit/frame_20261029_153706_573194.png')
        if cls.reference is None or cls.live_view is None or cls.other_view is None:
            raise RuntimeError('Offline tracker fixtures could not be loaded')

    def associated_box(self):
        K, dist, _ = tracker.load_calibration(tracker.DEFAULT_INTRINSICS)
        vertices, _ = tracker.load_edges(tracker.DEFAULT_MODEL, 700)
        rvec, tvec = tracker.load_pose(tracker.DEFAULT_POSE)
        projected = tracker.project(vertices, rvec, tvec, K, dist)
        return np.r_[projected.min(axis=0), projected.max(axis=0)]

    def make_args(self, output_dir, frame_count):
        return SimpleNamespace(
            camera=0,
            camera_name='',
            camera_fps=0.0,
            camera_fourcc='',
            pose=tracker.DEFAULT_POSE,
            model=tracker.DEFAULT_MODEL,
            intrinsics=tracker.DEFAULT_INTRINSICS,
            output=output_dir / 'pose.json',
            log=output_dir / 'track.csv',
            summary=output_dir / 'summary.json',
            no_log=False,
            max_frames=frame_count,
            max_edges=700,
            edge_samples=3,
            max_points=100,
            refresh=0,
            reseed_below=12,
            draw_interval=2,
            roi_margin=40,
            seed_radius=8.0,
            flow_error=20.0,
            fb_error=1.5,
            pnp_error=5.0,
            pnp_iterations=60,
            max_rms=3.5,
            min_inliers=12,
            min_inlier_ratio=.30,
            max_rotation_step=12.0,
            max_translation_step=15.0,
            min_depth=80.0,
            max_depth=1200.0,
            static_correction=True,
            static_reference_image=tracker.DEFAULT_STATIC_REFERENCE_IMAGE,
            static_reference_pose=tracker.DEFAULT_STATIC_REFERENCE_POSE,
            static_min_frames=20,
            static_check_interval=15,
            static_flow_median=.35,
            static_flow_p90=.8,
            static_search_margin=80,
            static_gain=.25,
            static_max_center_step=5.0,
            static_max_depth_step=5.0,
            static_max_angle_step=1.0,
        )

    def run_sequence(self, frame_count, frame_factory, capture_xyz=False):
        temp_root = tracker.DATA_ROOT / 'tmp'
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp_dir:
            output_dir = Path(temp_dir)
            args = self.make_args(output_dir, frame_count)
            if capture_xyz:
                args.xyz_jsonl = output_dir / 'center_xyz.jsonl'
            capture = FakeCapture(frame_count, frame_factory)

            def wait_key(_delay):
                time.sleep(.004)
                return 255

            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'waitKey', side_effect=wait_key), \
                    patch.object(cv2, 'getWindowProperty', return_value=1.0), \
                    patch.object(cv2, 'destroyAllWindows'):
                tracker.run(args)

            with args.log.open(newline='', encoding='utf-8') as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertTrue(rows)
                self.assertTrue(all(None not in row and all(value is not None for value in row.values())
                                    for row in rows))
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            if capture_xyz:
                records = [json.loads(line) for line in
                           args.xyz_jsonl.read_text(encoding='utf-8').splitlines()]
                return rows, summary, records
            return rows, summary

    def test_three_d_loss_allows_one_transient_failure_but_not_persistent_loss(self):
        # A single flow/support failure can be caused by hand jitter or blur;
        # it may use the last metric pose for a bounded reseed. Persistent
        # failure must still become a hard loss and cannot publish a seeded
        # background texture as a valid pose.
        for failure in ('flow', 'support'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(
                    dir=tracker.DATA_ROOT / 'tmp') as folder:
                args = self.make_args(Path(folder), 12)
                args.three_d_only_after_init = True
                args.static_correction = False
                args.xyz_jsonl = Path(folder) / 'xyz.jsonl'
                capture = FakeCapture(args.max_frames, lambda _: self.reference.copy())
                original_flow = cv2.calcOpticalFlowPyrLK
                original_support = tracker.tracking_support_quality

                def flow(*a, **kw):
                    if failure == 'flow' and capture.index >= 4:
                        return None, None, None
                    return original_flow(*a, **kw)

                def support(*a, **kw):
                    if failure == 'support' and capture.index >= 4:
                        return False
                    return original_support(*a, **kw)

                with patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=flow), \
                        patch.object(tracker, 'tracking_support_quality', side_effect=support), \
                        patch.object(cv2, 'imshow'), \
                        patch.object(cv2, 'waitKey', return_value=255), \
                        patch.object(cv2, 'getWindowProperty', return_value=1.), \
                        patch.object(cv2, 'destroyAllWindows'):
                    self.assertEqual(tracker.run(args, capture=capture), 0)
                records = [json.loads(line) for line in args.xyz_jsonl.read_text().splitlines()]
                self.assertTrue(any(r['valid'] for r in records[:3]))
                self.assertFalse(any(r['valid'] for r in records[7:]))
                rows = list(csv.DictReader(args.log.open()))
                self.assertTrue(any(r['tracking_status'] == 'TARGET LOST' for r in rows[7:]))

    def test_rgb_reference_recovers_a_sideways_target_in_the_same_window(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            output_dir = Path(folder)
            pose_path = output_dir / 'pose.json'
            pose_data = json.loads(tracker.DEFAULT_POSE.read_text(encoding='utf-8'))
            pose_data['rgb_image'] = str((output_dir / 'reference.png').resolve())
            pose_path.write_text(json.dumps(pose_data), encoding='utf-8')
            cv2.imencode('.png', self.reference)[1].tofile(pose_data['rgb_image'])
            args = self.make_args(output_dir, 52)
            args.pose = pose_path
            args.rgb_pose_recovery = True
            args.fast_pose_recovery = True
            args.three_d_only_after_init = True
            args.reacquire_on_loss = True
            args.reacquire_lost_frames = 100
            args.static_correction = False
            K, dist, _ = tracker.load_calibration(tracker.DEFAULT_INTRINSICS)
            _rvec, tvec = tracker.load_pose(pose_path)
            center = tracker.project(np.zeros((1, 3)), _rvec, tvec, K, dist)[0]
            vertices, _, _ = tracker.make_edge_points(tracker.DEFAULT_MODEL, 700, 3)
            mask = tracker.cv2.dilate(
                tracker.make_search_mask(self.reference.shape, vertices, _rvec, tvec,
                                         K, dist, 0), np.ones((3, 3), np.uint8))
            background = self.reference.copy()
            background[mask > 0] = cv2.inpaint(
                self.reference, mask, 5, cv2.INPAINT_TELEA)[mask > 0]
            matrix = cv2.getRotationMatrix2D(tuple(center), 90, 1.0)
            moved = cv2.warpAffine(self.reference, matrix, self.reference.shape[1::-1])
            moved_mask = cv2.warpAffine(mask, matrix, self.reference.shape[1::-1])
            sideways = background.copy()
            sideways[moved_mask > 0] = moved[moved_mask > 0]

            def frame(index):
                return self.reference if index < 12 else sideways

            capture = FakeCapture(args.max_frames, frame)
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                    patch.object(cv2, 'getWindowProperty', return_value=1.0), \
                    patch.object(cv2, 'destroyAllWindows'):
                result = tracker.run(args, capture=capture, detector=None)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, 0)
            self.assertGreater(summary['rgb_recovery']['accepted'], 0)
            self.assertIn('POSE CORRECTED', summary['status_counts'])
            self.assertEqual(summary['reacquire_requested'], False)
            expected = cv2.transform(tracker.project(
                np.array([[0, 0, 0], [0, 0, 75], [0, 0, -75]], np.float64),
                _rvec, tvec, K, dist)[None].astype(np.float32), matrix)[0]
            final = summary['final_pose']
            actual = tracker.project(
                np.array([[0, 0, 0], [0, 0, 75], [0, 0, -75]], np.float64),
                np.asarray(final['rvec']), np.asarray(final['tvec_mm']), K, dist)
            from core.rgb_pose_recovery import axis_angle_difference
            from core.reference_pose_corrector import axis_angle
            self.assertLess(abs(axis_angle_difference(axis_angle(actual), axis_angle(expected))), 4)
            self.assertLess(np.linalg.norm(actual[0] - expected[0]), 5)

    def test_three_d_mode_uses_detector_to_verify_without_depth_reentry(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 24)
            args.headless = True
            args.three_d_only_after_init = True
            args.reacquire_on_loss = True
            args.detector_interval = 4
            args.static_correction = False
            capture = FakeCapture(args.max_frames, lambda _index: self.reference)
            detector = object()
            with patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(tracker, 'check_target', return_value=(None, None, None)) as check:
                self.assertEqual(tracker.run(args, capture=capture, detector=detector), 0)
            self.assertTrue(check.called)
            self.assertTrue(all(call.args[1] is detector for call in check.call_args_list))
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertFalse(summary['reacquire_requested'])
            self.assertEqual(summary['frames'], args.max_frames)

    def test_xyz_jsonl_records_valid_and_invalid_frames(self):
        with patch('builtins.print') as output:
            rows, summary, records = self.run_sequence(
                35, lambda _index: self.reference, capture_xyz=True)
        self.assertEqual(len(records), len(rows))
        self.assertEqual(records[0]['frame_index'], 0)
        self.assertIsNone(records[0]['center_xyz_mm'])
        self.assertTrue(any(record['valid'] for record in records))
        self.assertEqual(sum('RGB camera XYZ' in str(call.args[0])
                             for call in output.call_args_list if call.args),
                         sum(record['valid'] and record['frame_index'] % 30 == 0
                             for record in records))
        for row, record in zip(rows, records):
            self.assertEqual(record['status'], row['status'])
            if record['valid']:
                np.testing.assert_allclose(record['center_xyz_mm'],
                                           [float(row[key]) for key in
                                            ('tx_mm', 'ty_mm', 'tz_mm')], atol=1e-4)
            else:
                self.assertIsNone(record['center_xyz_mm'])
        self.assertEqual(summary['xyz_jsonl'], str(
            (Path(summary['log']).parent / 'center_xyz.jsonl').resolve()))

    def test_maintenance_reseeding_keeps_tracking_and_box_continuous(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 45)
            args.headless = True
            args.three_d_only_after_init = True
            args.static_correction = False
            args.refresh = 4
            capture = FakeCapture(args.max_frames, lambda _index: self.reference)
            self.assertEqual(tracker.run(args, capture=capture), 0)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertGreater(summary['timing_ms']['seed_calls'], 5)
            self.assertEqual(summary['status_counts'].get('SEEDED'), 1)
            self.assertEqual(summary['display']['box_visibility_changes'], 1)
            with args.log.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row['status'] == 'TRACKING' for row in rows[1:]))
            self.assertTrue(all(row['box_visible'] == 'True' for row in rows[1:]))

    def test_reference_library_can_become_ready_during_tracking(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 12)
            args.headless = True
            args.static_correction = False
            search = SimpleNamespace(estimate=lambda *_: None, clear=lambda: None)
            initializer = SimpleNamespace(new_search=lambda: search)
            reads = []

            def provider():
                reads.append(True)
                return initializer if len(reads) >= 4 else None

            args.rgb_initializer_provider = provider
            capture = FakeCapture(args.max_frames, lambda _index: self.reference)
            self.assertEqual(tracker.run(args, capture=capture), 0)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(len(reads), 4)
            self.assertIsNotNone(summary['rgb_relocalization'])
            self.assertEqual(summary['frames'], args.max_frames)

    def test_alternating_detector_outliers_do_not_toggle_tracking(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 60)
            args.headless = True
            args.static_correction = False
            args.detector_interval = 4
            args.reacquire_on_loss = True
            capture = FakeCapture(args.max_frames, lambda _index: self.reference)
            with patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(tracker, 'check_target', return_value=(self.associated_box(), None, None)), \
                    patch.object(tracker, 'detection_matches_pose', side_effect=[True, False] * 30):
                self.assertEqual(tracker.run(args, capture=capture, detector=object()), 0)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertNotIn('VERIFYING POSE', summary['status_counts'])
            self.assertFalse(summary['reacquire_requested'])
            self.assertEqual(summary['display']['box_visibility_changes'], 1)

    def test_unrelated_detector_hit_does_not_hide_or_reset_tracking(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 40)
            args.headless = True
            args.static_correction = False
            args.three_d_only_after_init = True
            args.detector_interval = 2
            capture = FakeCapture(args.max_frames, lambda _: self.reference)
            detector = SimpleNamespace(get_box=Mock(return_value=[1, 1, 25, 25]),
                                       stats={}, error=None)
            with patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor):
                self.assertEqual(tracker.run(args, capture=capture, detector=detector), 0)
            summary = json.loads(args.summary.read_text())
            self.assertNotIn('VERIFYING POSE', summary['status_counts'])
            self.assertNotIn('TARGET LOST', summary['status_counts'])
            self.assertEqual(summary['current_detection_conflicts'], 0)
            self.assertGreater(detector.get_box.call_count, args.max_frames)
            self.assertFalse(summary['detector_armed'])
            self.assertEqual(summary['display']['box_visibility_changes'], 1)

    def test_detector_conflict_requests_fresh_depth_pose(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 40)
            args.headless = True
            args.static_correction = False
            args.reacquire_on_loss = True
            args.reacquire_model = Path('unused.pt')
            args.reacquire_lost_frames = 40
            capture = FakeCapture(40, lambda _index: self.reference)
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch('ultralytics.YOLO'), \
                    patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(tracker, 'detect_target_box',
                                 return_value=self.associated_box()), \
                    patch.object(tracker, 'detection_matches_pose',
                                 side_effect=itertools.chain([True], itertools.repeat(False))):
                result = tracker.run(args)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, tracker.REACQUIRE_EXIT_CODE)
            self.assertTrue(summary['reacquire_requested'])
            self.assertFalse(summary['final_pose']['tracking_valid'])
            self.assertIn('TARGET LOST', summary['status_counts'])
            self.assertIn('VERIFYING POSE', summary['status_counts'])
            with args.log.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row['status'] == 'VERIFYING POSE' for row in rows))

    def test_detector_misses_do_not_interrupt_valid_tracking(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 40)
            args.headless = True
            args.static_correction = False
            args.reacquire_on_loss = True
            args.reacquire_lost_frames = 40
            capture = FakeCapture(40, lambda _index: self.reference)
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(tracker, 'detect_target_box',
                                 side_effect=[self.associated_box()] + [None] * 3), \
                    patch.object(tracker, 'detection_matches_pose', return_value=True):
                result = tracker.run(args, detector=object())
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, 0)
            self.assertTrue(summary['detector_armed'])
            self.assertNotIn('TARGET LOST', summary['status_counts'])

    def test_missed_or_unassociated_detection_does_not_erase_pose_conflict(self):
        class ImmediateExecutor:
            def __init__(self, **_kwargs):
                pass

            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future

            def shutdown(self, **_kwargs):
                pass

        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 35)
            args.headless = True
            args.static_correction = False
            args.reacquire_on_loss = True
            args.reacquire_lost_frames = 35
            args.detector_interval = 4
            capture = FakeCapture(35, lambda _index: self.reference)
            conflict = self.associated_box()
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(tracker, 'ThreadPoolExecutor', ImmediateExecutor), \
                    patch.object(tracker, 'detect_target_box',
                                 side_effect=[conflict, None, conflict, [1, 1, 25, 25], conflict]), \
                    patch.object(tracker, 'detection_matches_pose', return_value=False):
                self.assertEqual(tracker.run(args, detector=object()),
                                 tracker.REACQUIRE_EXIT_CODE)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertTrue(summary['reacquire_requested'])
            self.assertFalse(summary['final_pose']['tracking_valid'])

    def test_live_capture_keeps_existing_window(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 2)
            args.static_correction = False
            args.camera_fps = 30.0
            capture = FakeCapture(2, lambda _index: self.reference)
            with patch.object(cv2, 'VideoCapture') as open_camera, \
                    patch.object(cv2, 'imshow') as imshow, \
                    patch.object(cv2, 'waitKey', return_value=255), \
                    patch.object(cv2, 'getWindowProperty', return_value=1), \
                    patch.object(capture, 'set', wraps=capture.set) as set_property, \
                    patch.object(cv2, 'destroyAllWindows') as destroy:
                self.assertEqual(tracker.run(
                    args, capture=capture, window_name='Astra depth 6DoF',
                    keep_window=True), 0)
            open_camera.assert_not_called()
            destroy.assert_not_called()
            self.assertNotIn(cv2.CAP_PROP_FPS,
                             [call.args[0] for call in set_property.call_args_list])
            self.assertTrue(all(call.args[0] == 'Astra depth 6DoF'
                                for call in imshow.call_args_list))

    def test_three_d_mode_can_request_depth_fallback_after_rgb_loss(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 20)
            args.headless = True
            args.static_correction = False
            args.three_d_only_after_init = True
            args.reacquire_on_loss = True
            args.depth_fallback_on_loss = True
            args.reacquire_lost_frames = 40
            args.max_axis_change_from_depth_deg = 18.0
            args.max_rotation_step = 30.0
            capture = FakeCapture(20, lambda _index: self.reference)
            with patch.object(tracker, 'axis_line_change_deg', return_value=20.0):
                result = tracker.run(args, capture=capture)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, tracker.REACQUIRE_EXIT_CODE)
            self.assertTrue(summary['reacquire_requested'])
            self.assertIn('REACQUIRE DEPTH', summary['status_counts'])

    def test_large_axis_change_requests_fresh_depth_pose(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 40)
            args.headless = True
            args.static_correction = False
            args.reacquire_on_loss = True
            args.reacquire_model = None
            args.reacquire_lost_frames = 40
            args.max_axis_change_from_depth_deg = 18.0
            args.max_rotation_step = 30.0  # Isolate the explicit depth-reacquire gate.
            capture = FakeCapture(40, lambda _index: self.reference)
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(tracker, 'axis_line_change_deg', return_value=20.0):
                result = tracker.run(args)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, tracker.REACQUIRE_EXIT_CODE)
            self.assertIn('REACQUIRE DEPTH', summary['status_counts'])

    def test_reacquisition_can_return_the_open_rgb_capture(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as folder:
            args = self.make_args(Path(folder), 20)
            args.headless = True
            args.static_correction = False
            args.reacquire_on_loss = True
            args.max_axis_change_from_depth_deg = 18.
            capture = FakeCapture(20, lambda _index: self.reference)
            with patch.object(tracker, 'axis_line_change_deg', return_value=20.), \
                    patch.object(capture, 'release') as release:
                result = tracker.run(args, capture=capture, keep_capture_on_reacquire=True)
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(result, tracker.REACQUIRE_EXIT_CODE)
            release.assert_not_called()
            self.assertFalse(summary['final_pose']['tracking_valid'])

    def test_static_reference_locks_without_false_motion(self):
        rows, summary = self.run_sequence(150, lambda _index: self.reference)
        self.assertGreater(summary['status_counts'].get('STATIC LOCKED', 0), 30)
        self.assertNotIn('MOTION RESEED', summary['status_counts'])
        locked_motion = [float(row['static_lock_motion_px']) for row in rows
                         if row['status'] == 'STATIC LOCKED']
        self.assertLess(max(locked_motion), tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX)

    def test_live_scale_changed_view_corrects_and_locks(self):
        rows, summary = self.run_sequence(180, lambda _index: self.live_view)
        self.assertGreater(summary['static_correction']['accepted'], 0)
        self.assertGreater(summary['status_counts'].get('STATIC CORRECTED', 0), 0)
        self.assertGreater(summary['status_counts'].get('STATIC LOCKED', 0), 0)
        self.assertGreater(float(rows[-1]['tz_mm']), 230.0)

    def test_slow_rotation_unlocks_static_pose(self):
        height, width = self.reference.shape[:2]

        def frame(index):
            angle = max(index - 104, 0) * .1
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            return cv2.warpAffine(self.reference, matrix, (width, height),
                                  borderMode=cv2.BORDER_REFLECT)

        rows, summary = self.run_sequence(170, frame)
        events = [row for row in rows if row['status'] == 'MOTION RESEED']
        self.assertGreater(summary['status_counts'].get('MOTION RESEED', 0), 0)
        self.assertTrue(all(int(row['frame_index']) >= 105 for row in events))
        self.assertTrue(any(float(row['static_lock_motion_px']) >
                            tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX for row in events))

    def test_slow_translation_unlocks_static_pose(self):
        height, width = self.reference.shape[:2]

        def frame(index):
            offset = max(index - 104, 0) * .15
            matrix = np.array([[1.0, 0.0, offset], [0.0, 1.0, 0.0]], np.float32)
            return cv2.warpAffine(self.reference, matrix, (width, height),
                                  borderMode=cv2.BORDER_REFLECT)

        rows, summary = self.run_sequence(170, frame)
        events = [row for row in rows if row['status'] == 'MOTION RESEED']
        self.assertGreater(summary['status_counts'].get('MOTION RESEED', 0), 0)
        self.assertTrue(all(int(row['frame_index']) >= 105 for row in events))
        self.assertTrue(any(float(row['static_lock_shift_px']) >
                            tracker.STATIC_MAX_LOCK_DISPLACEMENT_PX for row in events))

    def test_removed_object_reports_static_lost(self):
        blank = np.zeros_like(self.reference)
        rows, summary = self.run_sequence(
            140, lambda index: self.reference if index < 105 else blank)
        self.assertGreater(summary['status_counts'].get('STATIC LOST', 0), 0)
        events = [row for row in rows if row['status'] == 'STATIC LOST']
        self.assertTrue(all(int(row['frame_index']) >= 105 for row in events))

    def test_unconfirmed_view_never_static_locks(self):
        _rows, summary = self.run_sequence(120, lambda _index: self.other_view)
        self.assertNotIn('STATIC CORRECTED', summary['status_counts'])
        self.assertNotIn('STATIC LOCKED', summary['status_counts'])
        self.assertGreater(summary['static_correction']['checks'], 0)
        self.assertGreater(summary['static_correction']['rejected'], 0)

    def test_handoff_pose_can_initialize_tracker(self):
        path = tracker.DATA_ROOT / 'tracking/depth_global_current/rgb_pose_overlay.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        rvec, tvec = tracker.load_pose(path)
        np.testing.assert_allclose(rvec, data['rvec_rgb_model_rad'])
        np.testing.assert_allclose(tvec, data['tvec_rgb_model_mm'])

    def test_rejected_handoff_pose_cannot_initialize_tracker(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as temp_dir:
            path = Path(temp_dir) / 'rejected_pose.json'
            path.write_text(json.dumps({
                'rvec_rgb_model_rad': [0, 0, 0],
                'tvec_rgb_model_mm': [0, 0, 800],
                'depth_pose_quality': {'accepted': False},
            }), encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'not accepted'):
                tracker.load_pose(path)

    def test_recorded_timestamps_match_capture_metadata(self):
        video = tracker.DATA_ROOT / 'samples/integration_sessions/session_01/motion/clip_0001/rgb.avi'
        metadata = json.loads(video.with_name('metadata.json').read_text(encoding='utf-8'))
        timestamps = tracker.load_video_timestamps(video)
        self.assertEqual(len(timestamps), metadata['frame_count'])
        self.assertAlmostEqual(timestamps[-1], metadata['frame_timestamps_s'][-1])

    def test_paired_anchors_bound_depth_drift(self):
        root = tracker.DATA_ROOT / 'samples/integration_sessions'
        lateral = tracker.load_session_depth_bounds(
            root / 'session_01/motion/clip_0001/rgb.avi', 100.0)
        approach = tracker.load_session_depth_bounds(
            root / 'session_02/motion/clip_0001/rgb.avi', 100.0)
        self.assertEqual((lateral['min_mm'], lateral['max_mm']), (700.0, 900.0))
        self.assertEqual((approach['min_mm'], approach['max_mm']), (600.0, 1000.0))
        self.assertEqual(lateral['end_nominal_mm'], 800.0)

    def test_projected_coverage_rejects_near_exit_pose(self):
        K, dist, size = tracker.load_calibration(tracker.DEFAULT_INTRINSICS)
        vertices, _, _ = tracker.make_edge_points(tracker.DEFAULT_MODEL, 700, 3)
        pose_path = (tracker.DATA_ROOT / 'tracking/evaluations/captured_20261031/samples/'
                     'integration/session_03/start/rgb_tracker_pose.json')
        rvec, tvec = tracker.load_pose(pose_path)
        inside = tracker.projected_coverage(vertices, rvec, tvec, K, dist,
                                            (size[1], size[0], 3))
        near_exit = tracker.projected_coverage(
            vertices, rvec, np.array([-390.0, -20.0, 700.0]), K, dist,
            (size[1], size[0], 3))
        self.assertGreater(inside, .99)
        self.assertLess(near_exit, .95)

    def run_recorded_session(self, session_id):
        evaluation = tracker.DATA_ROOT / 'tracking/evaluations/captured_20261031'
        video = (tracker.DATA_ROOT / 'samples/integration_sessions' / session_id /
                 'motion/clip_0001/rgb.avi')
        start_pose = (evaluation / 'samples/integration' / session_id /
                      'start/rgb_tracker_pose.json')
        end_pose = (evaluation / 'samples/integration' / session_id /
                    'end/rgb_tracker_pose.json')
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as temp_dir:
            args = self.make_args(Path(temp_dir), 0)
            args.video = video
            args.pose = start_pose
            args.headless = True
            args.static_correction = False
            tracker.run(args)
            with args.log.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
        _, end_tvec = tracker.load_pose(end_pose)
        return rows, summary, end_tvec

    def test_recorded_exit_and_return_relocalizes(self):
        rows, summary, end_tvec = self.run_recorded_session('session_03')
        out_of_view = [row for row in rows
                       if 9.0 <= float(row['source_elapsed_s']) <= 18.0]
        self.assertTrue(out_of_view)
        self.assertTrue(all(row['status'] == 'OUT OF VIEW' for row in out_of_view))
        self.assertGreater(summary['session_appearance']['relocalizations'], 0)
        self.assertLess(np.linalg.norm(np.asarray(summary['final_pose']['tvec_mm']) -
                                       end_tvec), 50.0)

    def test_recorded_weak_consistent_match_relocalizes(self):
        _rows, summary, end_tvec = self.run_recorded_session('session_05')
        self.assertGreater(summary['session_appearance']['relocalizations'], 0)
        self.assertLess(np.linalg.norm(np.asarray(summary['final_pose']['tvec_mm']) -
                                       end_tvec), 50.0)

    def test_headless_video_stops_at_eof_and_writes_source_time(self):
        with tempfile.TemporaryDirectory(dir=tracker.DATA_ROOT / 'tmp') as temp_dir:
            output_dir = Path(temp_dir)
            video = output_dir / 'rgb.avi'
            video.touch()
            args = self.make_args(output_dir, 0)
            args.video = video
            args.headless = True
            args.static_correction = False
            capture = FakeCapture(3, lambda _index: self.reference)
            with patch.object(cv2, 'VideoCapture', return_value=capture), \
                    patch.object(cv2, 'imshow', side_effect=AssertionError('unexpected window')), \
                    patch.object(cv2, 'waitKey', side_effect=AssertionError('unexpected key poll')):
                tracker.run(args)
            with args.log.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            summary = json.loads(args.summary.read_text(encoding='utf-8'))
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary['frames'], 3)
            self.assertEqual(summary['settings']['timestamp_source'], 'video')
            self.assertTrue(summary['video_reached_eof'])
            self.assertIsNone(summary['end_nominal_depth_check_passed'])
            self.assertEqual([row['source_elapsed_s'] for row in rows],
                             ['0.05', '0.1', '0.15'])
            self.assertIn('tracking_valid', summary['final_pose'])


if __name__ == '__main__':
    unittest.main()
