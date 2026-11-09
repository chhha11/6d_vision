"""Multi-instance detection, identity association, pose isolation and CLI coverage."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.background_detector import BackgroundDetector
from core import live_depth_6dof as live
from core.multi_target_detection import detect_target_boxes
from core.multi_target_live import (IndependentPoses, TargetSet, run_multi_target,
                                    target_record)


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def result(boxes, scores):
    return [SimpleNamespace(boxes=SimpleNamespace(
        xyxy=[Tensor(box) for box in boxes], conf=scores))]


class DetectionTest(unittest.TestCase):
    def test_collects_all_objects_and_checks_rotation_even_after_first_hit(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        calls = []
        def predict(source, **kwargs):
            calls.append(source.shape)
            if len(calls) == 1:
                return result([[30, 40, 110, 140], [200, 60, 280, 160]], [.9, .8])
            if len(calls) > 2:
                return result([], [])
            # Turn 1 maps these to the first object and a third object.
            return result([[40, 530, 140, 610], [220, 210, 320, 290]], [.7, .85])
        boxes = detect_target_boxes(SimpleNamespace(predict=predict), frame, 1)
        np.testing.assert_array_equal(boxes,
            [[30, 40, 110, 140], [350, 220, 430, 320], [200, 60, 280, 160]])
        self.assertEqual(calls[:2], [(480, 640, 3), (640, 480, 3)])
        self.assertEqual(len(calls), 8)

    def test_diagonal_target_is_detected_and_restored_without_duplicate_boxes(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        corners = cv2.boxPoints(((160., 120.), (40., 100.), 45.))
        cv2.fillConvexPoly(frame, np.rint(corners).astype(np.int32), (255, 255, 255))
        def predict(source, **kwargs):
            mask = (source[:, :, 0] > 240).astype(np.uint8)
            x, y, width, height = cv2.boundingRect(cv2.findNonZero(mask))
            # Simulate the existing model's preference for an upright unit.
            return (result([[x, y, x+width, y+height]], [.8])
                    if height > width*1.8 else result([], []))
        boxes = detect_target_boxes(SimpleNamespace(predict=predict), frame, 1)
        self.assertEqual(len(boxes), 1)
        expected = np.r_[corners.min(axis=0), corners.max(axis=0)]
        np.testing.assert_allclose(boxes[0], expected, atol=3.)

    def test_invalid_boxes_and_no_objects(self):
        detector = SimpleNamespace(predict=lambda *a, **k: result(
            [[0, 0, 5, 5], [10, 10, np.nan, 60], [-20, 20, 100, 100]], [.9]*3))
        self.assertEqual(detect_target_boxes(detector, np.zeros((480, 640, 3), np.uint8)), [])

    def test_contained_duplicate_is_removed_after_pose_track_is_established(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        compact = np.array([290., 204., 511., 319.], np.float32)
        broad = np.array([244., 107., 542., 396.], np.float32)
        tracker = TargetSet()
        targets = tracker.update(frame, [broad, compact], 0.)
        self.assertEqual(len(targets), 2)
        measured = targets[1]
        measured.pose = measured.last_good_pose = object()
        measured.pose_source = frame
        measured.pose_time = 0.
        with patch('core.multi_target_live.advance_box', side_effect=lambda a, b, box: box):
            visible = tracker.update(frame, [broad, compact], .1)
        self.assertEqual([target.id for target in visible], [measured.id])
        np.testing.assert_array_equal(visible[0].box, compact)

    def test_background_advances_each_box_and_drops_only_failed_object(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        boxes = [np.array([10, 10, 40, 50]), np.array([80, 20, 120, 70])]
        detector = BackgroundDetector(lambda: object(), lambda *_: boxes, multiple=True)
        try:
            detector.loading.result(timeout=2)
            self.assertEqual(detector.get_boxes(frame), [])
            detector.pending.result(timeout=2)
            with patch('core.background_detector.advance_box', side_effect=[boxes[0]+1, None]):
                output = detector.get_boxes(frame)
            np.testing.assert_array_equal(output, [boxes[0]+1])
            detector.pending.result(timeout=2)
            with patch('core.background_detector.time.monotonic', return_value=time.monotonic()+10):
                self.assertEqual(detector.get_boxes(frame), [])
        finally:
            detector.close()


class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((240, 320, 3), np.uint8)
        self.boxes = [[20, 30, 80, 130], [150, 50, 210, 150]]
        self.tracker = TargetSet()

    def test_detection_order_changes_do_not_swap_ids(self):
        initial = self.tracker.update(self.frame, self.boxes, 0.)
        self.assertEqual([t.id for t in initial], [1, 2])
        found = self.tracker.update(self.frame, self.boxes[::-1], .1)
        self.assertEqual([t.id for t in found], [1, 2])
        np.testing.assert_array_equal(found[0].box, self.boxes[0])
        np.testing.assert_array_equal(found[1].box, self.boxes[1])

    def test_missing_object_hides_pose_and_short_reentry_keeps_id(self):
        first, second = self.tracker.update(self.frame, self.boxes, 0.)
        second.pose = object()
        second.search = object()
        visible = self.tracker.update(self.frame, self.boxes[:1], .1)
        self.assertEqual([t.id for t in visible], [first.id])
        self.assertIsNone(second.pose)
        self.assertIsNone(second.search)
        visible = self.tracker.update(self.frame, self.boxes, .2)
        self.assertEqual([t.id for t in visible], [1, 2])
        self.assertEqual(target_record(second)['state'], 'detected')
        self.assertIsNone(target_record(second)['tvec_mm'])

    def test_expired_identity_is_never_reused(self):
        self.tracker.update(self.frame, self.boxes, 0.)
        self.assertEqual(self.tracker.update(self.frame, [], .5), [])
        visible = self.tracker.update(self.frame, self.boxes, 2.)
        self.assertEqual([t.id for t in visible], [3, 4])

    def test_six_targets_survive_reordering_and_independent_departure(self):
        boxes = [[15 + col*100, 20 + row*110, 80 + col*100, 110 + row*110]
                 for row in range(2) for col in range(3)]
        targets = self.tracker.update(self.frame, boxes, 0.)
        self.assertEqual([t.id for t in targets], list(range(1, 7)))
        shuffled = [boxes[i] for i in (5, 2, 0, 4, 1, 3)]
        targets = self.tracker.update(self.frame, shuffled, .1)
        self.assertEqual([t.id for t in targets], list(range(1, 7)))
        for target, box in zip(targets, boxes):
            np.testing.assert_array_equal(target.box, box)
        targets = self.tracker.update(self.frame, [boxes[i] for i in (5, 2, 0, 4)], .2)
        self.assertEqual([t.id for t in targets], [1, 3, 5, 6])

    def test_one_detection_cannot_match_two_targets(self):
        self.tracker.update(self.frame, [[20, 20, 100, 120], [70, 20, 150, 120]], 0.)
        found = self.tracker.update(self.frame, [[45, 20, 125, 120]], .1)
        self.assertEqual(len(found), 1)
        self.assertEqual(len(self.tracker.targets), 2)


class IndependentPoseTest(unittest.TestCase):
    def test_searches_have_separate_temporal_state_and_run_round_robin(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        boxes = [[15 + col*100, 20 + row*110, 80 + col*100, 110 + row*110]
                 for row in range(2) for col in range(3)]
        targets = TargetSet().update(frame, boxes, 0.)
        calls = []
        searches = []
        def new_search():
            search_id = len(searches)
            def estimate(_frame, box, validation_box):
                calls.append((search_id, box.copy(), validation_box.copy()))
                return None
            search = SimpleNamespace(estimate=estimate)
            searches.append(search)
            return search
        initializer = SimpleNamespace(new_search=new_search)
        service = IndependentPoses(lambda: initializer)
        try:
            service.loading.result(timeout=2)
            for _ in range(7):
                service.update(targets, frame)
                service.pending.result(timeout=2)
            self.assertEqual([call[0] for call in calls], [0, 1, 2, 3, 4, 5, 0])
            self.assertEqual(len({id(t.search) for t in targets}), 6)
            np.testing.assert_array_equal(calls[1][1], targets[1].box)
            np.testing.assert_array_equal(calls[1][2], targets[1].box)
        finally:
            service.close()

    def test_verified_poses_belong_to_their_own_targets(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        targets = TargetSet().update(frame, [[20, 30, 80, 130], [150, 50, 210, 150]], 0.)
        poses = [SimpleNamespace(rvec=np.zeros(3), tvec=np.array([x, 0., 700.]))
                 for x in (-100., 100.)]
        remaining = iter(poses)
        def new_search():
            pose = next(remaining)
            return SimpleNamespace(estimate=lambda *a, **k: pose)
        initializer = SimpleNamespace(new_search=new_search, K=np.eye(3),
                                      dist=np.zeros(5), vertices=np.zeros((8, 3)))
        service = IndependentPoses(lambda: initializer)
        try:
            service.loading.result(timeout=2)
            with patch('core.multi_target_live.detection_matches_pose', return_value=True):
                service.update(targets, frame)
                service.pending.result(timeout=2)
                service.update(targets, frame)
                service.pending.result(timeout=2)
                service.update(targets, frame, schedule=False)
            self.assertIs(targets[0].pose, poses[0])
            self.assertIs(targets[1].pose, poses[1])
            self.assertEqual(target_record(targets[0])['tvec_mm'], [-100., 0., 700.])
            self.assertEqual(target_record(targets[1])['tvec_mm'], [100., 0., 700.])
        finally:
            service.close()

    def test_pose_from_before_a_detection_gap_is_discarded(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        tracker = TargetSet()
        targets = tracker.update(frame, [[20, 30, 80, 130]], 0.)
        service = IndependentPoses(lambda: SimpleNamespace(new_search=lambda:
            SimpleNamespace(estimate=lambda *a, **k: object())))
        try:
            service.loading.result(timeout=2)
            service.update(targets, frame)
            service.pending.result(timeout=2)
            tracker.update(frame, [], .1)
            targets = tracker.update(frame, [[20, 30, 80, 130]], .2)
            service.update(targets, frame, schedule=False)
            self.assertIsNone(targets[0].pose)
            self.assertIsNone(service.pending)
        finally:
            service.close()

    def test_uncalibrated_frame_size_cannot_produce_metric_pose(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        targets = TargetSet().update(frame, [[20, 30, 80, 130]], 0.)
        service = IndependentPoses(lambda: object(), image_size=(640, 480))
        try:
            targets[0].pose = object()
            service.update(targets, frame)
            self.assertIsNone(targets[0].pose)
            self.assertIsNone(service.pending)
            self.assertIn('calibrated', service.error)
        finally:
            service.close()

    def test_late_pose_for_removed_target_is_discarded(self):
        frame = np.zeros((240, 320, 3), np.uint8)
        targets = TargetSet().update(frame, [[20, 30, 80, 130]], 0.)
        service = IndependentPoses(lambda: SimpleNamespace(new_search=lambda:
            SimpleNamespace(estimate=lambda *a, **k: object())))
        try:
            service.loading.result(timeout=2)
            service.update(targets, frame)
            service.pending.result(timeout=2)
            service.update([], frame)
            self.assertIsNone(targets[0].pose)
            self.assertIsNone(service.pending)
        finally:
            service.close()


class MultiTargetRuntimeTest(unittest.TestCase):
    def test_headless_video_records_all_targets_and_releases_resources(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frames = iter([(True, frame)] * 3 + [(False, None)])
        closed = []
        cap = SimpleNamespace(read=lambda: next(frames), isOpened=lambda: True,
                              get=lambda _: 30., release=lambda: closed.append('camera'))
        detections = iter([[[20, 30, 80, 130], [150, 50, 210, 150]],
                           [[150, 50, 210, 150]], []])
        detector = SimpleNamespace(get_boxes=lambda _: next(detections),
                                   close=lambda: closed.append('detector'))
        with tempfile.TemporaryDirectory() as directory:
            args = live.parse_args(['--multi-target', '--video', 'test.avi', '--headless',
                                    '--no-rgb-first', '--output-dir', directory])
            with patch('core.multi_target_live.cv2.VideoCapture', return_value=cap), \
                    patch.object(live, 'prepare_live_detector', return_value=detector), \
                    patch('core.multi_target_live.cv2.imshow') as show:
                self.assertEqual(run_multi_target(args), 0)
                show.assert_not_called()
            rows = [json.loads(line) for line in
                    (Path(directory)/'multi_targets.jsonl').read_text().splitlines()]
            self.assertEqual([r['count'] for r in rows], [2, 1, 0])
            self.assertEqual(rows[1]['targets'][0]['id'], 2)
            self.assertIsNone(rows[0]['targets'][0]['tvec_mm'])
        self.assertEqual(closed, ['camera', 'detector'])

    def test_default_stays_single_target_and_invalid_combinations_fail(self):
        self.assertFalse(live.parse_args([]).multi_target)
        for flags in (['--headless'], ['--multi-target', '--offline-sample', 'sample'],
                      ['--multi-target', '--region-tracking']):
            with self.subTest(flags=flags), self.assertRaises(SystemExit), \
                    patch('sys.stderr'):
                live.parse_args(flags)
