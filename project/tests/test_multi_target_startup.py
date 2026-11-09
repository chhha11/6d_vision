"""Bounded progressive scan coverage and cold-search continuity."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core.multi_target_detection import detect_target_boxes, detect_live_target_boxes
from core.multi_target_live import TargetSet, compact_search_candidates, target_record
from tests.test_multi_target import result


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def test_progressive_sweep_keeps_original_and_covers_all_six_angles(self):
        angles, calls = [], []
        def rotate(frame, angle):
            angles.append(angle)
            return frame, np.array([[1., 0., 0.], [0., 1., 0.]])
        def predict(frame, **kwargs):
            calls.append(frame.shape)
            return result([[20, 30, 80, 130]], [.9]) if len(calls) % 4 == 1 else result([], [])
        with patch('core.multi_target_detection.diagonal_view', side_effect=rotate):
            for turn in (1, 3, 2):
                boxes = detect_live_target_boxes(SimpleNamespace(predict=predict), self.frame, turn)
                np.testing.assert_array_equal(boxes, [[20, 30, 80, 130]])
        self.assertEqual(len(calls), 12)
        self.assertEqual(sorted(angles), [-60., -45., -30., 30., 45., 60.])

    def test_six_units_return_on_first_original_pass(self):
        expected = [[20+c*180, 20+r*200, 100+c*180, 160+r*200]
                    for r in range(2) for c in range(3)]
        calls = []
        def predict(frame, **kwargs):
            calls.append(1)
            return result(expected, [.9]*6) if len(calls) == 1 else result([], [])
        boxes = detect_live_target_boxes(SimpleNamespace(predict=predict), self.frame, 1)
        np.testing.assert_array_equal(boxes, expected)
        self.assertEqual(len(calls), 4)

    def test_progressive_option_reaches_process_proxy_only_when_requested(self):
        from unittest.mock import Mock
        proxy = SimpleNamespace(get_boxes=Mock(return_value=[]))
        detect_live_target_boxes(proxy, self.frame, 3)
        proxy.get_boxes.assert_called_once_with(self.frame, fallback_rotation=3, progressive=True)
        proxy.get_boxes.reset_mock()
        detect_target_boxes(proxy, self.frame, 3)
        proxy.get_boxes.assert_called_once_with(self.frame, fallback_rotation=3)

    def test_search_survives_scan_gap_only_with_current_flow_and_bounded_age(self):
        tracks = TargetSet(search_grace_s=.75)
        target = tracks.update(self.frame, [[20, 30, 80, 130]], 0.)[0]
        search = target.search = object()
        with patch('core.multi_target_live.advance_box', side_effect=lambda a,b,c:c):
            found = tracks.update(self.frame, [], .3)
            self.assertEqual(found, [target])
            self.assertIs(target.search, search)
            self.assertFalse(target.detected)
            self.assertIsNone(target_record(target)['tvec_mm'])
            self.assertEqual(tracks.update(self.frame, [], .76), [])
            self.assertIsNone(target.search)

    def test_failed_flow_immediately_discards_uninitialized_search(self):
        tracks = TargetSet(search_grace_s=.75)
        target = tracks.update(self.frame, [[20, 30, 80, 130]], 0.)[0]
        target.search = object()
        with patch('core.multi_target_live.advance_box', return_value=None):
            self.assertEqual(tracks.update(self.frame, [], .1), [])
        self.assertIsNone(target.search)

    def test_containment_prioritizes_small_roi_without_merging_targets(self):
        targets = TargetSet().update(self.frame,
            [[10, 10, 500, 450], [30, 40, 110, 170], [200, 50, 280, 180]], 0.)
        self.assertEqual([t.id for t in compact_search_candidates(targets)], [2, 3])
        self.assertEqual(len(targets), 3)
        self.assertEqual(compact_search_candidates([targets[0]]), [targets[0]])

    def test_six_separate_targets_have_equal_search_priority(self):
        boxes = [[20+c*180, 20+r*200, 100+c*180, 160+r*200]
                 for r in range(2) for c in range(3)]
        targets = TargetSet().update(self.frame, boxes, 0.)
        self.assertEqual(compact_search_candidates(targets), targets)

    def test_enclosing_candidate_still_gets_a_search_slot(self):
        from core.multi_target_live import IndependentPoses
        from tests.test_multi_target_handheld import DeferredExecutor
        initializer = SimpleNamespace(new_search=lambda: SimpleNamespace(estimate=lambda *a, **k: None))
        service = IndependentPoses(lambda: initializer)
        service.loading.result(timeout=2)
        service.executor.shutdown(wait=True)
        service.executor = DeferredExecutor()
        self.addCleanup(service.close)
        targets = TargetSet().update(self.frame,
            [[10, 10, 500, 450], [30, 40, 110, 170]], 0.)
        chosen = []
        for _ in range(8):
            service.update(targets, self.frame)
            chosen.append(service.request[0])
            service.pending.set_result(None)
            service.update(targets, self.frame, schedule=False)
        self.assertEqual(chosen, [2, 2, 2, 1, 2, 2, 2, 1])

    def test_large_background_box_does_not_erase_tight_search_between_angles(self):
        tracks = TargetSet(search_grace_s=.75)
        target = tracks.update(self.frame, [[100, 100, 200, 250], [40, 40, 300, 350]], 0.)[0]
        search = target.search = object()
        with patch('core.multi_target_live.advance_box', side_effect=lambda a,b,c:c):
            found = tracks.update(self.frame, [[40, 40, 300, 350]], .2)
        self.assertIn(target, found)
        self.assertIs(target.search, search)
        self.assertFalse(target.detected)
        self.assertIsNone(target.pose)

    def test_broad_proposal_cannot_replace_a_single_compact_search(self):
        tracks = TargetSet(search_grace_s=.75)
        original = np.array([100, 100, 200, 250], np.float32)
        target = tracks.update(self.frame, [original], 0.)[0]
        search = target.search = object()
        with patch('core.multi_target_live.advance_box', side_effect=lambda a,b,c:c):
            tracks.update(self.frame, [[40, 40, 300, 350]], .2)
        np.testing.assert_array_equal(target.box, original)
        self.assertIs(target.search, search)
        self.assertTrue(target.visible)
