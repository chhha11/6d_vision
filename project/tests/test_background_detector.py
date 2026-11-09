import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.background_detector import BackgroundDetector, advance_box
from core import live_depth_6dof as live
from core import cad_video_tracker as tracker


class BackgroundDetectorTest(unittest.TestCase):
    def test_loading_and_inference_never_wait_on_caller_and_do_not_queue(self):
        load_started, load_finish, infer_started, infer_finish = (threading.Event() for _ in range(4))
        frames = []
        def load():
            load_started.set()
            self.assertTrue(load_finish.wait(2))
            return object()
        def infer(_model, frame, turn):
            frames.append((int(frame[0, 0, 0]), turn))
            infer_started.set()
            self.assertTrue(infer_finish.wait(2))
            return np.array([10, 10, 60, 60], np.float32)
        detector = BackgroundDetector(load, infer)
        frame = np.zeros((80, 80, 3), np.uint8)
        try:
            self.assertTrue(load_started.wait(2))
            self.assertIsNone(detector.get_box(frame))
            self.assertEqual(detector.stats['submitted'], 0)
            load_finish.set()
            detector.loading.result(timeout=2)
            detector.get_box(frame)
            self.assertTrue(infer_started.wait(2))
            frame[:] = 1
            for _ in range(10):
                self.assertIsNone(detector.get_box(frame))
            self.assertEqual(detector.stats['submitted'], 1)
            infer_finish.set()
            detector.pending.result(timeout=2)
            frame[:] = 2
            detector.get_box(frame)
            detector.pending.result(timeout=2)
            self.assertEqual(frames, [(0, 1), (2, 3)])
        finally:
            load_finish.set()
            infer_finish.set()
            detector.close()

    def test_failed_optional_model_does_not_raise_in_3d_thread(self):
        def load():
            raise RuntimeError('fixture: missing weights')
        detector = BackgroundDetector(load, lambda *_: None)
        try:
            with self.assertRaises(RuntimeError):
                detector.loading.result(timeout=2)
            self.assertIsNone(detector.get_box(np.zeros((80, 80, 3), np.uint8)))
            self.assertIn('missing weights', detector.error)
        finally:
            detector.close()

    def test_expired_box_cannot_verify_a_current_pose(self):
        frame = np.zeros((80, 80, 3), np.uint8)
        detector = BackgroundDetector(lambda: object(), lambda *_: np.array([10,10,60,60]))
        try:
            detector.loading.result(timeout=2)
            with patch('core.background_detector.time.monotonic', return_value=1.):
                detector.get_box(frame)
                detector.pending.result(timeout=2)
            with patch('core.background_detector.time.monotonic', return_value=2.):
                self.assertIsNone(detector.get_box(frame))
            self.assertEqual(detector.stats['expired'], 1)
        finally:
            detector.close()

    def test_detected_box_advances_with_measured_pixels(self):
        rng = np.random.default_rng(16)
        frame = np.zeros((160, 200, 3), np.uint8)
        frame[35:120, 40:140] = rng.integers(30, 240, (85, 100, 3), dtype=np.uint8)
        moved = cv2.warpAffine(frame, np.array([[1.,0.,7.],[0.,1.,-4.]]), (200,160))
        result = advance_box(frame, moved, np.array([40,35,140,120]))
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [47,31,147,116], atol=1.)
        self.assertIsNone(advance_box(frame, np.zeros_like(frame), [40,35,140,120]))


class Independent3dSearchTest(unittest.TestCase):
    def test_global_startup_pose_is_possible_without_any_detector(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0.,0.,500.]))
        seen = []
        def estimate(_frame, box, validation_box=None):
            seen.append((box.copy(), validation_box))
            return pose
        search = SimpleNamespace(estimate=estimate, clear=lambda: None,
                                 K=np.eye(3), dist=np.zeros(5), vertices=np.zeros((8,3)))
        initializer = SimpleNamespace(new_search=lambda: search)
        hint = dict(initializer_provider=lambda: initializer)
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch('core.rgb_global_initializer.advance_pose', return_value=pose), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', side_effect=lambda _: (time.sleep(.002) or 255)), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_3d_target(object(), None, hint))
        np.testing.assert_array_equal(seen[0][0], [0.,0.,160.,120.])
        self.assertIsNone(seen[0][1])
        self.assertIs(hint['rgb_pose'], pose)
        self.assertIsNone(hint['box'])

    def test_moving_rgb_startup_advances_pose_without_waiting_for_stillness(self):
        from core.rgb_global_initializer import solve_correspondences
        from core.cad_video_tracker import project
        K = np.array([[530.,0,320.],[0,530.,240.],[0,0,1.]])
        dist = np.zeros(5)
        points = np.array([[x,y,((x+y)%30)] for x in (-60.,-30.,0.,30.,60.)
                           for y in (-40.,0.,40.)])
        rvec, tvec = np.array([.2,-.15,.1]), np.array([20.,-15.,600.])
        uv = project(points,rvec,tvec,K,dist)
        pose = solve_correspondences(points,uv,K,dist,points,None)
        self.assertIsNotNone(pose)
        source = np.random.default_rng(9).integers(0,255,(480,640,3),dtype=np.uint8)
        current = cv2.warpAffine(source,np.array([[1.,0.,3.],[0.,1.,2.]]),(640,480))
        search = SimpleNamespace(estimate=lambda *_a,**_k: pose, clear=lambda: None,
                                 K=K,dist=dist,vertices=points)
        initializer = SimpleNamespace(new_search=lambda: search)
        hint = dict(initializer_provider=lambda: initializer,settle_seconds=1000.)
        box = np.r_[uv.min(0),uv.max(0)].astype(np.float32)
        detector = SimpleNamespace(get_box=lambda *_a: box)
        count = 0
        def read(*_):
            nonlocal count, current
            count += 1
            current = cv2.warpAffine(source,np.array([[1.,0.,3.*(count-1)],
                                                      [0.,1.,2.*(count-1)]]),(640,480))
            return current
        deadline=time.monotonic()+5.
        def key(_):
            time.sleep(.002)
            return ord('q') if time.monotonic()>deadline else 255
        with patch.object(live,'read_rgb',side_effect=read), \
                patch.object(live,'target_motion_fraction',return_value=1.), \
                patch.object(cv2,'imshow'),patch.object(cv2,'waitKey',side_effect=key), \
                patch.object(cv2,'getWindowProperty',return_value=1):
            self.assertTrue(live.wait_for_3d_target(object(),detector,hint))
        actual=project(points,hint['rgb_pose'].rvec,hint['rgb_pose'].tvec,K,dist)
        self.assertLess(np.linalg.norm(actual-(uv+np.array([3.,2.])*(count-1)),axis=1).mean(),.3)
        np.testing.assert_array_equal(hint['rgb_frame'],current)
        self.assertGreater(np.linalg.norm(hint['rgb_pose'].tvec-pose.tvec),2.)

    def test_yolo_roi_gets_a_turn_after_unsuccessful_global_search(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        box = np.array([20.,30.,80.,100.], np.float32)
        seen = []
        pose = object()
        def estimate(_frame, region, validation_box=None):
            seen.append((region.copy(), validation_box))
            return pose if validation_box is not None else None
        search = SimpleNamespace(estimate=estimate, clear=lambda: None,
                                 K=np.eye(3), dist=np.zeros(5), vertices=np.zeros((8,3)))
        hint = dict(initializer_provider=lambda: SimpleNamespace(new_search=lambda: search),
                    rgb_fallback_timeout_s=100.)
        initializer = hint['initializer_provider']()
        hint['initializer_provider'] = lambda: initializer
        detector = SimpleNamespace(get_box=lambda *_: box)
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch('core.rgb_global_initializer.advance_pose', return_value=pose), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', side_effect=lambda _: (time.sleep(.002) or 255)), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_3d_target(object(), detector, hint))
        self.assertEqual(len(seen), 2)
        np.testing.assert_array_equal(seen[1][0], box)
        np.testing.assert_array_equal(seen[1][1], box)
        self.assertEqual(hint['search_stats'], dict(
            global_checks=1, yolo_roi_checks=1, completed=2, candidates=1,
            expired=0, handoff_rejected=0, depth_fallbacks=0))

    def test_hard_loss_alternates_yolo_region_with_detector_free_search(self):
        frame = np.zeros((120,160,3), np.uint8)
        box = np.array([20.,30.,80.,100.], np.float32)
        seen = []
        def estimate(_frame, region, validation_box=None):
            seen.append((region.copy(), validation_box))
            return None
        detector = SimpleNamespace(get_box=lambda *_a, **_k: box)
        search = SimpleNamespace(estimate=estimate)
        tracker.check_target(frame, detector, None, 1, pose_search=search,
                             global_search=True, prefer_roi=False)
        tracker.check_target(frame, detector, None, 3, pose_search=search,
                             global_search=True, prefer_roi=True)
        np.testing.assert_array_equal(seen[0][0], [0.,0.,160.,120.])
        self.assertIsNone(seen[0][1])
        np.testing.assert_array_equal(seen[1][0], box)
        np.testing.assert_array_equal(seen[1][1], box)


if __name__ == '__main__':
    unittest.main()
