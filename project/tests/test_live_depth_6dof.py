import json
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from core import live_depth_6dof as live
from core.paths import DATA_ROOT


SAMPLE = DATA_ROOT / "samples" / "global_dataset" / "P001"


class LiveDepth6DoFTest(unittest.TestCase):
    def setUp(self):
        # UI is mocked in this suite; keep the fake window open unless a test
        # explicitly closes it. The real preview now handles window close.
        window = patch.object(cv2, 'getWindowProperty', return_value=1)
        window.start()
        self.addCleanup(window.stop)
        identity = patch.object(live.PinnedCamera, 'bind', return_value=None)
        identity.start()
        self.addCleanup(identity.stop)

    def test_live_overlaps_depth_fit_with_rgb_reopen(self):
        class FakeCamera:
            def release(self):
                pass

        frame = np.zeros((480, 640, 3), np.uint8)
        fit_started = threading.Event()
        fit_release = threading.Event()

        def fit(*_args):
            fit_started.set()
            self.assertTrue(fit_release.wait(2), "depth fit did not overlap RGB reopen")
            return {}

        def reopen(*_args):
            self.assertTrue(fit_started.wait(2), "depth fit was not started before RGB reopen")
            fit_release.set()
            return FakeCamera(), frame

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(
                camera_name='USB Camera', camera=0, output_dir=output,
                preview_frames=1, extrinsics=live.DEFAULT_EXTRINSICS,
                depth_warmup=0, depth_timeout_s=5,
                rgb_reopen_attempts=1, model=live.DEFAULT_MODEL,
                intrinsics=live.DEFAULT_INTRINSICS, settings=live.Settings(),
                box_display_y_offset_px=12.0)
            with patch.object(live, 'resolve_camera_index', return_value=(0, None)), \
                    patch.object(live, 'open_camera', side_effect=lambda *_: FakeCamera()), \
                    patch.object(live, 'reopen_rgb', side_effect=reopen), \
                    patch.object(live, 'read_rgb', return_value=frame), \
                    patch.object(live, 'run_depth_worker'), \
                    patch.object(live, 'fit_depth_capture', side_effect=fit), \
                    patch.object(live, 'initialize_capture',
                                 side_effect=lambda *a: a[2] / 'rgb_initial_pose.json'), \
                    patch.object(live, 'run_tracker', return_value=0), \
                    patch.object(live, 'run_with_preview', side_effect=lambda task, *_a: task()), \
                    patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'waitKey', return_value=255), \
                    patch.object(cv2, 'destroyWindow'), \
                    patch.object(cv2, 'destroyAllWindows'):
                self.assertEqual(live.run_live(args), 0)
            self.assertTrue(fit_started.is_set())

    def test_live_restarts_capture_after_tracker_requests_depth(self):
        class FakeCamera:
            def release(self):
                pass

        frame = np.zeros((480, 640, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(
                camera_name='USB Camera', camera=0, output_dir=output,
                preview_frames=1, extrinsics=live.DEFAULT_EXTRINSICS,
                depth_warmup=0, depth_timeout_s=5,
                rgb_reopen_attempts=1, model=live.DEFAULT_MODEL,
                intrinsics=live.DEFAULT_INTRINSICS, settings=live.Settings(),
                box_display_y_offset_px=12.0)
            with patch.object(live, 'resolve_camera_index', return_value=(0, None)), \
                    patch.object(live, 'open_camera', side_effect=lambda *_: FakeCamera()), \
                    patch.object(live, 'reopen_rgb',
                                 side_effect=lambda *_: (FakeCamera(), frame)), \
                    patch.object(live, 'read_rgb', return_value=frame), \
                    patch.object(live, 'run_depth_worker'), \
                    patch.object(live, 'fit_depth_capture', return_value={}), \
                    patch.object(live, 'initialize_capture',
                                 side_effect=lambda *a: a[2] / 'rgb_initial_pose.json'), \
                    patch.object(live, 'run_tracker', side_effect=[2, 0]) as run_tracker, \
                    patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'waitKey', return_value=255), \
                    patch.object(cv2, 'destroyWindow'), \
                    patch.object(cv2, 'destroyAllWindows'):
                self.assertEqual(live.run_live(args), 0)
            self.assertEqual(run_tracker.call_count, 2)
            self.assertTrue((output / 'attempt_001').is_dir())
            self.assertTrue((output / 'attempt_002').is_dir())
            self.assertEqual(run_tracker.call_args_list[1].args[2],
                             output / 'attempt_002' / 'tracking')

    def test_live_tracker_requests_depth_reacquisition(self):
        args = SimpleNamespace(model=live.DEFAULT_MODEL,
                               intrinsics=live.DEFAULT_INTRINSICS,
                               box_display_y_offset_px=12.0,
                               video=None, max_frames=0,
                               reacquire_model=live.DETECTOR_MODEL)
        command = live.tracker_command(
            args, Path('pose.json'), Path('tracking'), 0)
        self.assertIn('--reacquire-on-loss', command)
        self.assertIn('--depth-fallback-on-loss', command)
        self.assertEqual(command[command.index('--fb-error') + 1], '2.5')
        self.assertEqual(command[command.index('--max-rotation-step') + 1], '22')
        self.assertEqual(command[command.index('--max-translation-step') + 1], '35')
        self.assertEqual(command[command.index('--reacquire-model') + 1],
                         str(live.DETECTOR_MODEL))
        self.assertNotIn('--max-axis-change-from-depth-deg', command)

    def test_live_tracker_reuses_capture_detector_and_window(self):
        args = SimpleNamespace(model=live.DEFAULT_MODEL,
                               intrinsics=live.DEFAULT_INTRINSICS,
                               box_display_y_offset_px=12.0,
                               video=None, max_frames=0,
                               reacquire_model=live.DETECTOR_MODEL)
        capture, detector = object(), object()
        args.rgb_initializer = object()
        args.rgb_initializer_provider = lambda: args.rgb_initializer
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(live, 'run_rgb_tracker', return_value=2) as run, \
                    patch.object(live.subprocess, 'run') as subprocess_run:
                self.assertEqual(live.run_tracker(
                    args, Path('pose.json'), Path(folder), 0,
                    capture, detector), 2)
            subprocess_run.assert_not_called()
            self.assertIs(run.call_args.kwargs['capture'], capture)
            self.assertIs(run.call_args.kwargs['detector'], detector)
            self.assertIs(run.call_args.args[0].rgb_initializer, args.rgb_initializer)
            self.assertIs(run.call_args.args[0].rgb_initializer_provider,
                          args.rgb_initializer_provider)
            self.assertEqual(run.call_args.kwargs['window_name'], live.WINDOW)
            self.assertTrue(run.call_args.kwargs['keep_window'])

    def test_empty_initial_scene_retries_when_target_enters(self):
        class FakeCamera:
            def release(self):
                pass

        frame = np.zeros((480, 640, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            args = SimpleNamespace(
                camera_name='USB Camera', camera=0, output_dir=output,
                preview_frames=1, extrinsics=live.DEFAULT_EXTRINSICS,
                depth_warmup=0, depth_timeout_s=5,
                rgb_reopen_attempts=1, model=live.DEFAULT_MODEL,
                reacquire_model=live.DETECTOR_MODEL,
                intrinsics=live.DEFAULT_INTRINSICS, settings=live.Settings(),
                box_display_y_offset_px=12.0)
            pose = output / 'attempt_002' / 'rgb_initial_pose.json'
            with patch.object(live, 'resolve_camera_index', return_value=(0, None)), \
                    patch.object(live, 'open_camera', side_effect=lambda *_: FakeCamera()), \
                    patch.object(live, 'reopen_rgb',
                                 side_effect=lambda *_: (FakeCamera(), frame)), \
                    patch.object(live, 'read_rgb', return_value=frame), \
                    patch.object(live, 'run_depth_worker'), \
                    patch.object(live, 'fit_depth_capture', return_value={}), \
                    patch.object(live, 'initialize_capture', side_effect=[
                        RuntimeError('No cluster matches the energy-unit dimensions.'), pose]), \
                    patch('ultralytics.YOLO'), \
                    patch.object(live, 'wait_for_target', return_value=True) as search, \
                    patch.object(live, 'run_tracker', return_value=0) as run_tracker, \
                    patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'waitKey', return_value=255), \
                    patch.object(cv2, 'destroyWindow'), \
                    patch.object(cv2, 'destroyAllWindows'):
                self.assertEqual(live.run_live(args), 0)
            self.assertEqual(search.call_count, 2)
            self.assertEqual(run_tracker.call_args.args[1], pose)

    def test_rgb_search_ignores_unrelated_scene_changes(self):
        frame = np.zeros((480, 640, 3), np.uint8)

        class FakeDetector:
            def __init__(self):
                self.calls = 0

            def predict(self, *_args, **_kwargs):
                self.calls += 1
                boxes = (np.array([[576, 0, 639, 403]], np.float32)
                         if self.calls <= 2 else
                         np.array([[302, 214, 453, 297]], np.float32))
                return [SimpleNamespace(boxes=SimpleNamespace(
                    conf=torch.tensor([.54]), xyxy=torch.from_numpy(boxes)))]

        detector = FakeDetector()
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_target(object(), detector))
        self.assertEqual(detector.calls, 4)

    def test_search_accepts_late_entry_with_a_missed_inference(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([280, 130, 400, 355], np.float32)
        hint = {}
        with patch.object(live, 'detect_target_box',
                          side_effect=[None, None, box, None, box]) as detect, \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_target(object(), object(), hint))
        self.assertEqual(detect.call_count, 5)
        np.testing.assert_array_equal(hint['box'], box)

    def test_reacquire_waits_for_short_settle_period(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([280, 130, 400, 355], np.float32)
        hint = {'settle_seconds': .3}
        with patch.object(live, 'detect_target_box', return_value=box), \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(live.time, 'monotonic', side_effect=[0, .1, .1, .2, .4]), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_target(object(), object(), hint))

    def test_space_does_not_switch_to_depth_while_rgb_references_load(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([280, 130, 400, 355], np.float32)
        hint = {'initializer_provider': lambda: None,
                'references_loading': lambda: True,
                'references_ready': lambda: False}
        with patch.object(live, 'detect_target_box', return_value=box), \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', side_effect=[32, ord('q')]):
            self.assertFalse(live.wait_for_target(object(), object(), hint))
        self.assertNotIn('box', hint)

    def test_async_search_keeps_reading_frames_while_detector_runs(self):
        started = threading.Event()
        release = threading.Event()
        frame = np.zeros((480, 640, 3), np.uint8)
        reads = []

        def detect(*_args):
            started.set()
            release.wait(2)
            return None

        def read(*_args):
            reads.append(True)
            if len(reads) > 2:
                release.set()
            return frame

        def key(_delay):
            if len(reads) >= 3:
                return ord('q')
            started.wait(1)
            return 255

        with patch.object(live, 'detect_target_box', side_effect=detect), \
                patch.object(live, 'read_rgb', side_effect=read), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', side_effect=key):
            self.assertFalse(live.wait_for_target(
                object(), object(), {'async_detection': True}))
        self.assertGreaterEqual(len(reads), 3)

    def test_search_can_handoff_rgb_pose_while_target_is_moving(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0., 0., 700.]),
                               points=np.zeros((12, 3)), pixels=np.zeros((12, 2)),
                               reference='fixture', rms_px=1., inlier_ratio=.9,
                               elapsed_ms=2.)
        initializer = SimpleNamespace(estimate=lambda _frame, _box: pose)
        hint = {'initializer_provider': lambda: initializer}
        with patch.object(live, 'detect_target_box',
                          return_value=np.array([280, 130, 400, 355], np.float32)), \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255):
            self.assertTrue(live.wait_for_target(object(), object(), hint))
        self.assertIs(hint['rgb_pose'], pose)
        self.assertEqual(hint['rgb_frame'].shape, frame.shape)

    def test_empty_depth_workspace_is_reported_only_for_zero_points(self):
        with tempfile.TemporaryDirectory() as folder:
            fit = Path(folder) / 'depth_fit'
            fit.mkdir()
            diagnostics = fit / 'diagnostics.json'
            self.assertFalse(live.empty_depth_workspace(Path(folder)))
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder)), 5.)
            diagnostics.write_text(json.dumps({'workspace_points': 0}), encoding='utf-8')
            self.assertTrue(live.empty_depth_workspace(Path(folder)))
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder)), 10.)
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder), 2), 20.)
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder), 5), 30.)
            diagnostics.write_text(json.dumps({'workspace_points': 25}), encoding='utf-8')
            self.assertFalse(live.empty_depth_workspace(Path(folder)))
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder)), 5.)
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder), 2), 10.)
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder), 3), 20.)
            self.assertEqual(live.failed_depth_retry_seconds(Path(folder), 4), 30.)

    def test_empty_depth_cooldown_keeps_rgb_search_live(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        now = [0.]
        hint = {'initializer_provider': lambda: SimpleNamespace(estimate=lambda *_: None),
                'references_ready': lambda: True, 'rgb_fallback_timeout_s': 0.,
                'depth_retry_after': 5.}

        def read(*_args):
            now[0] += 1.
            return frame

        with patch.object(live, 'detect_target_box',
                          return_value=np.array([280, 130, 400, 355], np.float32)), \
                patch.object(live, 'read_rgb', side_effect=read) as reads, \
                patch.object(live.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live._wait_for_target(object(), object(), hint, 'Searching'))
        self.assertEqual(reads.call_count, 5)

    def test_rgb_fallback_waits_for_configured_interval(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        now = [0.]
        hint = {'initializer_provider': lambda: SimpleNamespace(estimate=lambda *_: None),
                'references_ready': lambda: True,
                'rgb_fallback_timeout_s': live.DEFAULT_RGB_FALLBACK_TIMEOUT_S}

        def read(*_args):
            now[0] += .2
            return frame

        with patch.object(live, 'detect_target_box',
                          return_value=np.array([280, 130, 400, 355], np.float32)), \
                patch.object(live, 'read_rgb', side_effect=read) as reads, \
                patch.object(live.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255):
            self.assertTrue(live._wait_for_target(object(), object(), hint, 'Searching'))
        self.assertGreaterEqual(now[0], live.DEFAULT_RGB_FALLBACK_TIMEOUT_S)
        self.assertLess(now[0], live.DEFAULT_RGB_FALLBACK_TIMEOUT_S + .8)
        self.assertGreaterEqual(reads.call_count, 9)

    def test_async_confirmation_does_not_start_an_extra_inference(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([280, 130, 400, 355], np.float32)
        submissions = []

        def submit(task, *args):
            submissions.append(True)
            result = Future()
            result.set_result(task(*args))
            return result

        with patch.object(live, 'detect_target_box', return_value=box), \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255):
            self.assertTrue(live._wait_for_target(
                object(), object(), {}, 'Searching', SimpleNamespace(submit=submit)))
        self.assertEqual(len(submissions), 2)

    def test_stale_async_result_resets_the_old_confirmation(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([280, 130, 400, 355], np.float32)
        times = iter([0., .1, 1., 1.1, 1.2])
        now = [0.]

        def read(*_args):
            now[0] = next(times)
            return frame

        def submit(task, *args):
            result = Future()
            result.set_result(task(*args))
            return result

        with patch.object(live, 'detect_target_box', return_value=box) as detect, \
                patch.object(live, 'read_rgb', side_effect=read) as reads, \
                patch.object(live.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255):
            self.assertTrue(live._wait_for_target(
                object(), object(), {}, 'Searching', SimpleNamespace(submit=submit)))
        self.assertEqual(reads.call_count, 5)
        self.assertEqual(detect.call_count, 4)

    def test_rotation_inside_unchanged_box_delays_depth_until_settled(self):
        vertical = np.zeros((480, 640, 3), np.uint8)
        horizontal = vertical.copy()
        cv2.rectangle(vertical, (302, 164), (338, 316), (180, 180, 180), -1)
        cv2.rectangle(horizontal, (244, 222), (396, 258), (180, 180, 180), -1)
        frames = iter([vertical, horizontal, vertical, horizontal] + [horizontal] * 4)
        now = [-1.]

        def read(*_args):
            now[0] += 1.
            return next(frames)

        with patch.object(live, 'detect_target_box',
                          return_value=np.array([230, 150, 410, 330], np.float32)), \
                patch.object(live, 'read_rgb', side_effect=read) as reads, \
                patch.object(live.time, 'monotonic', side_effect=lambda: now[0]), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255):
            self.assertTrue(live.wait_for_target(
                object(), object(), {'settle_seconds': 2.5}))
        self.assertEqual(reads.call_count, 8)

    def test_search_does_not_confirm_different_moving_boxes(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        first = np.array([100, 150, 190, 300], np.float32)
        second = first + [100, 0, 100, 0]
        with patch.object(live, 'detect_target_box',
                          side_effect=[first, second, first, first]) as detect, \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_target(object(), object()))
        self.assertEqual(detect.call_count, 4)

    def test_search_size_limits_are_independent_of_portrait_or_landscape(self):
        self.assertTrue(live.plausible_target_box([200, 80, 320, 330], (480, 640, 3)))
        self.assertTrue(live.plausible_target_box([160, 180, 410, 300], (480, 640, 3)))
        self.assertFalse(live.plausible_target_box([576, 0, 639, 403], (480, 640, 3)))
        self.assertFalse(live.plausible_target_box([0, 0, 639, 479], (480, 640, 3)))

    def test_target_entering_from_an_image_edge_is_kept_for_rgb_search(self):
        self.assertTrue(live.plausible_target_box([0, 178, 52, 318], (480, 640, 3)))
        self.assertTrue(live.plausible_target_box([588, 175, 640, 318], (480, 640, 3)))

    def test_partial_entry_keeps_searching_instead_of_switching_to_depth(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        hint = {}
        with patch.object(live, 'detect_target_box',
                          return_value=np.array([0, 178, 52, 318], np.float32)), \
                patch.object(live, 'read_rgb', return_value=frame) as reads, \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', side_effect=[255, 255, 255, ord('q')]):
            self.assertFalse(live.wait_for_target(object(), object(), hint))
        self.assertEqual(reads.call_count, 4)
        self.assertNotIn('box', hint)

    def test_search_cycles_and_reuses_successful_rotation(self):
        import torch

        frame = np.zeros((480, 640, 3), np.uint8)
        # In a counterclockwise rotated image this maps to [280,130,400,355].
        rotated_box = [130, 240, 355, 360]

        class Detector:
            def __init__(self):
                self.shapes = []

            def predict(self, image, **_kwargs):
                self.shapes.append(image.shape[:2])
                boxes = ([rotated_box] if image.shape[:2] == (640, 480) else [])
                return [SimpleNamespace(boxes=SimpleNamespace(
                    conf=torch.ones(len(boxes)),
                    xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)))]

        detector = Detector()
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_for_target(object(), detector))
        self.assertEqual(detector.shapes, [(480, 640), (640, 480)] * 2)

    def test_manual_search_without_detector_waits_for_space(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        hint = {}
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', side_effect=[255, ord(' ') ]):
            self.assertTrue(live.wait_for_target(object(), None, hint,
                                                 'No YOLO model'))
        self.assertNotIn('box', hint)

    def test_manual_search_can_select_rgb_constraint_box(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        hint = {}
        with patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'selectROI', return_value=(100, 120, 200, 160)), \
                patch.object(cv2, 'waitKey', side_effect=[ord('r'), ord(' ') ]):
            self.assertTrue(live.wait_for_target(object(), None, hint,
                                                 'No YOLO model'))
        np.testing.assert_array_equal(hint['box'], [100, 120, 300, 280])

    def test_inverted_detector_box_maps_back_to_camera_frame(self):
        class FakeDetector:
            def __init__(self):
                self.calls = 0

            def predict(self, *_args, **_kwargs):
                self.calls += 1
                boxes = (np.empty((0, 4), np.float32) if self.calls == 1 else
                         np.array([[187, 183, 338, 266]], np.float32))
                return [SimpleNamespace(boxes=SimpleNamespace(
                    conf=torch.ones(len(boxes)), xyxy=torch.from_numpy(boxes)))]

        box = live.detect_target_box(FakeDetector(), np.zeros((480, 640, 3), np.uint8))
        np.testing.assert_array_equal(box, [302, 214, 453, 297])

    def test_empty_scene_waits_for_stable_entry(self):
        baseline = np.zeros((480, 640, 3), np.uint8)
        entered = baseline.copy()
        entered[200:240, 280:320] = 100
        with patch.object(live, 'read_rgb', side_effect=[
                baseline, entered, entered, entered]) as read_rgb, \
                patch.object(live.time, 'monotonic', side_effect=[0, .1, .1, .5]), \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', return_value=255), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            self.assertTrue(live.wait_to_retry(object(), 'No cluster', True, baseline))
        self.assertEqual(read_rgb.call_count, 4)

    def test_offline_sample_uses_verified_calibration_snapshot(self):
        expected = SAMPLE / "rgb_depth_extrinsics.npz"
        self.assertEqual(live.sample_extrinsics(SAMPLE, None), expected)
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder)
            shutil.copy2(SAMPLE / "manifest.json", sample / "manifest.json")
            (sample / expected.name).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "missing or changed"):
                live.sample_extrinsics(sample, None)

    def test_rgb_motion_during_depth_capture_is_rejected(self):
        before = live.read_image(SAMPLE / "rgb_before.png")
        after = live.read_image(SAMPLE / "rgb_after.png")
        K, dist, _ = live.load_calibration(live.DEFAULT_INTRINSICS)
        corners = live.load_model_box(live.DEFAULT_MODEL)
        rvec = np.array([0, np.pi / 2, 0], np.float64)
        tvec = np.array([52, -27, 709], np.float64)
        stable = live.validate_rgb_stillness(before, after, corners,
                                              K, dist, rvec, tvec)
        self.assertLess(np.linalg.norm(stable["global_shift_px"]), 1)
        shifted = cv2.warpAffine(after, np.array([[1, 0, 8], [0, 1, 0]], np.float32),
                                 (after.shape[1], after.shape[0]))
        with self.assertRaisesRegex(RuntimeError, "scene changed"):
            live.validate_rgb_stillness(before, shifted, corners,
                                         K, dist, rvec, tvec)

    def test_real_depth_sample_produces_rgb_camera_center(self):
        with tempfile.TemporaryDirectory() as folder:
            pose_path = live.initialize_capture(
                SAMPLE / "rgb_before.png", SAMPLE / "depth", Path(folder),
                live.DEFAULT_MODEL, live.sample_extrinsics(SAMPLE, None),
                live.DEFAULT_INTRINSICS, live.Settings(), 12,
                SAMPLE / "rgb_after.png")
            pose = json.loads(pose_path.read_text(encoding="utf-8"))
            np.testing.assert_allclose(pose["tvec_rgb_model_mm"],
                                       [51.9742, -27.0088, 709.0951], atol=1.0)
            self.assertTrue(pose["depth_pose_quality"]["accepted"])
            self.assertIsNotNone(pose["rgb_stillness"])
            self.assertTrue((Path(folder) / "rgb_initial_box.png").is_file())
            diagnostics = json.loads(
                (Path(folder) / 'depth_fit' / 'diagnostics.json').read_text(encoding='utf-8'))
            with patch.object(live, 'initialize_depth',
                              side_effect=AssertionError('Depth was fitted twice')):
                cached_path = live.initialize_capture(
                    SAMPLE / 'rgb_before.png', SAMPLE / 'depth', Path(folder) / 'cached',
                    live.DEFAULT_MODEL, live.sample_extrinsics(SAMPLE, None),
                    live.DEFAULT_INTRINSICS, live.Settings(), 12,
                    SAMPLE / 'rgb_after.png', diagnostics)
                cached = json.loads(cached_path.read_text(encoding='utf-8'))
                np.testing.assert_allclose(cached['T_rgb_model'], pose['T_rgb_model'])
                with patch.object(live, 'validate_rgb_stillness',
                                  side_effect=RuntimeError('RGB scene changed')):
                    with self.assertRaisesRegex(RuntimeError, 'RGB scene changed'):
                        live.initialize_capture(
                            SAMPLE / 'rgb_before.png', SAMPLE / 'depth', Path(folder) / 'moved',
                            live.DEFAULT_MODEL, live.sample_extrinsics(SAMPLE, None),
                            live.DEFAULT_INTRINSICS, live.Settings(), 12,
                            SAMPLE / 'rgb_after.png', diagnostics)

    def test_precomputed_fit_still_rejects_bad_pose_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(live, 'initialize_depth',
                              side_effect=AssertionError('Depth was fitted twice')):
                with self.assertRaisesRegex(RuntimeError, 'Depth pose rejected'):
                    live.initialize_capture(
                        SAMPLE / 'rgb_before.png', SAMPLE / 'depth', Path(folder),
                        live.DEFAULT_MODEL, live.sample_extrinsics(SAMPLE, None),
                        live.DEFAULT_INTRINSICS, live.Settings(), 12,
                        diagnostics={'pose_quality': {'accepted': False}})

    def test_partial_depth_requires_independent_rgb_verification(self):
        diagnostics = {'pose_quality': {'accepted': True, 'requires_rgb_verification': True},
                       'pose': {'model_to_camera': np.eye(4).tolist()}}
        diagnostics['pose']['model_to_camera'][2][3] = 700.
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'partial geometry conflicts'):
                live.initialize_capture(
                    SAMPLE / 'rgb_before.png', SAMPLE / 'depth', Path(folder),
                    live.DEFAULT_MODEL, live.sample_extrinsics(SAMPLE, None),
                    live.DEFAULT_INTRINSICS, live.Settings(), 12,
                    diagnostics=diagnostics)


if __name__ == "__main__":
    unittest.main()
