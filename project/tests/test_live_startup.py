"""Startup readiness and ownership checks without camera hardware."""
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core import live_depth_6dof as live


class LiveStartupTest(unittest.TestCase):
    def test_missing_bundled_detector_does_not_block_live_startup(self):
        with patch.object(live, 'DETECTOR_MODEL', Path('/tmp/missing-reacquire-model.pt')):
            args = live.parse_args(['--camera', '0'])
        self.assertIsNone(args.reacquire_model)

    def test_import_does_not_load_inference_or_geometry_runtimes(self):
        result = subprocess.run([
            sys.executable, '-c',
            'import json,sys; from core import live_depth_6dof; '
            'print(json.dumps([n for n in ("torch","open3d","trimesh") if n in sys.modules]))',
        ], cwd=live.PROJECT_ROOT, capture_output=True, text=True, timeout=15, check=True)
        self.assertEqual(json.loads(result.stdout), [])

    def test_camera_open_overlaps_detector_preparation(self):
        camera_started, detector_started = threading.Event(), threading.Event()
        camera, detector = SimpleNamespace(release=lambda: None), object()
        frame = np.zeros((480, 640, 3), np.uint8)

        def open_camera(*_args):
            camera_started.set()
            self.assertTrue(detector_started.wait(2), 'camera blocked detector preparation')
            return camera

        def prepare_detector(*_args):
            detector_started.set()
            self.assertTrue(camera_started.wait(2), 'detector blocked camera opening')
            return detector

        args = SimpleNamespace(reacquire_model=live.DETECTOR_MODEL)
        with patch.object(live, 'open_camera', side_effect=open_camera), \
                patch.object(live, 'prepare_live_detector', side_effect=prepare_detector), \
                patch.object(live, 'read_rgb', return_value=frame), \
                patch.object(live, 'run_with_preview', side_effect=lambda task, *_: task()):
            result = live.prepare_live_session(args, 0, {})
        self.addCleanup(result[1].close)
        self.assertIs(result[0], camera)
        self.assertIs(result[1].loading.result(timeout=2), detector)

    def test_camera_returns_before_slow_detector_is_ready(self):
        loading, finish = threading.Event(), threading.Event()
        camera = SimpleNamespace(release=lambda: None)
        def slow_detector(*_args):
            loading.set()
            self.assertTrue(finish.wait(2))
            return object()
        def open_camera(*_args):
            self.assertTrue(loading.wait(2))
            return camera
        detector = None
        try:
            with patch.object(live, 'open_camera', side_effect=open_camera), \
                    patch.object(live, 'prepare_live_detector', side_effect=slow_detector), \
                    patch.object(live, 'run_with_preview', side_effect=lambda task, *_: task()):
                cap, detector = live.prepare_live_session(
                    SimpleNamespace(reacquire_model=live.DETECTOR_MODEL), 4, {})
            self.assertIs(cap, camera)
            self.assertFalse(detector.loading.done(), 'camera startup waited for YOLO')
        finally:
            finish.set()
            if detector is not None:
                detector.close()

    def test_search_starts_before_cad_ready_and_fit_waits_for_cad(self):
        cad_started, cad_release, cad_ready = (threading.Event() for _ in range(3))
        frame = np.zeros((480, 640, 3), np.uint8)
        released = []
        camera = SimpleNamespace(release=lambda: released.append(True))

        def geometry(*_args):
            cad_started.set()
            self.assertTrue(cad_release.wait(2), 'search waited for CAD before looking for target')
            cad_ready.set()

        def search(*_args):
            self.assertTrue(cad_started.wait(2))
            self.assertFalse(cad_ready.is_set(), 'search was delayed until CAD was ready')
            return True

        def depth(*_args):
            cad_release.set()

        def fit(*_args):
            self.assertTrue(cad_ready.is_set(), 'fit raced with CAD cache construction')
            return {}

        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder)
            args = SimpleNamespace(
                camera_name='USB Camera', camera=0, output_dir=output, preview_frames=1,
                extrinsics=live.DEFAULT_EXTRINSICS, depth_warmup=0, depth_timeout_s=8,
                rgb_reopen_attempts=1, model=live.DEFAULT_MODEL,
                intrinsics=live.DEFAULT_INTRINSICS, settings=live.Settings(),
                box_display_y_offset_px=12.)
            stack.enter_context(patch.object(live, 'resolve_camera_index', return_value=(0, None)))
            stack.enter_context(patch.object(live.PinnedCamera, 'bind', return_value=None))
            stack.enter_context(patch.object(live, 'prepare_live_session',
                                            return_value=(camera, object())))
            for name, task in [('prepare_live_geometry', geometry), ('wait_for_target', search),
                               ('run_depth_worker', depth), ('fit_depth_capture', fit)]:
                stack.enter_context(patch.object(live, name, side_effect=task))
            stack.enter_context(patch.object(live, 'read_rgb', return_value=frame))
            stack.enter_context(patch.object(live, 'reopen_rgb', return_value=(camera, frame)))
            stack.enter_context(patch.object(live, 'run_with_preview',
                                            side_effect=lambda task, *_a, **_kw: task()))
            stack.enter_context(patch.object(live, 'initialize_capture',
                                            return_value=output / 'pose.json'))
            stack.enter_context(patch.object(live, 'run_tracker', return_value=0))
            for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch.object(cv2, name))
            stack.enter_context(patch.object(cv2, 'waitKey', return_value=255))
            self.assertEqual(live.run_live(args), 0)
            timing = json.loads((output / 'startup_timing.json').read_text(encoding='utf-8'))
            self.assertEqual(timing['status'], 'geometry_ready')
            self.assertIn('search_ready_s', timing)

    def test_rejected_depth_pose_backs_off_while_rgb_search_continues(self):
        # Reproduce the real sequence: depth fit succeeds, RGB handoff rejects
        # the partial pose, then the next search must retain a growing cooldown.
        for detector in (object(), None):
            with self.subTest(detector=detector is not None), \
                    tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                output = Path(folder)
                frame = np.zeros((480, 640, 3), np.uint8)
                camera = SimpleNamespace(release=lambda: None)
                initializer = SimpleNamespace(views=[object()])
                args = SimpleNamespace(
                    camera_name='USB Camera', camera=0, output_dir=output,
                    preview_frames=1, extrinsics=live.DEFAULT_EXTRINSICS,
                    depth_warmup=0, depth_timeout_s=8, rgb_reopen_attempts=1,
                    model=live.DEFAULT_MODEL, intrinsics=live.DEFAULT_INTRINSICS,
                    settings=live.Settings(), box_display_y_offset_px=12.,
                    rgb_first=True, rgb_initializer=initializer)
                searches = []

                def search(_cap, _detector, hint, *_args):
                    searches.append(dict(hint))
                    return len(searches) <= 2

                values = dict(resolve_camera_index=(0, None),
                              prepare_live_session=(camera, detector),
                              prepare_rgb_initializer=initializer,
                              prepare_live_geometry=None, read_rgb=frame,
                              reopen_rgb=(camera, frame), run_depth_worker=None,
                              fit_depth_capture={})
                for name, value in values.items():
                    stack.enter_context(patch.object(live, name, return_value=value))
                stack.enter_context(patch.object(live.PinnedCamera, 'bind', return_value=None))
                stack.enter_context(patch.object(live, 'wait_for_target', side_effect=search))
                stack.enter_context(patch.object(live, 'run_with_preview',
                                                side_effect=lambda task, *_a, **_k: task()))
                stack.enter_context(patch.object(live, 'initialize_capture', side_effect=
                    RuntimeError('Depth pose rejected: partial geometry conflicts with RGB target')))
                retry = stack.enter_context(patch.object(live, 'wait_to_retry', return_value=False))
                for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
                    stack.enter_context(patch.object(cv2, name))
                stack.enter_context(patch.object(cv2, 'waitKey', return_value=255))
                self.assertEqual(live.run_live(args), 0)
                retry.assert_not_called()
                self.assertEqual(len(searches), 3)
                self.assertEqual(searches[1].get('depth_fit_failures'), 1)
                self.assertEqual(searches[2].get('depth_fit_failures'), 2)
                self.assertGreater(searches[2]['depth_retry_after'], searches[1]['depth_retry_after'])
                events = [json.loads(line) for line in
                          (output / 'session_events.jsonl').read_text().splitlines()]
                cooldowns = [event['cooldown_s'] for event in events
                             if event['stage'] == 'failed_depth_rgb_retry']
                self.assertEqual(cooldowns, [5., 10.])

    def test_async_search_keeps_running_during_depth_cooldown_and_counts_retries(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        box = np.array([200., 150., 400., 300.])
        now = [0.]
        search = SimpleNamespace(estimate=lambda *_a, **_k: None)
        initializer = SimpleNamespace(new_search=lambda: search)
        detector = SimpleNamespace(get_box=lambda _frame: box)
        hint = dict(initializer_provider=lambda: initializer, depth_retry_after=5.,
                    rgb_fallback_timeout_s=0., settle_seconds=.35)

        class ImmediateExecutor:
            def __init__(self, **_kwargs): pass
            def submit(self, fn, *args):
                result = Future()
                result.set_result(fn(*args))
                return result
            def shutdown(self, **_kwargs): pass

        def read(*_args):
            now[0] += .1
            if now[0] > 12.:
                self.fail('depth fallback never became ready')
            return frame

        with ExitStack() as stack:
            stack.enter_context(patch.object(live, 'ThreadPoolExecutor', ImmediateExecutor))
            stack.enter_context(patch.object(live, 'read_rgb', side_effect=read))
            stack.enter_context(patch.object(live.time, 'monotonic', side_effect=lambda: now[0]))
            stack.enter_context(patch.object(cv2, 'imshow'))
            stack.enter_context(patch.object(cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(cv2, 'waitKey', return_value=255))
            self.assertTrue(live.wait_for_3d_target(object(), detector, hint))
            self.assertGreaterEqual(now[0], 5.)
            first_checks = hint['search_stats']['global_checks']
            self.assertGreater(first_checks, 20)
            hint['depth_retry_after'] = 10.
            self.assertTrue(live.wait_for_3d_target(object(), detector, hint))
            self.assertGreaterEqual(now[0], 10.)
            self.assertGreater(hint['search_stats']['global_checks'], first_checks)
            self.assertEqual(hint['search_stats']['depth_fallbacks'], 2)
            # A verified RGB result can finish immediately during the cooldown.
            hint['depth_retry_after'] = 100.
            pose = object()
            search.estimate = lambda *_a, **_k: pose
            search.K = search.dist = search.vertices = None
            with patch('core.rgb_global_initializer.advance_pose', return_value=pose):
                self.assertTrue(live.wait_for_3d_target(object(), detector, hint))
            self.assertIs(hint['rgb_pose'], pose)
            self.assertLess(now[0], 11.)
            self.assertEqual(hint['search_stats']['depth_fallbacks'], 2)


if __name__ == '__main__':
    unittest.main()
