"""Regression checks for native-worker hangs and camera recovery."""
import itertools
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core import live_depth_6dof as live


class DepthWorkerRecoveryTest(unittest.TestCase):
    def launch_script(self, script, children):
        popen = subprocess.Popen

        def launch(_command, **kwargs):
            child = popen([sys.executable, '-u', '-c', script], **kwargs)
            children.append(child)
            return child
        return launch

    def test_hung_worker_is_killed_reaped_and_logged(self):
        children = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(live.subprocess, 'Popen', side_effect=self.launch_script(
                    "import time; print('WORKER STARTED', flush=True); time.sleep(60)", children)):
                with self.assertRaisesRegex(RuntimeError, 'watchdog timeout'):
                    live.run_depth_worker(output, 0, .5)
            self.assertIsNotNone(children[0].poll(), 'a hung native child was left running')
            log = (output / 'worker.log').read_text(encoding='utf-8')
            self.assertIn('WATCHDOG timeout', log)
            self.assertIn('WORKER STARTED', log)

    def test_cancel_reaps_child_before_returning_to_ui(self):
        children = []
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(live.subprocess, 'Popen', side_effect=self.launch_script(
                    'import time; time.sleep(60)', children)):
                with self.assertRaises(live.PreviewCancelled):
                    live.run_depth_worker(Path(folder), 0, 8, cancel_event=cancel)
            self.assertIsNotNone(children[0].poll())

    def test_abnormal_worker_exit_keeps_progress_and_output(self):
        children = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'capture_progress.json').write_text(
                json.dumps({'stage': 'waiting_depth_frame', 'index': 2}), encoding='utf-8')
            with patch.object(live.subprocess, 'Popen', side_effect=self.launch_script(
                    "import sys; print('native failure details'); sys.exit(17)", children)):
                with self.assertRaisesRegex(RuntimeError, '(?s)exit 17.*waiting_depth_frame'):
                    live.run_depth_worker(output, 0, 8)
            self.assertIn('native failure details',
                          (output / 'worker.log').read_text(encoding='utf-8'))

    def test_q_cancels_background_depth_task(self):
        cancel = threading.Event()

        def task():
            self.assertTrue(cancel.wait(2), 'preview ignored Q during depth capture')
            raise live.PreviewCancelled()

        with patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=ord('q')), \
                patch.object(cv2, 'getWindowProperty', return_value=1):
            with self.assertRaises(live.PreviewCancelled):
                live.run_with_preview(task, np.zeros((480, 640, 3), np.uint8),
                                      'Depth', cancel_event=cancel)

    def test_rgb_reopen_retries_without_closing_the_window(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        camera = object()
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder), rgb_reopen_attempts=1)
            with patch.object(live, 'reopen_rgb', side_effect=[
                    RuntimeError('camera busy'), cv2.error('DirectShow interrupted'),
                    (camera, frame)]) as reopen, \
                    patch.object(live, 'run_with_preview', side_effect=lambda task, *_: task()), \
                    patch.object(live.time, 'monotonic', side_effect=itertools.count(step=2)):
                recovered, _ = live.recover_rgb(args, 0, frame)
            self.assertIs(recovered, camera)
            self.assertEqual(reopen.call_count, 3)
            events = [json.loads(line) for line in
                      (Path(folder) / 'session_events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(events), 2)
            self.assertIn('DirectShow interrupted', events[-1]['traceback'])

    def live_args(self, output):
        return SimpleNamespace(
            camera_name='USB Camera', camera=0, output_dir=output,
            preview_frames=1, extrinsics=live.DEFAULT_EXTRINSICS,
            depth_warmup=0, depth_timeout_s=8, rgb_reopen_attempts=1,
            model=live.DEFAULT_MODEL, intrinsics=live.DEFAULT_INTRINSICS,
            settings=live.Settings(), box_display_y_offset_px=12.)

    def mock_live(self, stack, frame):
        stack.enter_context(patch.object(live.PinnedCamera, 'bind', return_value=None))
        camera = SimpleNamespace(release=lambda: None)
        for name, value in [('resolve_camera_index', (0, None)), ('open_camera', camera),
                            ('read_rgb', frame), ('fit_depth_capture', {}),
                            ('wait_to_retry', True)]:
            stack.enter_context(patch.object(live, name, return_value=value))
        stack.enter_context(patch.object(live, 'run_with_preview',
                                        side_effect=lambda task, *_a, **_kw: task()))
        stack.enter_context(patch.object(live, 'initialize_capture',
                                        side_effect=lambda *args: args[2] / 'pose.json'))
        for name in ('namedWindow', 'imshow', 'destroyAllWindows'):
            stack.enter_context(patch.object(cv2, name))
        stack.enter_context(patch.object(cv2, 'waitKey', return_value=255))
        return camera

    def test_depth_failure_recovers_and_next_attempt_tracks(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder)
            camera = self.mock_live(stack, frame)
            stack.enter_context(patch.object(live, 'reopen_rgb', return_value=(camera, frame)))
            stack.enter_context(patch.object(live, 'run_depth_worker',
                                            side_effect=[RuntimeError('driver exited'), None]))
            tracker = stack.enter_context(patch.object(live, 'run_tracker', return_value=0))
            self.assertEqual(live.run_live(self.live_args(output)), 0)
            self.assertEqual(tracker.call_count, 1)
            timing = json.loads((output / 'attempt_001' / 'initialization_timing.json')
                                .read_text(encoding='utf-8'))
            self.assertEqual(timing['status'], 'rejected')
            self.assertEqual(tracker.call_args.args[2], output / 'attempt_002' / 'tracking')

    def test_post_tracker_rgb_reopen_failure_recovers(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            output = Path(folder)
            camera = self.mock_live(stack, frame)
            stack.enter_context(patch.object(live, 'reopen_rgb', side_effect=[
                (camera, frame), RuntimeError('camera busy after reacquisition'),
                (camera, frame), (camera, frame)]))
            stack.enter_context(patch.object(live.time, 'monotonic',
                                            side_effect=itertools.count(step=2)))
            stack.enter_context(patch.object(live, 'run_depth_worker'))
            tracker = stack.enter_context(patch.object(live, 'run_tracker', side_effect=[2, 0]))
            self.assertEqual(live.run_live(self.live_args(output)), 0)
            self.assertEqual(tracker.call_count, 2)
            events = (output / 'session_events.jsonl').read_text(encoding='utf-8')
            self.assertIn('camera busy after reacquisition', events)

    def test_main_persists_unexpected_exception(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output_dir=Path(folder), offline_sample=None, extrinsics=None)
            with patch.object(live, 'parse_args', return_value=args), \
                    patch.object(live, 'run_live', side_effect=AttributeError('unexpected driver result')):
                self.assertEqual(live.main(), 1)
            self.assertIn('AttributeError: unexpected driver result',
                          (Path(folder) / 'last_error.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
