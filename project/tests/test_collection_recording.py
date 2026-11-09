import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.collection_recording import SessionRecording, wrap_camera, collection_key
from core import live_depth_6dof as live


class FakeCamera:
    def __init__(self):
        self.closed = False

    def read(self):
        return True, np.full((48, 64, 3), 70, np.uint8)

    def release(self):
        self.closed = True


class CollectionRecordingTest(unittest.TestCase):
    def test_search_frames_markers_and_pose_share_frame_indices_after_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'continuous'
            rec = SessionRecording(root, 100)
            args = SimpleNamespace(collection_recording=rec)
            cap = wrap_camera(FakeCamera(), args)
            self.assertIs(wrap_camera(cap, args), cap)
            cap.read()  # Before initialization: no pose yet.
            self.assertEqual(collection_key(cap, ord('U')), ord('U'))
            cap.release()
            cap = wrap_camera(FakeCamera(), args)
            cap.read()
            rec.pose(dict(valid=False, display_status='SEARCHING'))
            cap.read()
            rec.pose(dict(valid=True, tvec_mm=[0, 0, 600]))
            cap.release()
            rec.close()
            rec.close()
            meta = json.loads((root / 'metadata.json').read_text())
            self.assertEqual(meta['recorded_frames'], 3)
            self.assertTrue(np.all(np.diff(meta['frame_timestamps_s']) > 0))
            records = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
            markers = [r for r in records if r['event'] == 'user_marker']
            self.assertEqual(markers[0]['label'], 'visible_but_unrecognized')
            self.assertEqual(markers[0]['video_frame_index'], 0)
            poses = [r for r in records if r['event'] == 'tracker_observation']
            self.assertEqual([p['video_frame_index'] for p in poses], [1, 2])
            saved = cv2.VideoCapture(str(root / 'rgb.avi'))
            try:
                self.assertEqual(int(saved.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
                ok, image = saved.read()
                self.assertTrue(ok)
                self.assertLess(abs(float(image.mean()) - 70), 2)
            finally:
                saved.release()

    def test_failed_writer_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            rec = SessionRecording(Path(folder) / 'continuous', 100)
            with patch.object(cv2, 'VideoWriter') as writer:
                writer.return_value.isOpened.return_value = False
                with self.assertRaises(OSError):
                    rec.write(np.zeros((48, 64, 3), np.uint8))
            rec.close()
            self.assertEqual(json.loads((rec.directory / 'metadata.json').read_text())['recorded_frames'], 0)

    def test_deadline_and_no_collection_are_safe(self):
        cap = FakeCamera()
        self.assertIs(wrap_camera(cap, SimpleNamespace()), cap)
        self.assertEqual(collection_key(None, 255), 255)
        with tempfile.TemporaryDirectory() as folder:
            rec = SessionRecording(Path(folder) / 'continuous', 10)
            with patch('core.collection_recording.time.perf_counter', return_value=rec.started + 11):
                self.assertEqual(rec.key(255), ord('q'))
            rec.close()

    def test_recovery_camera_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            rec = SessionRecording(Path(folder) / 'continuous', 100)
            args = SimpleNamespace(collection_recording=rec, rgb_reopen_attempts=1,
                                   output_dir=Path(folder))
            frame = np.zeros((48, 64, 3), np.uint8)
            with patch.object(live, 'reopen_rgb', return_value=(FakeCamera(), frame)), \
                 patch.object(live, 'run_with_preview', side_effect=lambda task, *_: task()):
                cap, _ = live.recover_rgb(args, 0, frame)
                cap.read()
            rec.close()
            self.assertEqual(len(rec.timestamps), 1)


if __name__ == '__main__':
    unittest.main()
