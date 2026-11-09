import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.tracking_diagnostics import TrackingDiagnostics


class TrackingDiagnosticsTest(unittest.TestCase):
    def make_recorder(self, folder, **kwargs):
        return TrackingDiagnostics(folder, np.eye(3), np.zeros(5), Path('model.stl'),
                                   Path('intrinsics.npz'), **kwargs)

    @staticmethod
    def count_video_frames(path):
        cap = cv2.VideoCapture(str(path))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        return count

    def test_recording_keeps_recent_frames_with_pose_and_actual_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.make_recorder(folder, record=True, duration_s=1,
                                          segment_duration_s=.25, fps=30)
            frame = np.full((64, 80, 3), 50, np.uint8)
            records = [dict(frame_index=6, elapsed_s=10., valid=True, status='TRACKING'),
                       dict(frame_index=7, elapsed_s=10.15, valid=False, status='TARGET LOST'),
                       dict(frame_index=8, elapsed_s=11.1, valid=True, status='TRACKING')]
            for record in records:
                recorder.write(frame, record)
            recorder.close()

            metadata = json.loads((Path(folder) / 'metadata.json').read_text())
            np.testing.assert_allclose(metadata['frame_timestamps_s'], [0.15, 1.1])
            self.assertEqual(metadata['recorded_frames'], 2)
            self.assertEqual(metadata['rolling_window_s'], 1.)
            rows = [json.loads(x) for x in (Path(folder) / 'poses.jsonl').read_text().splitlines()]
            self.assertEqual([x['video_frame_index'] for x in rows], [0, 1])
            self.assertEqual([x['frame_index'] for x in rows], [7, 8])
            self.assertFalse(rows[0]['valid'])
            self.assertEqual(self.count_video_frames(Path(folder) / 'rgb.avi'), len(rows))
            self.assertEqual(list(Path(folder).glob('rgb.part-*.avi')), [])

    def test_old_completed_segments_are_deleted_while_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.make_recorder(folder, record=True, duration_s=1,
                                          segment_duration_s=.25, fps=30)
            frame = np.zeros((32, 48, 3), np.uint8)
            for index, elapsed in enumerate((0., .3, .6, 1.3, 1.6)):
                recorder.write(frame, dict(frame_index=index, elapsed_s=elapsed,
                                           valid=True, status='TRACKING'))
                if elapsed == 1.3:
                    parts = list(Path(folder).glob('rgb.part-*.avi'))
                    self.assertLessEqual(len(parts), 3)
            recorder.close()
            self.assertEqual(list(Path(folder).glob('rgb.part-*.avi')), [])
            metadata = json.loads((Path(folder) / 'metadata.json').read_text())
            self.assertLessEqual(metadata['frame_timestamps_s'][-1]
                                 - metadata['frame_timestamps_s'][0], 1.)

    def test_snapshot_preserves_raw_pixels_and_invalid_status_separately_from_overlay(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = self.make_recorder(folder)
            frame = np.full((64, 80, 3), 70, np.uint8)
            overlay = frame.copy()
            overlay[10:20] = 255
            record = dict(frame_index=9, elapsed_s=4.3, valid=False, status='TARGET LOST',
                          rvec_rad=[0, 0, 0], tvec_mm=[0, 0, 500], display_pose=None)
            path = recorder.snapshot(frame, record, overlay)
            saved = json.loads(path.read_text())
            np.testing.assert_array_equal(cv2.imread(str(path.parent / saved['raw_image'])), frame)
            np.testing.assert_array_equal(cv2.imread(str(path.parent / saved['overlay_image'])), overlay)
            self.assertFalse(saved['valid'])
            self.assertEqual(saved['tvec_mm'], [0, 0, 500])
            self.assertEqual(saved['image_size'], [80, 64])
            recorder.close()


if __name__ == '__main__':
    unittest.main()
