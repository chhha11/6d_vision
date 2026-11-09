import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.capture_motion_dataset import (
    ClipRecorder, capture_metrics, completed_clip_count, completed_labels,
    next_clip_number, parse_labels,
)


class CaptureMotionDatasetTest(unittest.TestCase):
    def test_clip_number_never_reuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "clip_0001").mkdir()
            (root / "clip_0003").mkdir()
            (root / "clip_bad").mkdir()
            self.assertEqual(next_clip_number(root), 4)

    def test_completed_count_ignores_short_and_failed_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index, status in enumerate(("complete", "short", "failed"), 1):
                clip = root / f"clip_{index:04d}"
                clip.mkdir()
                (clip / "metadata.json").write_text(json.dumps({"status": status}))
            self.assertEqual(completed_clip_count(root), 1)

    def test_metrics_record_real_frame_rate_and_short_clip(self):
        timestamps = [index / 20.0 for index in range(400)]
        complete = capture_metrics(100.0, 120.0, timestamps, 20.0, 15.0, "duration")
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["frame_count"], 400)
        self.assertEqual(complete["average_capture_fps"], 20.0)
        short = capture_metrics(100.0, 110.0, timestamps[:200], 20.0, 15.0, "manual")
        self.assertEqual(short["status"], "short")
        low_fps = capture_metrics(100.0, 120.0, timestamps[:200], 20.0, 15.0,
                                  "duration")
        self.assertEqual(low_fps["status"], "low_fps")

    def test_labels_reject_empty_entry(self):
        self.assertEqual(parse_labels("translation, rotation"),
                         ("translation", "rotation"))
        with self.assertRaises(ValueError):
            parse_labels("translation,,occlusion")
        with self.assertRaises(ValueError):
            parse_labels("translation,translation")

    def test_recorder_decodes_all_written_frames_and_resumes_by_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("core.capture_motion_dataset.time.perf_counter",
                       side_effect=[100.0, 101.0]):
                recorder = ClipRecorder(root / "clip_0001", "translation", {},
                                        30.0, 0.5, 3.0, (64, 48))
                for index in range(4):
                    frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
                    recorder.append(frame, 100.1 + index * 0.2)
                metadata = recorder.finish("duration")
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["video_decoded_frame_count"], 4)
            self.assertTrue(metadata["video_retimed"])
            self.assertAlmostEqual(metadata["video_playback_duration_s"], 1.0,
                                   delta=0.05)
            capture = cv2.VideoCapture(str(root / "clip_0001" / "rgb.avi"))
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 4.0,
                                   delta=0.05)
            capture.release()
            self.assertEqual(completed_labels(root), {"translation"})
            (root / "clip_0001" / "rgb.avi").unlink()
            self.assertEqual(completed_labels(root), set())


if __name__ == "__main__":
    unittest.main()
