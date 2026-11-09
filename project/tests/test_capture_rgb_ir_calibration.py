import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import capture_rgb_ir_calibration as capture


class FakeCamera:
    def __init__(self):
        self.released = False

    def read(self):
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class CaptureRgbIrCalibrationTest(unittest.TestCase):
    def test_save_rechecks_corners_on_current_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            camera = FakeCamera()
            args = Namespace(output=Path(folder), target_count=1)
            prior_corners = np.zeros((54, 1, 2), dtype=np.float32)
            with (patch.object(capture, "find_board", side_effect=[
                    (True, prior_corners), (False, None)]) as find_board,
                  patch.object(capture, "capture_pair") as capture_pair,
                  patch.object(capture.cv2, "imshow"),
                  patch.object(capture.cv2, "waitKey",
                               side_effect=[-1, ord("s"), ord("q")]),
                  patch.object(capture.cv2, "destroyAllWindows")):
                capture.run_interactive(camera, args)
            self.assertEqual(find_board.call_count, 2)
            capture_pair.assert_not_called()
            self.assertTrue(camera.released)


if __name__ == "__main__":
    unittest.main()
