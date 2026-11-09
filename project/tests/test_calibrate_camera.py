import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

from core import calibrate_camera as camera


class CalibrateCameraTest(unittest.TestCase):
    def test_open_camera_does_not_reapply_matching_dimensions(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            camera.cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            camera.cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        }.get(prop, 0.0)
        with patch.object(camera.cv2, "VideoCapture", return_value=cap) as opener:
            result = camera.open_camera(2, 640, 480)
        opener.assert_called_once_with(2, camera.CAMERA_BACKEND)
        self.assertIs(result, cap)
        cap.set.assert_called_once_with(camera.cv2.CAP_PROP_BUFFERSIZE, 1)

    def test_open_camera_sets_dimensions_when_driver_returns_different_mode(self):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            camera.cv2.CAP_PROP_FRAME_WIDTH: 1280.0,
            camera.cv2.CAP_PROP_FRAME_HEIGHT: 720.0,
        }.get(prop, 0.0)
        with patch.object(camera.cv2, "VideoCapture", return_value=cap):
            camera.open_camera(2, 640, 480)
        self.assertEqual(cap.set.call_args_list, [
            call(camera.cv2.CAP_PROP_FRAME_WIDTH, 640),
            call(camera.cv2.CAP_PROP_FRAME_HEIGHT, 480),
            call(camera.cv2.CAP_PROP_BUFFERSIZE, 1),
        ])

    def test_calibration_rejects_capture_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "images"
            images.mkdir()
            camera.write_image(
                images / "frame_0001.png", np.zeros((480, 640, 3), dtype=np.uint8))
            settings = {
                "subject": "checkerboard",
                "board_size": [9, 6],
                "square_size_mm": 25.0,
                "image_size": [640, 480],
            }
            args = Namespace(images=images, output=root / "output", square_mm=25.0)
            for changed, expected in (
                ({"board_size": [8, 6]}, "Checkerboard dimensions"),
                ({"subject": "object"}, "checkerboard captures"),
                ({"image_size": [800, 600]}, "Image size differs"),
            ):
                with self.subTest(changed=changed):
                    current = {**settings, **changed}
                    (images / "capture_settings.json").write_text(
                        json.dumps(current), encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, expected):
                        camera.calibrate(args)


if __name__ == "__main__":
    unittest.main()
