import math
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core import astra_depth_probe as probe


class AstraDepthProbeTest(unittest.TestCase):
    def test_frame_timeout_does_not_call_blocking_read(self):
        stream = SimpleNamespace(read_frame=lambda: self.fail('read_frame must not block'))
        with patch.object(probe.openni2, 'wait_for_any_stream', return_value=None) as wait:
            with self.assertRaisesRegex(TimeoutError, 'depth frame timeout'):
                probe.read_frame_with_timeout(stream, .25)
        wait.assert_called_once_with([stream], timeout=.25)

    def test_driver_is_closed_before_saving_owned_depth_copies(self):
        closed = []
        buffer = np.full((4, 6), 700, np.uint16)

        class Frame:
            width, height, frameIndex, timestamp = 6, 4, 1, 1000

            def get_buffer_as_uint16(self):
                return buffer

            def close(self):
                buffer[:] = 0  # exercise ownership after the driver releases a frame

        class Stream:
            def get_mirroring_enabled(self):
                return False

            def set_mirroring_enabled(self, value):
                pass

            def start(self):
                pass

            def read_frame(self):
                buffer[:] = 700
                return Frame()

            def get_video_mode(self):
                return SimpleNamespace(resolutionX=6, resolutionY=4, fps=30, pixelFormat='1mm')

            def get_horizontal_fov(self):
                return math.pi / 2

            get_vertical_fov = get_horizontal_fov

            def close(self):
                closed.append('stream')

        stream = Stream()
        device = SimpleNamespace(
            get_device_info=lambda: SimpleNamespace(name=b'Astra', vendor=b'Orbbec', uri=b'USB'),
            is_image_registration_mode_supported=lambda _: False,
            get_image_registration_mode=lambda: 'OFF', create_depth_stream=lambda: stream,
            close=lambda: closed.append('device'))
        save_archive = np.savez_compressed

        def save(*args, **kwargs):
            self.assertEqual(closed, ['stream', 'device', 'runtime'])
            np.testing.assert_array_equal(kwargs['depth_frames_mm'], 700)
            save_archive(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), sdk=Path(folder), warmup=1,
                                   frames=5, frame_timeout_s=2., register_to_color=False)
            with patch.object(probe.openni2, 'initialize'), \
                    patch.object(probe.openni2.Device, 'enumerate_uris', return_value=['USB']), \
                    patch.object(probe.openni2.Device, 'open_any', return_value=device), \
                    patch.object(probe.openni2, 'wait_for_any_stream', return_value=stream), \
                    patch.object(probe.openni2, 'unload', side_effect=lambda: closed.append('runtime')), \
                    patch.object(probe.np, 'savez_compressed', side_effect=save):
                probe.capture(args)
            self.assertEqual(closed, ['stream', 'device', 'runtime'])
            progress = json.loads((Path(folder) / 'capture_progress.json').read_text())
            self.assertEqual(progress['stage'], 'capture_complete')

    def test_depth_stats_ignores_missing_pixels(self):
        depth = np.array([[0, 100], [200, 300]], dtype=np.uint16)
        stats = probe.depth_stats(depth)
        self.assertEqual(stats["valid_pixels"], 3)
        self.assertEqual(stats["total_pixels"], 4)
        self.assertEqual(stats["valid_ratio"], 0.75)
        self.assertEqual(stats["depth_mm_min"], 100)
        self.assertEqual(stats["depth_mm_max"], 300)

    def test_estimated_intrinsics_from_right_angle_fov(self):
        intrinsics = probe.estimated_intrinsics(640, 480, math.pi / 2, math.pi / 2)
        self.assertAlmostEqual(intrinsics["fx"], 320.0)
        self.assertAlmostEqual(intrinsics["fy"], 240.0)
        self.assertEqual(intrinsics["cx"], 319.5)
        self.assertEqual(intrinsics["cy"], 239.5)

    def test_empty_depth_preview_is_black(self):
        preview = probe.make_preview(np.zeros((3, 4), dtype=np.uint16))
        self.assertEqual(preview.shape, (3, 4, 3))
        self.assertFalse(np.any(preview))


if __name__ == "__main__":
    unittest.main()
