import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import render_tracking_replay as replay


class ReplayBoxOffsetTest(unittest.TestCase):
    def test_decorate_frame_passes_selected_display_offset(self):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        row = {
            "status": "TRACKING", "source_elapsed_s": "1.0", "frame_index": "0",
            "rvec_x": "0", "rvec_y": "0", "rvec_z": "0",
            "tx_mm": "0", "ty_mm": "0", "tz_mm": "800",
        }
        corners = np.zeros((8, 3), dtype=np.float64)
        K = np.eye(3, dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        with patch.object(replay, "render_box_layer",
                          return_value=np.zeros_like(frame)) as draw_box:
            replay.decorate_frame(frame, row, corners, K, dist,
                                  side_by_side=False, box_display_y_offset_px=23)
        self.assertEqual(draw_box.call_args.args[-1], 23)

    def test_render_rejects_nonfinite_display_offset_before_opening_files(self):
        placeholder = Path("unused")
        with self.assertRaisesRegex(ValueError, "Box display Y offset"):
            replay.render(placeholder, placeholder, placeholder, placeholder,
                          placeholder, box_display_y_offset_px=float("nan"))

    def test_cli_uses_saved_offset_unless_explicitly_overridden(self):
        result = {"video": "unused.mp4", "written_frames": 1,
                  "size": (240, 160), "fps": 30.0, "keyframes": None}
        with patch.object(replay, "load_box_display_y_offset", return_value=23) as load_saved, \
                patch.object(replay, "render", return_value=result) as render:
            with patch("sys.argv", ["render_tracking_replay.py"]):
                replay.main()
            self.assertEqual(render.call_args.args[-1], 23)
            self.assertEqual(load_saved.call_count, 1)
            with patch("sys.argv", ["render_tracking_replay.py",
                                    "--box-display-y-offset-px", "7"]):
                replay.main()
            self.assertEqual(render.call_args.args[-1], 7.0)
            self.assertEqual(load_saved.call_count, 1)


if __name__ == "__main__":
    unittest.main()
