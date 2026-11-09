import unittest

import numpy as np

from core import analyze_board_reference as board


class AnalyzeBoardReferenceTest(unittest.TestCase):
    def test_camera_motion_sign_and_frame(self):
        first_camera_from_board = board.matrix4(np.eye(3), np.array([0, 0, 500]))
        current_camera_from_board = board.matrix4(np.eye(3), np.array([-100, 20, 500]))
        current_board_from_camera = np.linalg.inv(current_camera_from_board)
        motion = board.camera_motion_in_first_frame(
            first_camera_from_board, current_board_from_camera)
        np.testing.assert_allclose(motion[:3, 3], [100, -20, 0], atol=1e-10)
        np.testing.assert_allclose(motion[:3, :3], np.eye(3), atol=1e-10)

    def test_corner_drift_uses_all_corners(self):
        before = np.zeros((54, 1, 2), dtype=np.float32)
        after = np.full((54, 1, 2), [0.3, 0.4], dtype=np.float32)
        drift = board.board_corner_drift(before, after)
        self.assertAlmostEqual(drift["median_px"], 0.5)
        self.assertAlmostEqual(drift["p95_px"], 0.5)


if __name__ == "__main__":
    unittest.main()
