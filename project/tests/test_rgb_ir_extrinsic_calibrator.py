import tempfile
import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from core import rgb_ir_extrinsic_calibrator as calibrator


def synthetic_views():
    board = (9, 6)
    obj = calibrator.checkerboard_object_points(board, 25.0)
    K_ir = np.array([[575.0, 0.0, 319.5], [0.0, 576.0, 239.5], [0.0, 0.0, 1.0]])
    K_rgb = np.array([[530.0, 0.0, 316.0], [0.0, 531.0, 241.0], [0.0, 0.0, 1.0]])
    dist_ir = np.zeros((1, 5), dtype=np.float64)
    dist_rgb = np.zeros((1, 5), dtype=np.float64)
    relative_rvec = np.array([0.004, -0.012, 0.002], dtype=np.float64)
    R_rgb_ir, _ = cv2.Rodrigues(relative_rvec)
    t_rgb_ir = np.array([24.0, 0.8, 1.5], dtype=np.float64)
    objects, ir_points, rgb_points = [], [], []
    for index in range(18):
        rvec_ir = np.array([
            -0.22 + 0.025 * (index % 6),
            -0.18 + 0.07 * (index % 5),
            -0.20 + 0.08 * (index % 4),
        ], dtype=np.float64)
        tvec_ir = np.array([
            -115.0 + 22.0 * (index % 7),
            -85.0 + 20.0 * (index % 6),
            620.0 + 22.0 * index,
        ], dtype=np.float64)
        ir_projected, _ = cv2.projectPoints(obj, rvec_ir, tvec_ir, K_ir, dist_ir)
        R_ir_board, _ = cv2.Rodrigues(rvec_ir)
        R_rgb_board = R_rgb_ir @ R_ir_board
        t_rgb_board = R_rgb_ir @ tvec_ir + t_rgb_ir
        rgb_rvec, _ = cv2.Rodrigues(R_rgb_board)
        rgb_projected, _ = cv2.projectPoints(obj, rgb_rvec, t_rgb_board, K_rgb, dist_rgb)
        objects.append(obj.copy())
        ir_points.append(ir_projected.astype(np.float32))
        rgb_points.append(rgb_projected.astype(np.float32))
    truth = {
        "K_ir": K_ir,
        "K_rgb": K_rgb,
        "dist_rgb": dist_rgb,
        "R_rgb_ir": R_rgb_ir,
        "t_rgb_ir_mm": t_rgb_ir,
    }
    return objects, ir_points, rgb_points, truth


class RgbIrExtrinsicCalibratorTest(unittest.TestCase):
    def test_checkerboard_object_points_use_millimeters(self):
        points = calibrator.checkerboard_object_points((3, 2), 25.0)
        np.testing.assert_array_equal(
            points,
            [[0, 0, 0], [25, 0, 0], [50, 0, 0],
             [0, 25, 0], [25, 25, 0], [50, 25, 0]],
        )

    def test_align_corner_order_recovers_180_degree_reversal(self):
        rgb = np.arange(12, dtype=np.float32).reshape(6, 1, 2)
        ir = rgb[::-1].copy()
        aligned, order, scores = calibrator.align_corner_order(
            ir, rgb, (640, 480), (640, 480), (3, 2))
        np.testing.assert_array_equal(aligned, rgb)
        self.assertEqual(order, "rotated_180")
        self.assertLess(scores["rotated_180"], scores["original"])

    def test_discovers_pair_directories_and_reports_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete = root / "pair_0001"
            complete.mkdir()
            (complete / "rgb.png").touch()
            (complete / "ir_median_uint16.png").touch()
            incomplete = root / "pair_0002"
            incomplete.mkdir()
            (incomplete / "rgb.png").touch()
            pairs, missing = calibrator.discover_pairs(root)
            self.assertEqual(pairs, [
                ("pair_0001", complete / "rgb.png", complete / "ir_median_uint16.png")])
            self.assertEqual(missing[0]["name"], "pair_0002")
            self.assertIn("ir_median_uint16.png", missing[0]["reason"])

    def test_pair_manifest_rejects_mismatched_board_and_mirror(self):
        with tempfile.TemporaryDirectory() as folder:
            pair = Path(folder)
            manifest = {
                "accepted": True,
                "board_size": [9, 6],
                "square_size_mm": 25.0,
                "ir_metadata": {
                    "ir_mirroring": {"active": False},
                    "video_mode": {"width": 640, "height": 480},
                },
            }
            path = pair / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(calibrator.pair_capture_rejection(
                pair, (9, 6), 25.0, (640, 480)))
            self.assertIn("square size", calibrator.pair_capture_rejection(
                pair, (9, 6), 24.0, (640, 480)))
            manifest["ir_metadata"]["ir_mirroring"]["active"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIn("mirrored", calibrator.pair_capture_rejection(
                pair, (9, 6), 25.0, (640, 480)))

    def test_view_diversity_reports_position_and_tilt_span(self):
        centers = np.array([[0, 0, 600], [100, 80, 800]], dtype=np.float64)
        normals = np.array([[0, 0, 1], [0.5, 0, np.sqrt(0.75)]], dtype=np.float64)
        diversity = calibrator.summarize_view_diversity(centers, normals)
        np.testing.assert_allclose(diversity["board_center_span_ir_mm"], [100, 80, 200])
        self.assertAlmostEqual(diversity["max_board_normal_separation_deg"], 30.0)

    def test_synthetic_stereo_calibration_recovers_ir_to_rgb_transform(self):
        objects, ir_points, rgb_points, truth = synthetic_views()
        solved = calibrator.solve_calibration(
            objects, ir_points, rgb_points, (640, 480),
            truth["K_rgb"], truth["dist_rgb"])
        rotation_error = calibrator.rotation_angle_deg(
            solved["R_rgb_ir"] @ truth["R_rgb_ir"].T)
        translation_error = np.linalg.norm(
            solved["t_rgb_ir_mm"] - truth["t_rgb_ir_mm"])
        self.assertLess(rotation_error, 0.05)
        self.assertLess(translation_error, 0.5)
        self.assertLess(solved["stereo_rms_px"], 0.01)

    def test_fixed_depth_calibration_recovers_depth_to_rgb_transform(self):
        objects, depth_pixels, rgb_points, truth = synthetic_views()
        solved = calibrator.solve_depth_calibration(
            objects, depth_pixels, rgb_points, (640, 480),
            truth["K_ir"], truth["K_rgb"], truth["dist_rgb"])
        rotation_error = calibrator.rotation_angle_deg(
            solved["R_rgb_depth"] @ truth["R_rgb_ir"].T)
        translation_error = np.linalg.norm(
            solved["t_rgb_depth_mm"] - truth["t_rgb_ir_mm"])
        self.assertLess(rotation_error, 0.01)
        self.assertLess(translation_error, 0.1)
        self.assertLess(solved["depth_stereo_rms_px"], 0.01)

    def test_cross_reprojection_uses_ir_to_rgb_direction(self):
        objects, ir_points, rgb_points, truth = synthetic_views()
        solved = calibrator.solve_calibration(
            objects, ir_points, rgb_points, (640, 480),
            truth["K_rgb"], truth["dist_rgb"])
        detection = calibrator.PairDetection(
            "synthetic", objects[0], ir_points[0], rgb_points[0], "original")
        errors = calibrator.cross_reprojection_errors(detection, solved)
        self.assertLess(errors["ir_pose_to_rgb_rms_px"], 0.02)
        self.assertLess(errors["rgb_pose_to_ir_rms_px"], 0.02)


if __name__ == "__main__":
    unittest.main()
