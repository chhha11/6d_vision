import unittest
from copy import deepcopy

import numpy as np

from core import validate_rgb_depth_extrinsics as validator


class ValidateRgbDepthExtrinsicsTest(unittest.TestCase):
    def test_rays_from_matrix_uses_pixel_center(self):
        K = np.array([[100.0, 0.0, 10.0], [0.0, 200.0, 20.0], [0.0, 0.0, 1.0]])
        rays = validator.rays_from_matrix(np.array([[10.0, 20.0], [20.0, 40.0]]), K)
        np.testing.assert_allclose(rays, [[0, 0, 1], [0.1, 0.1, 1]])

    def test_intersect_plane_returns_metric_points(self):
        rays = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
        plane = np.array([0.0, 0.0, 1.0, -500.0])
        points = validator.intersect_plane(rays, plane)
        np.testing.assert_allclose(points, [[0, 0, 500], [50, 0, 500]])

    def test_adjacent_spacing_reports_checker_pitch(self):
        x, y = np.meshgrid(np.arange(9) * 25.0, np.arange(6) * 25.0)
        points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 600.0)))
        spacing = validator.adjacent_spacing(points)
        self.assertAlmostEqual(spacing["horizontal_median_mm"], 25.0)
        self.assertAlmostEqual(spacing["vertical_median_mm"], 25.0)
        self.assertAlmostEqual(spacing["combined_p95_abs_error_mm"], 0.0)

    def test_capture_contract_requires_saved_depth_model_and_unmirrored_streams(self):
        image = np.zeros((480, 640), dtype=np.uint16)
        K_depth = np.array([
            [577.8, 0.0, 319.5],
            [0.0, 577.8, 239.5],
            [0.0, 0.0, 1.0],
        ])
        ir_metadata = {
            "video_mode": {"width": 640, "height": 480},
            "ir_mirroring": {"active": False},
            "device": {"uri": "astra"},
        }
        depth_metadata = {
            "video_mode": {"width": 640, "height": 480, "depth_unit_mm": 1},
            "depth_mirroring": {"active": False},
            "device": {"uri": "astra"},
            "intrinsics_estimate": {
                "fx": 577.8, "fy": 577.8, "cx": 319.5, "cy": 239.5,
            },
        }
        validator.validate_capture_contract(
            image, image, image, ir_metadata, depth_metadata, (640, 480), K_depth)
        wrong_depth = deepcopy(depth_metadata)
        wrong_depth["intrinsics_estimate"]["fx"] = 590.0
        with self.assertRaisesRegex(RuntimeError, "differ from calibrated K_depth"):
            validator.validate_capture_contract(
                image, image, image, ir_metadata, wrong_depth, (640, 480), K_depth)
        mirrored_ir = deepcopy(ir_metadata)
        mirrored_ir["ir_mirroring"]["active"] = True
        with self.assertRaisesRegex(RuntimeError, "IR stream must be unmirrored"):
            validator.validate_capture_contract(
                image, image, image, mirrored_ir, depth_metadata,
                (640, 480), K_depth)


if __name__ == "__main__":
    unittest.main()
