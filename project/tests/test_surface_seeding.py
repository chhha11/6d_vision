import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from core.surface_seeding import VisibleSurfaceSeeder


class SurfaceSeedingTest(unittest.TestCase):
    def test_first_visible_surface_reprojects_exactly_and_excludes_background(self):
        K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'cube.stl'
            mesh = o3d.geometry.TriangleMesh.create_box(100, 100, 100)
            mesh.translate([-50, -50, -50])
            mesh.compute_vertex_normals()
            self.assertTrue(o3d.io.write_triangle_mesh(str(model), mesh))
            seeder = VisibleSurfaceSeeder(model, K, np.zeros(5))
            pixels = np.float32([[320, 240], [330, 250], [0, 0]])
            points, valid = seeder.intersect(pixels, np.zeros(3), np.array([0, 0, 600.]))
            np.testing.assert_array_equal(valid, [True, True, False])
            np.testing.assert_allclose(points[:2, 2], -50., atol=1e-4)
            projected, _ = cv2.projectPoints(points[:2], np.zeros(3),
                                             np.array([0, 0, 600.]), K, np.zeros(5))
            np.testing.assert_allclose(projected.reshape(-1, 2), pixels[:2], atol=1e-5)
            self.assertTrue(np.isnan(points[2]).all())

    def test_seed_correspondences_do_not_move_a_stationary_pose(self):
        from core import cad_video_tracker as tracker

        K, dist, _ = tracker.load_calibration(tracker.DEFAULT_INTRINSICS)
        rvec, tvec = tracker.load_pose(tracker.DEFAULT_POSE)
        vertices, _, _ = tracker.make_edge_points(tracker.DEFAULT_MODEL, 700, 3)
        image = cv2.imdecode(np.fromfile(tracker.DEFAULT_STATIC_REFERENCE_IMAGE,
                                          np.uint8), cv2.IMREAD_COLOR)
        seeder = VisibleSurfaceSeeder(tracker.DEFAULT_MODEL, K, dist)
        points, pixels = seeder.seed(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                     vertices, rvec, tvec, 100)
        self.assertGreaterEqual(len(points), 12)
        ok, fitted_rvec, fitted_tvec, inliers = cv2.solvePnPRansac(
            points.astype(np.float32), pixels, K, dist, rvec.copy(), tvec.copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        self.assertTrue(ok)
        self.assertEqual(len(inliers), len(points))
        np.testing.assert_allclose(fitted_tvec.reshape(3), tvec, atol=.01)
        self.assertLess(tracker.axis_line_change_deg(rvec, fitted_rvec.reshape(3)), .01)
