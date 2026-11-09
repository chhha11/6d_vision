import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

from core import depth_global_initializer as initializer
from core.paths import DATA_ROOT


class DepthGlobalInitializerTest(unittest.TestCase):
    def test_rgb_constraint_masks_points_using_calibrated_depth_transform(self):
        K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
        transform = np.eye(4)
        transform[0, 3] = 100.
        constraint = initializer.RgbConstraint(np.array([310., 230, 330, 250]),
                                               K, np.zeros(5), transform)
        np.testing.assert_array_equal(constraint.contains(np.array([
            [-100., 0, 700], [0, 0, 700], [-100., 0, -700]])), [True, False, False])

    def test_near_partial_depth_requires_confirmed_rgb_region(self):
        sample = DATA_ROOT / 'tracking/live_depth_6dof/20261101_113247/attempt_009'
        settings = replace(initializer.Settings(), min_depth_mm=400.)
        with np.load(sample / 'rgb_depth_extrinsics.npz') as calibration:
            constraint = initializer.RgbConstraint(
                np.array([278., 138, 395, 337]), calibration['K_rgb'].copy(),
                calibration['dist_rgb'].copy(), calibration['T_rgb_depth'].copy())
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'No cluster matches'):
                initializer.run(sample / 'depth/depth_burst.npz', sample / 'depth/metadata.json',
                                 initializer.DEFAULT_MODEL, output / 'unconstrained', settings)
            diagnostics = initializer.run(
                sample / 'depth/depth_burst.npz', sample / 'depth/metadata.json',
                initializer.DEFAULT_MODEL, output / 'constrained', settings, constraint)
        self.assertTrue(diagnostics['pose_quality']['accepted'])
        self.assertTrue(diagnostics['pose_quality']['partial_depth'])
        self.assertTrue(diagnostics['pose_quality']['requires_rgb_verification'])
        np.testing.assert_array_equal(diagnostics['rgb_constraint']['box'], constraint.box)

    def test_nonzero_median_ignores_missing_depth(self):
        frames = np.array([
            [[0, 10], [0, 40]],
            [[100, 20], [0, 0]],
            [[200, 30], [0, 60]],
        ], dtype=np.uint16)
        actual = initializer.nonzero_median(frames)
        np.testing.assert_array_equal(actual, [[150, 20], [0, 50]])

    def test_depth_to_points_uses_camera_coordinates(self):
        depth = np.zeros((3, 4), dtype=np.uint16)
        depth[1, 2] = 1000
        intrinsics = initializer.Intrinsics(fx=100, fy=200, cx=1, cy=1)
        points, pixels = initializer.depth_to_points(depth, intrinsics)
        np.testing.assert_allclose(points, [[10, 0, 1000]])
        np.testing.assert_array_equal(pixels, [[2, 1]])

    def test_robust_pca_recovers_long_axis(self):
        x = np.linspace(-75, 75, 101)
        points = np.column_stack((x, 0.15 * np.sin(x), 0.1 * np.cos(x)))
        _, axes, extents, _ = initializer.robust_pca(points)
        self.assertGreater(abs(float(axes[0, 0])), 0.999)
        self.assertGreater(extents[0], 140)
        self.assertLess(extents[1], 1)

    def test_end_face_hypotheses_use_thin_pca_normal_as_long_axis(self):
        candidate = initializer.ClusterCandidate(
            0, 700, np.zeros(3), np.eye(3),
            np.array([96., 88., 22.]), np.array([100., 94., 25.]), 0., True)
        hypotheses = list(initializer.rotation_hypotheses(candidate, 120.))
        self.assertEqual(len(hypotheses), 12)
        self.assertEqual({item[3] for item in hypotheses}, {'side', 'end_face'})
        end_face = [item for item in hypotheses if item[3] == 'end_face']
        # With identity axes the cap normal is the camera z axis (model long
        # axis), while the old side hypothesis incorrectly uses camera x.
        self.assertTrue(all(np.allclose(np.abs(item[2][:, 2]), [0., 0., 1.])
                            for item in end_face))

    def test_cap_evidence_rejects_small_or_misplaced_patches(self):
        grid = np.linspace(-44., 44., 35)
        x, y = np.meshgrid(grid, grid)
        keep = x*x + y*y <= 47.5**2
        cap = np.column_stack((x[keep], y[keep], np.full(np.count_nonzero(keep), 72.)))
        pose = {'model_to_camera': np.eye(4).tolist(), 'axis_source': 'end_face'}
        self.assertTrue(initializer.end_face_evidence(cap, pose)['accepted'])
        self.assertFalse(initializer.end_face_evidence(cap * [.4, .4, 1.], pose)['accepted'])
        self.assertFalse(initializer.end_face_evidence(cap + [0., 0., -25.], pose)['accepted'])
        self.assertFalse(initializer.end_face_evidence(cap * [1.6, 1.6, 1.], pose)['accepted'])
        self.assertFalse(initializer.end_face_evidence(
            cap, {**pose, 'axis_source': 'side'})['accepted'])

    def test_end_face_saved_failures_fit_and_handoff_without_parallel_rails(self):
        from core import live_depth_6dof as live
        for attempt in (3, 15, 24):
            session = '20261101_162802' if attempt == 3 else '20261101_162307'
            sample = DATA_ROOT / 'tracking/live_depth_6dof' / session / f'attempt_{attempt:03d}'
            previous = json.loads((sample / 'depth_fit/diagnostics.json').read_text())
            self.assertFalse(previous['pose_quality']['accepted'])
            with np.load(sample / 'rgb_depth_extrinsics.npz') as calibration:
                constraint = initializer.RgbConstraint(
                    np.array(previous['rgb_constraint']['box']), calibration['K_rgb'].copy(),
                    calibration['dist_rgb'].copy(), calibration['T_rgb_depth'].copy())
            with self.subTest(attempt=attempt), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                diagnostics = initializer.run(
                    sample / 'depth/depth_burst.npz', sample / 'depth/metadata.json',
                    initializer.DEFAULT_MODEL, output / 'depth_fit',
                    replace(initializer.Settings(), min_depth_mm=400.), constraint)
                self.assertTrue(diagnostics['pose_quality']['accepted'])
                self.assertTrue(diagnostics['pose_quality']['end_face_geometry']['accepted'])
                self.assertEqual(diagnostics['pose']['axis_source'], 'end_face')
                self.assertGreaterEqual(diagnostics['pose']['fitness'], .98)
                self.assertLess(diagnostics['pose']['inlier_rmse_mm'], 2.)
                # This exercises real RGB extent, axis and stillness checks,
                # including an end-on view with no observable parallel rails.
                path = live.initialize_capture(
                    sample / 'rgb_before.png', sample / 'depth', output,
                    live.DEFAULT_MODEL, sample / 'rgb_depth_extrinsics.npz',
                    live.DEFAULT_INTRINSICS, live.Settings(), 12.,
                    sample / 'rgb_after.png', diagnostics)
                self.assertTrue(path.is_file())
                # A valid cap fit must still fail when it belongs to a
                # different RGB detection; the geometry flag is no bypass.
                diagnostics['rgb_constraint']['box'] = (
                    constraint.box + np.array([200., 0., 200., 0.])).tolist()
                with self.assertRaisesRegex(RuntimeError, 'partial geometry conflicts'):
                    live.initialize_capture(
                        sample / 'rgb_before.png', sample / 'depth', output / 'wrong_box',
                        live.DEFAULT_MODEL, sample / 'rgb_depth_extrinsics.npz',
                        live.DEFAULT_INTRINSICS, live.Settings(), 12.,
                        sample / 'rgb_after.png', diagnostics)

    def test_adjusted_capture_produces_valid_pose(self):
        session = DATA_ROOT / "tracking" / "depth_probe_adjusted_unmirrored"
        with tempfile.TemporaryDirectory() as folder:
            diagnostics = initializer.run(
                session / "depth_burst.npz",
                session / "metadata.json",
                DATA_ROOT / "models" / "energy_unit_centered_mm.stl",
                Path(folder),
                initializer.Settings(),
            )
            selected = diagnostics["selected_cluster"]
            pose = diagnostics["pose"]
            self.assertTrue(selected["accepted"])
            self.assertGreater(selected["point_count"], 700)
            self.assertGreater(selected["robust_extents_mm"][0], 140)
            self.assertLess(selected["robust_extents_mm"][0], 160)
            self.assertGreater(pose["fitness"], 0.95)
            self.assertLess(pose["inlier_rmse_mm"], 3.0)
            self.assertGreater(pose["translation_mm"][2], 680)
            self.assertLess(pose["translation_mm"][2], 750)
            self.assertTrue((Path(folder) / "diagnostics.json").exists())
            self.assertTrue((Path(folder) / "segmentation_preview.png").exists())
            self.assertTrue((Path(folder) / "registration_preview.png").exists())
            self.assertTrue(diagnostics["pose_quality"]["accepted"])
            self.assertFalse(diagnostics["pose_quality"]["long_axis_direction_confident"])
            self.assertFalse(diagnostics["pose_quality"]["fully_disambiguated"])
            self.assertTrue(diagnostics["pose_quality"]["roll_ambiguous"])

    def test_rotated_capture_recovers_split_target(self):
        session = DATA_ROOT / "tracking" / "global_pose_trials" / "trial_20261030_03" / "depth"
        with tempfile.TemporaryDirectory() as folder:
            diagnostics = initializer.run(
                session / "depth_burst.npz",
                session / "metadata.json",
                DATA_ROOT / "models" / "energy_unit_centered_mm.stl",
                Path(folder),
                initializer.Settings(),
            )
        self.assertEqual(diagnostics["cluster_eps_used_mm"], 20.0)
        self.assertTrue(diagnostics["selected_cluster"]["accepted"])
        self.assertGreater(diagnostics["selected_cluster"]["point_count"], 1000)
        self.assertTrue(diagnostics["pose_quality"]["accepted"])
        self.assertLess(diagnostics["pose"]["inlier_rmse_mm"], 3.0)

    def test_empty_negative_writes_rejection_diagnostics(self):
        sample = DATA_ROOT / "samples" / "global_dataset" / "N001" / "depth"
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "No cluster matches"):
                initializer.run(
                    sample / "depth_burst.npz", sample / "metadata.json",
                    DATA_ROOT / "models" / "energy_unit_centered_mm.stl",
                    output, initializer.Settings())
            diagnostics = json.loads(
                (output / "diagnostics.json").read_text(encoding="utf-8"))
        self.assertEqual(diagnostics["workspace_points"], 0)
        self.assertEqual(diagnostics["clusters"], [])
        self.assertFalse(diagnostics["pose_quality"]["accepted"])
        self.assertIsNone(diagnostics["pose"])


if __name__ == "__main__":
    unittest.main()
