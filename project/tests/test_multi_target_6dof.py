"""Metric PnP and cuboid projection checks, independent of camera acceptance."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.cad_video_tracker import BOX_SIGNS
from core.live_depth_6dof import parse_args
from core.multi_target_live import (Target, draw_targets, project_target_box,
                                   target_record, target_color)
from core.rgb_global_initializer import solve_correspondences


class MultiTarget6DofTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.vertices = BOX_SIGNS * [47.5, 47.5, 75.]
        self.K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.initializer = SimpleNamespace(vertices=self.vertices, K=self.K, dist=self.dist)
        def silhouette(shape, vertices, rvec, tvec, **kwargs):
            pixels, _ = cv2.projectPoints(vertices, rvec, tvec, self.K, self.dist)
            mask = np.zeros(shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(pixels).astype(np.int32)), 255)
            return mask
        self.seeder = SimpleNamespace(mask=silhouette)
        self.targets = []
        self.truth = []
        rng = np.random.default_rng(26)
        points = rng.uniform(-1., 1., (80, 3)) * [45., 45., 72.]
        for index in range(6):
            rvec = np.array([.25 + index*.08, -.3 + index*.1, .1*index])
            tvec = np.array([(index % 3 - 1)*250., (index//3 - .5)*250., 900. + 20.*index])
            projected, _ = cv2.projectPoints(self.vertices, rvec, tvec, self.K, self.dist)
            projected = projected.reshape(-1, 2)
            box = np.r_[projected.min(axis=0), projected.max(axis=0)]
            pixels, _ = cv2.projectPoints(points, rvec, tvec, self.K, self.dist)
            pose = solve_correspondences(points, pixels.reshape(-1, 2), self.K,
                self.dist, self.vertices, box, image_shape=self.frame.shape)
            self.assertIsNotNone(pose)
            self.targets.append(Target(index+1, box, 0., pose=pose))
            self.truth.append((rvec, tvec, projected))

    def test_six_solved_poses_project_their_own_metric_corners(self):
        for target, (rvec, tvec, projected) in zip(self.targets, self.truth):
            np.testing.assert_allclose(target.pose.tvec, tvec, atol=.001)
            rotation, _ = cv2.Rodrigues(rvec)
            actual_rotation, _ = cv2.Rodrigues(target.pose.rvec)
            np.testing.assert_allclose(actual_rotation, rotation, atol=1e-5)
            camera, pixels = project_target_box(target, self.initializer)
            np.testing.assert_allclose(camera, self.vertices @ rotation.T + tvec, atol=.001)
            np.testing.assert_allclose(pixels, projected, atol=.001)
            record = target_record(target, self.initializer)
            self.assertEqual(record['state'], 'pose_verified')
            np.testing.assert_allclose(record['box_corners_camera_mm'], camera)
            np.testing.assert_allclose(record['box_corners_image_px'], pixels)

    def test_six_contours_keep_id_colors_without_boxes_or_labels(self):
        with patch('core.multi_target_live.cv2.rectangle') as rectangles, \
                patch('core.multi_target_live.cv2.putText') as labels:
            canvas = draw_targets(self.frame, self.targets, self.initializer,
                                  show_axes=False, show_box=True, outline_seeder=self.seeder)
            reordered = draw_targets(self.frame, self.targets[::-1], self.initializer,
                                     show_axes=False, outline_seeder=self.seeder)
        rectangles.assert_not_called()
        labels.assert_not_called()
        np.testing.assert_array_equal(canvas, reordered)
        colors = {target_color(t.id) for t in self.targets}
        self.assertEqual(len(colors), 6)
        for color in colors:
            self.assertTrue(np.any(np.all(canvas == color, axis=2)))

    def test_contour_draw_deduplicates_broad_and_compact_tracks(self):
        compact = self.targets[0]
        width, height = compact.box[2:] - compact.box[:2]
        broad_box = compact.box + np.array([-1.25 * width, -1.25 * height,
                                             1.25 * width, 1.25 * height])
        broad = Target(99, broad_box, 0., pose=compact.pose)
        with patch('core.multi_target_live.render_cad_pose_layer',
                   return_value=np.zeros_like(self.frame)) as render:
            draw_targets(self.frame, [broad, compact], self.initializer,
                         show_axes=False, outline_seeder=self.seeder)
        render.assert_called_once()

    def test_contour_draw_deduplicates_similar_overlapping_verified_tracks(self):
        first = self.targets[0]
        width, height = first.box[2:] - first.box[:2]
        shifted_box = first.box + np.array([.18 * width, .08 * height,
                                             .18 * width, .08 * height])
        second = Target(99, shifted_box, 0., pose=first.pose)
        with patch('core.multi_target_live.render_cad_pose_layer',
                   return_value=np.zeros_like(self.frame)) as render:
            draw_targets(self.frame, [first, second], self.initializer,
                         show_axes=False, outline_seeder=self.seeder)
        render.assert_called_once()

    def test_loss_of_one_pose_hides_only_its_contour_without_fallback_box(self):
        before = draw_targets(self.frame, self.targets, self.initializer,
                              show_axes=False, outline_seeder=self.seeder)
        self.targets[2].pose = None
        with patch('core.multi_target_live.cv2.rectangle') as rectangles:
            after = draw_targets(self.frame, self.targets, self.initializer,
                                 show_axes=False, outline_seeder=self.seeder)
        rectangles.assert_not_called()
        self.assertFalse(np.any(np.all(after == target_color(self.targets[2].id), axis=2)))
        for target in self.targets:
            if target.pose is not None:
                mask = np.all(before == target_color(target.id), axis=2)
                np.testing.assert_array_equal(after[mask], before[mask])
        record = target_record(self.targets[2], self.initializer)
        self.assertIsNone(record['box_corners_camera_mm'])
        self.assertIsNone(record['box_corners_image_px'])
        self.assertIsNone(record['tvec_mm'])

    def test_rotating_one_target_changes_only_its_cuboid(self):
        before = [project_target_box(t, self.initializer)[1].copy() for t in self.targets]
        target = self.targets[0]
        target.pose = SimpleNamespace(rvec=target.pose.rvec + [0., 0., .6], tvec=target.pose.tvec)
        after = [project_target_box(t, self.initializer)[1] for t in self.targets]
        self.assertFalse(np.allclose(before[0], after[0]))
        for a, b in zip(before[1:], after[1:]):
            np.testing.assert_array_equal(a, b)

    def test_invalid_pose_never_draws_a_cuboid_or_axes(self):
        for tvec in (np.array([0., 0., -500.]), np.array([0., 0., np.nan])):
            target = self.targets[0]
            target.pose = SimpleNamespace(rvec=np.zeros(3), tvec=tvec)
            with patch('core.multi_target_live.cv2.line') as lines, \
                    patch('core.multi_target_live.cv2.drawFrameAxes') as axes:
                draw_targets(self.frame, [target], self.initializer)
            lines.assert_not_called()
            axes.assert_not_called()
            self.assertEqual(target_record(target, self.initializer)['state'], 'detected')

    def test_stationary_display_fit_requires_consensus(self):
        target = self.targets[0]
        candidate = np.array([[1., 0., 9.], [0., 1., -6.]], np.float32)
        target.tracked_frames = 0
        with patch('core.multi_target_live.estimate_display_affine',
                   return_value=candidate):
            draw_targets(self.frame, [target], self.initializer,
                         show_axes=False, outline_seeder=self.seeder)
        self.assertIsNone(target.display_affine)
        self.assertEqual(target.display_candidate_hits, 1)

        target.tracked_frames = 8
        with patch('core.multi_target_live.estimate_display_affine',
                   return_value=candidate):
            draw_targets(self.frame, [target], self.initializer,
                         show_axes=False, outline_seeder=self.seeder)
        np.testing.assert_allclose(target.display_affine, candidate)
        self.assertEqual(target.display_candidate_hits, 0)

    def test_stationary_outlier_does_not_move_locked_display_fit(self):
        target = self.targets[0]
        locked = np.array([[1., 0., 4.], [0., 1., -3.]], np.float32)
        outlier = np.array([[1., 0., 18.], [0., 1., 12.]], np.float32)
        target.display_affine = locked.copy()
        target.display_refit_at = 0
        target.tracked_frames = 24
        with patch('core.multi_target_live.estimate_display_affine',
                   return_value=outlier):
            draw_targets(self.frame, [target], self.initializer,
                         show_axes=False, outline_seeder=self.seeder)
        np.testing.assert_array_equal(target.display_affine, locked)
        self.assertEqual(target.display_candidate_hits, 1)

    def test_contour_defaults_preserve_axes_and_display_toggle(self):
        self.assertFalse(parse_args([]).show_box)
        self.assertFalse(parse_args(['--multi-target']).show_box)
        self.assertTrue(parse_args(['--multi-target']).cad_outline)
        self.assertTrue(parse_args(['--multi-target', '--show-box']).show_box)
        self.assertFalse(parse_args(['--multi-target', '--no-show-box']).show_box)
        with patch('core.pose_visualization.cv2.arrowedLine', wraps=cv2.arrowedLine) as axes:
            draw_targets(self.frame, self.targets, self.initializer, outline_seeder=self.seeder)
        self.assertEqual(axes.call_count, 18)
        with patch('core.pose_visualization.cv2.arrowedLine') as axes:
            draw_targets(self.frame, self.targets, self.initializer,
                         outline_seeder=self.seeder, show_axes=False)
        axes.assert_not_called()


if __name__ == '__main__':
    unittest.main()
