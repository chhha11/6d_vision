import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from core.pose_visualization import model_axis_points, render_cad_pose_layer


class PoseVisualizationTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[x, y, z] for x in (-47.5, 47.5)
                                  for y in (-47.5, 47.5) for z in (-75., 75.)])
        self.K = np.array([[400., 0., 160.], [0., 400., 120.], [0., 0., 1.]])
        self.dist = np.zeros(5)
        self.rvec = np.array([.6, -.3, .1])
        self.tvec = np.array([0., 0., 600.])
        mask = np.zeros((240, 320), np.uint8)
        mask[85:155, 120:200] = 255
        self.seeder = SimpleNamespace(mask=lambda *a, **kw: mask.copy())

    def render(self, offset=0., **kwargs):
        return render_cad_pose_layer((240, 320, 3), self.vertices, self.K, self.dist,
                                     self.rvec, self.tvec, self.seeder,
                                     display_y_offset_px=offset, **kwargs)

    def test_z_arrow_follows_rotated_model_long_axis(self):
        with patch.object(cv2, 'arrowedLine', wraps=cv2.arrowedLine) as arrows:
            self.render()
        self.assertEqual(arrows.call_count, 3)
        expected, _ = cv2.projectPoints(np.array([[0., 0., 0.], [0., 0., 75.]]),
                                        self.rvec, self.tvec, self.K, self.dist)
        z_call = arrows.call_args_list[2]
        np.testing.assert_array_equal(z_call.args[1:3],
                                      np.rint(expected.reshape(-1, 2)).astype(int))
        np.testing.assert_array_equal(model_axis_points(self.vertices)[3], [0, 0, 75])

    def test_offset_shifts_contour_and_axes_together_without_changing_pose(self):
        before_r, before_t = self.rvec.copy(), self.tvec.copy()
        layer = self.render()
        shifted = self.render(12.)
        np.testing.assert_array_equal(layer[:-12], shifted[12:])
        self.assertFalse(shifted[:12].any())
        np.testing.assert_array_equal(self.rvec, before_r)
        np.testing.assert_array_equal(self.tvec, before_t)

    def test_shape_changing_display_correction_is_rejected(self):
        base = self.render(show_axes=False)
        legacy = self.render(
            show_axes=False,
            display_affine=np.array([[1.18, .07, -4.],
                                     [.03, .86, 9.]], np.float32))
        np.testing.assert_array_equal(legacy, base)

    def test_translation_display_correction_keeps_shape(self):
        base = self.render(show_axes=False)
        shifted = self.render(show_axes=False,
                              display_affine=np.array([[1., 0., 5.],
                                                       [0., 1., -3.]], np.float32))
        self.assertFalse(np.array_equal(shifted, base))
        base_y, base_x = np.where(np.any(base, axis=2))
        shifted_y, shifted_x = np.where(np.any(shifted, axis=2))
        self.assertEqual(int(shifted_x.min() - base_x.min()), 5)
        self.assertEqual(int(shifted_y.min() - base_y.min()), -3)
        self.assertEqual(int(shifted_x.max() - base_x.max()), 5)
        self.assertEqual(int(shifted_y.max() - base_y.max()), -3)

    def test_no_yellow_box_and_axes_can_be_disabled(self):
        with patch.object(cv2, 'arrowedLine', wraps=cv2.arrowedLine) as arrows:
            layer = self.render(show_axes=False)
        self.assertEqual(arrows.call_count, 0)
        self.assertTrue(layer[85, 160, 1] > 0)
        self.assertFalse(layer[120, 160].any())
        self.assertFalse(np.any(np.all(layer == [0, 255, 255], axis=2)))

    def test_invalid_or_behind_camera_pose_has_no_overlay(self):
        self.tvec[2] = 0.
        self.assertFalse(self.render().any())
        self.tvec[2] = np.nan
        self.assertFalse(self.render().any())


if __name__ == '__main__':
    unittest.main()
