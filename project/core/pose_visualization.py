"""Render CAD silhouettes and model axes without changing the measured pose."""
from __future__ import annotations

import cv2
import numpy as np


def display_edge_distance(frame):
    """Return a capped edge distance map shared by all targets in one frame."""
    image = np.asarray(frame)
    if image.ndim < 2:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(gray, 45, 110)
    if int(np.count_nonzero(edges)) < 24:
        return None
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def estimate_display_affine(frame, vertices, K, dist, rvec, tvec, seeder,
                             display_box=None, max_shift_px=20.,
                             min_improvement_px=.55, edge_distance=None,
                             max_rotation_deg=0., max_shear=0.,
                             allow_shape_adjustment=False):
    """Estimate a small image-space contour correction from current edges.

    This is deliberately a display-only operation. It searches only a bounded
    image translation by default and returns an affine matrix only when the
    current image edges give a clear improvement. Shape-changing corrections
    are disabled for live tracking because blur, occlusion, and background
    edges can otherwise make a correct CAD silhouette stretch or shear. The
    detector box narrows the search region but is not treated as the object's
    exact silhouette.

    ``allow_shape_adjustment`` is retained for offline diagnostics. The live
    multi-target renderer leaves it disabled so the metric 6DoF pose remains
    the sole source of contour shape.
    """
    image = np.asarray(frame)
    if image.ndim < 2:
        return None
    try:
        projected, _ = cv2.projectPoints(
            np.asarray(vertices, np.float64).reshape(-1, 3),
            np.asarray(rvec, np.float64).reshape(3),
            np.asarray(tvec, np.float64).reshape(3), K, dist)
        projected = projected.reshape(-1, 2)
        if not np.isfinite(projected).all() or np.max(np.abs(projected)) >= 1e6:
            return None
        mask = seeder.mask(image.shape, vertices, rvec, tvec, erode=False)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 8]
        if not contours:
            return None
        contour = np.concatenate(contours, axis=0)
        if len(contour) > 180:
            indices = np.linspace(0, len(contour) - 1, 180).astype(int)
            contour = contour[indices]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        distance = (display_edge_distance(image) if edge_distance is None
                    else np.asarray(edge_distance))
        if distance is None or distance.ndim != 2:
            return None
        h, w = gray.shape[:2]
        bound = np.r_[projected.min(axis=0), projected.max(axis=0)]
        if display_box is not None:
            box = np.asarray(display_box, np.float64).reshape(-1)
            if box.size == 4 and np.isfinite(box).all():
                bound = np.array([min(bound[0], box[0]), min(bound[1], box[1]),
                                 max(bound[2], box[2]), max(bound[3], box[3])])
        roi = np.array([max(0., bound[0] - 36.), max(0., bound[1] - 36.),
                        min(float(w), bound[2] + 36.), min(float(h), bound[3] + 36.)])
        center = (bound[:2] + bound[2:]) * .5
        valid_base = ((contour[:, 0] >= roi[0]) & (contour[:, 0] < roi[2]) &
                      (contour[:, 1] >= roi[1]) & (contour[:, 1] < roi[3]))
        contour = contour[valid_base]
        if len(contour) < 12:
            return None

        def score(matrix, shift_x, shift_y):
            points = (contour - center) @ matrix.T + center + [shift_x, shift_y]
            pixels = np.rint(points).astype(np.int32)
            valid = ((pixels[:, 0] >= roi[0]) & (pixels[:, 0] < roi[2]) &
                     (pixels[:, 1] >= roi[1]) & (pixels[:, 1] < roi[3]))
            if np.count_nonzero(valid) < 12:
                return 1e6
            # Cap distant background edges so a single unrelated edge cannot
            # win the correction.
            return float(np.mean(np.minimum(
                distance[pixels[valid, 1], pixels[valid, 0]], 8.)))

        base = score(np.eye(2, dtype=np.float32), 0., 0.)
        identity = np.eye(2, dtype=np.float32)
        best = (base, identity, 0., 0.)
        shifts = tuple(np.linspace(-max_shift_px, max_shift_px, 5))
        # Translation is safe for display alignment: it cannot change the
        # projected CAD proportions or create the large lobes seen during
        # motion. Keep the old shape search available only to explicitly
        # requested offline callers.
        matrices = (identity,)
        if allow_shape_adjustment:
            scales = (.98, 1., 1.02)
            matrices = tuple(np.array([[scale_x, 0.], [0., scale_y]], np.float32)
                             for scale_x in scales for scale_y in scales)
        for matrix in matrices:
            for shift_x in shifts:
                for shift_y in shifts:
                    candidate = score(matrix, shift_x, shift_y)
                    if candidate < best[0]:
                        best = (candidate, matrix, shift_x, shift_y)

        # Optional offline refinement. It is intentionally unreachable from
        # the live renderer unless a caller opts in explicitly.
        if allow_shape_adjustment and (max_rotation_deg or max_shear):
            _, base_matrix, base_shift_x, base_shift_y = best
            angles = np.deg2rad((-abs(max_rotation_deg),
                                 -abs(max_rotation_deg) / 2., 0.,
                                 abs(max_rotation_deg) / 2., abs(max_rotation_deg)))
            max_shear = float(max(0., min(abs(max_shear), .08)))
            shears = (-max_shear, 0., max_shear)
            local_shifts = (-5., 0., 5.)
            for angle in angles:
                c, s = np.cos(angle), np.sin(angle)
                rotation = np.array([[c, -s], [s, c]], np.float32)
                for shear_x in shears:
                    for shear_y in shears:
                        shear = np.array([[1., shear_x], [shear_y, 1.]], np.float32)
                        matrix = rotation @ shear @ base_matrix
                        determinant = float(np.linalg.det(matrix))
                        if not .92 <= determinant <= 1.08:
                            continue
                        for shift_x in local_shifts:
                            for shift_y in local_shifts:
                                candidate = score(matrix, base_shift_x + shift_x,
                                                  base_shift_y + shift_y)
                                if candidate < best[0]:
                                    best = (candidate, matrix,
                                            base_shift_x + shift_x,
                                            base_shift_y + shift_y)
        if (not np.isfinite(best[0]) or best[0] >= base - min_improvement_px
                or best[0] > 5.5):
            return None
        _, matrix, shift_x, shift_y = best
        offset = center + [shift_x, shift_y] - matrix @ center
        return np.column_stack((matrix, offset)).astype(np.float32)
    except (cv2.error, ValueError, TypeError):
        return None


def model_axis_points(vertices):
    """Model Z is the energy unit's long axis; origin is the CAD origin."""
    extent = np.ptp(np.asarray(vertices, np.float64), axis=0)
    lengths = extent * np.array([.4, .4, .5])
    return np.vstack((np.zeros(3), np.diag(lengths)))


def render_cad_pose_layer(frame_shape, vertices, K, dist, rvec, tvec, seeder,
                          display_y_offset_px=0., show_axes=True,
                          contour_color=(70, 255, 70), display_box=None,
                          display_affine=None,
                          display_box_max_scale_error=.25,
                          display_box_max_shift_px=24.):
    """Shift the entire display layer together, never its metric pose inputs.

    Axis colors follow the usual RGB convention: X red, Y green, Z blue.
    These are object/model axes, not the camera optical axes.
    """
    layer = np.zeros(frame_shape, np.uint8)
    rvec = np.asarray(rvec, np.float64).reshape(3)
    tvec = np.asarray(tvec, np.float64).reshape(3)
    if not np.isfinite(np.r_[rvec, tvec, display_y_offset_px]).all():
        return layer
    rotation, _ = cv2.Rodrigues(rvec)
    if np.min(np.asarray(vertices) @ rotation[2] + tvec[2]) <= 1.:
        return layer
    silhouette = seeder.mask(frame_shape, vertices, rvec, tvec, erode=False)
    contours, _ = cv2.findContours(silhouette, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(layer, contours, -1, contour_color, 2, cv2.LINE_AA)
    if show_axes:
        points = model_axis_points(vertices)
        pixels, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
        pixels = pixels.reshape(-1, 2)
        camera_points = points @ rotation.T + tvec
        if np.isfinite(pixels).all() and np.max(np.abs(pixels)) < 1e6:
            origin = tuple(np.rint(pixels[0]).astype(int))
            colors = ((40, 40, 255), (40, 220, 40), (255, 120, 40))
            for index, (label, color) in enumerate(zip('XYZ', colors), 1):
                if camera_points[index, 2] <= 1.:
                    continue
                tip = tuple(np.rint(pixels[index]).astype(int))
                if np.linalg.norm(pixels[index] - pixels[0]) >= 2.:
                    cv2.arrowedLine(layer, origin, tip, color, 3, cv2.LINE_AA,
                                    tipLength=.15)
                if 0 <= tip[0] < frame_shape[1] and 0 <= tip[1] < frame_shape[0]:
                    position = (tip[0] + 4, tip[1] - 5)
                    cv2.putText(layer, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                .6, (8, 8, 8), 4, cv2.LINE_AA)
                    cv2.putText(layer, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                                .6, color, 2, cv2.LINE_AA)

    # Apply an optional translation-only correction estimated from current
    # image edges. A display correction must never change the CAD silhouette's
    # shape; reject legacy or stale matrices containing scale, rotation, or
    # shear so a moving target cannot acquire a stretched contour.
    if display_affine is not None:
        affine = np.asarray(display_affine, np.float32)
        if affine.shape == (2, 3) and np.isfinite(affine).all():
            linear = affine[:, :2]
            if (np.max(np.abs(linear - np.eye(2, dtype=np.float32))) <= 1e-4
                    and np.max(np.abs(affine[:, 2])) <= 24.):
                layer = cv2.warpAffine(layer, affine,
                                       (frame_shape[1], frame_shape[0]),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT)
    if display_y_offset_px:
        layer = cv2.warpAffine(layer, np.array([[1., 0., 0.],
                                               [0., 1., display_y_offset_px]]),
                               (frame_shape[1], frame_shape[0]))
    return layer
