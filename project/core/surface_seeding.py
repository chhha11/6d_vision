"""Attach image features to the first visible CAD surface along camera rays."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=2)
def _surface_scene(path: str, mtime_ns: int, size: int):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise RuntimeError(f'Cannot read CAD mesh: {path}')
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    # Build the acceleration structure before the target enters the image.
    scene.cast_rays(o3d.core.Tensor([[0., 0., -1000., 0., 0., 1.]],
                                     dtype=o3d.core.Dtype.Float32))
    return scene


def load_surface_scene(model: Path):
    path = model.resolve()
    stat = path.stat()
    return _surface_scene(str(path), stat.st_mtime_ns, stat.st_size)


class VisibleSurfaceSeeder:
    def __init__(self, model: Path, K: np.ndarray, dist: np.ndarray):
        self.scene = load_surface_scene(model)
        self.K, self.dist = K, dist

    def intersect(self, pixels, rvec, tvec):
        import open3d as o3d

        pixels = np.asarray(pixels, np.float64).reshape(-1, 2)
        if not len(pixels):
            return np.empty((0, 3), np.float64), np.empty(0, bool)
        normalized = cv2.undistortPoints(pixels.reshape(-1, 1, 2), self.K,
                                         self.dist).reshape(-1, 2)
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3))
        origin = -np.asarray(tvec).reshape(3) @ rotation
        directions = np.column_stack((normalized, np.ones(len(pixels)))) @ rotation
        rays = np.column_stack((np.broadcast_to(origin, directions.shape), directions))
        distances = self.scene.cast_rays(
            o3d.core.Tensor(rays.astype(np.float32)))['t_hit'].numpy()
        valid = np.isfinite(distances) & (distances > 1.0)
        points = np.full((len(pixels), 3), np.nan)
        points[valid] = origin + directions[valid] * distances[valid, None]
        return points, valid

    def mask(self, shape, vertices, rvec, tvec, *, erode=True):
        projected, _ = cv2.projectPoints(vertices, np.asarray(rvec).reshape(3, 1),
                                         np.asarray(tvec).reshape(3, 1), self.K, self.dist)
        projected = projected.reshape(-1, 2)
        mask = np.zeros(shape[:2], np.uint8)
        if not np.isfinite(projected).all():
            return mask
        height, width = mask.shape
        left, top = np.maximum(np.floor(projected.min(axis=0)), 0).astype(int)
        right, bottom = np.minimum(np.ceil(projected.max(axis=0)) + 1,
                                    [width, height]).astype(int)
        if right <= left or bottom <= top:
            return mask
        rows, cols = np.mgrid[top:bottom, left:right]
        pixels = np.column_stack((cols.ravel(), rows.ravel()))
        _, valid = self.intersect(pixels, rvec, tvec)
        mask[top:bottom, left:right] = valid.reshape(rows.shape).astype(np.uint8) * 255
        # Exclude silhouette boundaries, gaps and the background behind rails.
        return cv2.erode(mask, np.ones((3, 3), np.uint8)) if erode else mask

    def seed(self, gray, vertices, rvec, tvec, max_points):
        mask = self.mask(gray.shape, vertices, rvec, tvec)
        features = cv2.goodFeaturesToTrack(
            gray, maxCorners=max_points, qualityLevel=.008, minDistance=5,
            mask=mask, blockSize=5, useHarrisDetector=False)
        if features is None:
            return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)
        pixels = features.reshape(-1, 2)
        points, valid = self.intersect(pixels, rvec, tvec)
        if np.count_nonzero(valid) < 8:
            return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)
        return points[valid], pixels[valid].astype(np.float32)
