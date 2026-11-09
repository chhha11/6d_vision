"""Refresh measured multi-target tracks using visible CAD surface features."""
from dataclasses import replace

import cv2
import numpy as np


def replenish_pose(pose, frame, initializer, max_points=120):
    """Add points only at a freshly verified pose, for subsequent-frame tracking.

    Ray hits name the visible CAD surface; these are not another pose measurement
    or independent reference support. Keep existing measured correspondences and
    exclude nearby duplicates so newly seeded points cannot crowd them out.
    """
    seeder = getattr(initializer, 'tracking_seeder', None)
    if seeder is None or len(pose.points) >= max_points:
        return pose
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    points, pixels = seeder.seed(gray, initializer.vertices, pose.rvec, pose.tvec, max_points)
    if not len(pixels):
        return pose
    kept_points, kept_pixels = [], list(np.asarray(pose.pixels, np.float32))
    for point, pixel in zip(points, pixels):
        if not np.isfinite(point).all() or not np.isfinite(pixel).all():
            continue
        if kept_pixels and np.min(np.linalg.norm(np.asarray(kept_pixels) - pixel, axis=1)) < 5.:
            continue
        kept_points.append(point)
        kept_pixels.append(pixel)
        if len(kept_pixels) >= max_points:
            break
    if not kept_points:
        return pose
    return replace(pose, points=np.concatenate((pose.points, kept_points)),
                   pixels=np.asarray(kept_pixels, np.float32))


def needs_replenishment(pose, previous_pose, seeded_rvec, age):
    """Refresh newly exposed surfaces before rotation exhausts the old view."""
    # Horizontal yaw exposes and hides CAD surfaces quickly.  Refresh before
    # the old view is exhausted instead of waiting for a large point-count
    # drop, which lets PnP keep using correspondences from the newly visible
    # surface.
    if age >= 12 or len(pose.points) < 36:
        return True
    if previous_pose is not None and len(pose.points) < .7 * len(previous_pose.points):
        return True
    if seeded_rvec is not None:
        current, _ = cv2.Rodrigues(np.asarray(pose.rvec, np.float64))
        seeded, _ = cv2.Rodrigues(np.asarray(seeded_rvec, np.float64))
        cosine = np.clip((np.trace(current @ seeded.T) - 1.) / 2., -1., 1.)
        if cosine <= np.cos(np.deg2rad(5.)):
            return True
    return age >= 4 and len(pose.points) < 56


def advance_pose_trail(pose, frames, initializer, *, replenish=True):
    """Bring a verified background result through every captured intermediate frame."""
    from .rgb_global_initializer import advance_pose

    if replenish:
        pose = replenish_pose(pose, frames[0], initializer)
    seeded_rvec, age = pose.rvec.copy(), 0
    for source, current in zip(frames, frames[1:]):
        previous = pose
        pose = advance_pose(previous, source, current, initializer.K, initializer.dist,
                            initializer.vertices, None, retry_pnp=True)
        if pose is None:
            return None
        age += 1
        if replenish and needs_replenishment(pose, previous, seeded_rvec, age):
            pose = replenish_pose(pose, current, initializer)
            seeded_rvec, age = pose.rvec.copy(), 0
    return pose


def reference_improves_pose(current, reference, initializer):
    """Require independent current-image correspondences to demonstrate drift.

    Reference audits propagate only descriptor matches, without CAD reseeding.
    Compare both poses against that SAME evidence; old self-seeded inlier counts
    cannot establish accuracy, and a delayed result alone cannot replace a pose.
    """
    from .cad_video_tracker import project

    errors = []
    for pose in (current, reference):
        projected = project(reference.points, pose.rvec, pose.tvec,
                            initializer.K, initializer.dist)
        errors.append(np.linalg.norm(projected - reference.pixels, axis=1))
    if not all(np.isfinite(error).all() for error in errors) or len(errors[0]) < 12:
        return False
    old_median, new_median = [float(np.median(error)) for error in errors]
    return (np.sqrt(np.mean(errors[1] ** 2)) <= 2.5
            and old_median > 2.5 and old_median - new_median >= 1.)
