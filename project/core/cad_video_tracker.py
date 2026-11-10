"""CPU RGB CAD tracker using optical flow plus PnP.

The object must be placed approximately as in the saved initialization image
for the first frame. This is a lightweight validation tracker, not an
arbitrary-pose detector. Press R to reseed, S to save the current pose, and
Q/Esc to quit in live mode. Recorded RGB clips can be replayed with --video.
"""
from __future__ import annotations

from .collection_recording import collection_key, collection_display

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from .cad_pose_overlay import (DEFAULT_INTRINSICS, DEFAULT_MODEL, load_calibration,
                              load_edges, pose_vector)
from .reference_pose_corrector import ReferencePoseCorrector
from .rgb_pose_recovery import RgbPoseRecovery, axis_matches_pose, observe_long_axis

from .pose_visualization import render_cad_pose_layer
from .paths import DATA_ROOT

DEFAULT_POSE = DATA_ROOT / 'samples/energy_unit/frame_20261029_153630_557391_cad_overlay.json'
DEFAULT_STATIC_REFERENCE_IMAGE = DATA_ROOT / 'samples/energy_unit/frame_20261029_153630_557391.png'
DEFAULT_STATIC_REFERENCE_POSE = DATA_ROOT / 'samples/energy_unit/frame_20261029_153630_557391_pose_confirmed.json'

STATIC_MIN_READY_INLIERS = 12
STATIC_MIN_LOCK_INLIERS = 12
STATIC_MIN_MATCH_INLIERS = 10
STATIC_MIN_INLIER_RATIO = .45
STATIC_MIN_SPREAD_PX2 = 1000.0
STATIC_MAX_CENTER_ERROR_PX = 80.0
STATIC_MAX_ANGLE_ERROR_DEG = 10.0
STATIC_MAX_SCALE_ERROR = .35
STATIC_MAX_LOCK_DISPLACEMENT_PX = 1.5
DEFAULT_DETECTOR_INTERVAL = 12
# A hand-held object can lose a few optical-flow points for a frame or two
# because of motion blur or a specular highlight. Keep that transient loss
# recoverable before invalidating the 3D pose and waiting for RGB relocalization.
MAX_SOFT_LOSS_FRAMES = 3


def load_pose(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if 'depth_pose_quality' in data and not data['depth_pose_quality'].get('accepted', False):
        raise RuntimeError(f'Depth initialization pose is not accepted: {path}')
    if 'rgb_pose_quality' in data and not data['rgb_pose_quality'].get('accepted', False):
        raise RuntimeError(f'RGB initialization pose is not accepted: {path}')
    if 'rvec_rgb_model_rad' in data:
        rvec = np.asarray(data['rvec_rgb_model_rad'], dtype=np.float64).reshape(3)
    elif 'rvec' in data:
        rvec = np.asarray(data['rvec'], dtype=np.float64).reshape(3)
    else:
        args = argparse.Namespace(rx=float(data['rvec_deg'][0]), ry=float(data['rvec_deg'][1]),
                                  rz=float(data['rvec_deg'][2]), tx=0, ty=0, tz=1)
        rvec, _ = pose_vector(args)
    if 'tvec_rgb_model_mm' in data:
        tvec = np.asarray(data['tvec_rgb_model_mm'], dtype=np.float64).reshape(3)
    elif 'tvec_mm' in data:
        tvec = np.asarray(data['tvec_mm'], dtype=np.float64).reshape(3)
    else:
        tvec = np.asarray(data['tvec'], dtype=np.float64).reshape(3)
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all() or tvec[2] <= 0:
        raise RuntimeError(f'Invalid finite positive-depth pose: {path}')
    return rvec, tvec


def load_video_timestamps(video: Path):
    metadata_path = video.with_name('metadata.json')
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    values = metadata.get('frame_timestamps_s')
    if values is None:
        return None
    timestamps = np.asarray(values, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all() or \
            timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(f'Invalid frame timestamps: {metadata_path}')
    return timestamps


def load_session_depth_bounds(video: Path, margin_mm: float):
    """Use accepted RGB-D anchors to bound drift in a paired motion clip."""
    manifest_path = video.parent.parent.parent / 'session_manifest.json'
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    recorded_video = manifest.get('motion_rgb', {}).get('video')
    if not recorded_video or (manifest_path.parent / recorded_video).resolve() != video.resolve():
        return None
    anchors = [manifest.get('start_rgb_d', {}), manifest.get('end_rgb_d', {})]
    if not all(anchor.get('accepted') and anchor.get('depth_qc_passed') for anchor in anchors):
        return None
    distances = np.asarray([anchor.get('nominal_distance_mm') for anchor in anchors],
                           dtype=np.float64)
    if not np.isfinite(distances).all() or np.any(distances <= 0):
        raise RuntimeError(f'Invalid anchor distances: {manifest_path}')
    reference_image = (manifest_path.parent / Path(anchors[0]['manifest']).parent / 'rgb.png')
    return dict(manifest=str(manifest_path.resolve()),
                session_id=manifest.get('session_id'),
                reference_image=str(reference_image.resolve()),
                start_nominal_mm=float(distances[0]), end_nominal_mm=float(distances[1]),
                min_mm=float(distances.min() - margin_mm),
                max_mm=float(distances.max() + margin_mm),
                margin_mm=float(margin_mm))


@lru_cache(maxsize=4)
def make_edge_points(model: Path, max_edges: int, samples_per_edge: int = 3):
    vertices, edges = load_edges(model, max_edges)
    # The STL contains hundreds of thousands of duplicated triangle vertices.
    # Keep only vertices referenced by the sampled edges so projection stays cheap.
    referenced, remapped = np.unique(edges.reshape(-1), return_inverse=True)
    vertices = vertices[referenced]
    edges = remapped.reshape(-1, 2).astype(np.int32)
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    ts = np.linspace(0.0, 1.0, max(samples_per_edge, 2), dtype=np.float64)[:, None, None]
    points = (a[None] * (1 - ts) + b[None] * ts).reshape(-1, 3)
    points = np.unique(np.round(points, decimals=6), axis=0)
    return vertices, edges, points


def project(points, rvec, tvec, K, dist):
    uv, _ = cv2.projectPoints(points, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


def image_roi(projected, image_shape, margin, extra_points=None):
    """Return a clipped x1, y1, x2, y2 ROI around finite image points."""
    h, w = image_shape[:2]
    groups = [np.asarray(projected).reshape(-1, 2)]
    if extra_points is not None and len(extra_points):
        groups.append(np.asarray(extra_points).reshape(-1, 2))
    points = np.concatenate(groups, axis=0)
    valid = np.isfinite(points).all(axis=1) & (np.abs(points) < 1e6).all(axis=1)
    if not valid.any():
        return 0, 0, w, h
    points = points[valid]
    x1 = max(0, int(np.floor(points[:, 0].min() - margin)))
    y1 = max(0, int(np.floor(points[:, 1].min() - margin)))
    x2 = min(w, int(np.ceil(points[:, 0].max() + margin + 1)))
    y2 = min(h, int(np.ceil(points[:, 1].max() + margin + 1)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return 0, 0, w, h
    return x1, y1, x2, y2


def projected_segments(projected, edges):
    segments = projected[edges]
    valid = np.isfinite(segments).all(axis=(1, 2)) & (np.abs(segments) < 1e6).all(axis=(1, 2))
    return np.rint(segments[valid]).astype(np.int32)


def projected_coverage(vertices, rvec, tvec, K, dist, image_shape):
    """Fraction of the projected CAD convex hull inside the image rectangle."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    camera_z = (vertices @ rotation[2]) + float(tvec[2])
    if not np.isfinite(camera_z).all() or np.any(camera_z <= 1.0):
        return 0.0
    projected = project(vertices, rvec, tvec, K, dist).astype(np.float32)
    if not np.isfinite(projected).all():
        return 0.0
    hull = cv2.convexHull(projected)
    area = float(cv2.contourArea(hull))
    if area <= 1.0:
        return 0.0
    h, w = image_shape[:2]
    rectangle = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    visible, _ = cv2.intersectConvexConvex(hull, rectangle)
    return float(np.clip(visible / area, 0.0, 1.0))


def seed_points(gray, vertices, edges, model_points, rvec, tvec, K, dist, max_points,
                seed_radius=8.0, roi_margin=20, surface_seeder=None):
    """Associate strong image corners with nearby projected CAD edge samples."""
    if surface_seeder is not None:
        return surface_seeder.seed(gray, vertices, rvec, tvec, max_points)
    h, w = gray.shape
    projected = project(model_points, rvec, tvec, K, dist)
    projected_vertices = project(vertices, rvec, tvec, K, dist)
    x1, y1, x2, y2 = image_roi(projected_vertices, gray.shape, roi_margin)
    local_segments = projected_segments(projected_vertices - np.array([x1, y1]), edges)
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    if len(local_segments):
        cv2.polylines(mask, list(local_segments), False, 255,
                      max(3, int(round(seed_radius * 2 + 1))), cv2.LINE_8)
    max_corners = max(max_points * 4, 200)
    features = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=max_corners,
                                       qualityLevel=.008, minDistance=5, mask=mask,
                                       blockSize=5, useHarrisDetector=False)
    if features is None:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)
    features = features.reshape(-1, 2).astype(np.float32)
    features += np.array([x1, y1], np.float32)
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 0] >= 0) & (projected[:, 0] < w) & \
            (projected[:, 1] >= 0) & (projected[:, 1] < h)
    projected_valid = projected[valid]
    model_valid = model_points[valid]
    if len(projected_valid) == 0:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)
    # BFMatcher performs the 2-D nearest-neighbour search in native code and
    # avoids allocating the former samples x corners distance tensor.
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = sorted(matcher.match(features, projected_valid.astype(np.float32)),
                     key=lambda match: match.distance)
    used_model = set()
    model_out, image_out = [], []
    for match in matches:
        if match.distance > seed_radius:
            break
        model_index = int(match.trainIdx)
        if model_index in used_model:
            continue
        used_model.add(model_index)
        model_out.append(model_valid[model_index])
        image_out.append(features[match.queryIdx])
        if len(model_out) >= max_points:
            break
    if len(model_out) < 8:
        return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)
    return np.asarray(model_out, np.float64), np.asarray(image_out, np.float32)


BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)
BOX_SIGNS = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)
DISPLAY_POSE_STATES = {'TRACKING', 'STATIC LOCKED', 'STATIC CORRECTED',
                       'RGB RECOVERED', 'POSE CHECK', 'POSE CORRECTED',
                       'POSE RELOCALIZED', 'REGION TRACKED'}
VISIBLE_POSE_STATES = DISPLAY_POSE_STATES | {'TRACKING HOLD'}
# Display-only correction for the CAD overlay. This is deliberately kept
# separate from metric rvec/tvec, which are passed to the robot unchanged.
BOX_DISPLAY_Y_OFFSET_PX = 8.0
BOX_DISPLAY_SETTINGS_PATH = DATA_ROOT / 'tracking' / 'box_display_settings.json'
REACQUIRE_EXIT_CODE = 2


class PoseDisplayState:
    """Keep brief maintenance failures from flashing the box on and off.

    Only measured poses renew the grace period. Held poses never become valid
    output, and confirmed loss hides them immediately.
    """

    HARD_LOSS = {'TARGET LOST', 'OUT OF VIEW', 'IDENTITY MISMATCH',
                 'REACQUIRE DEPTH', 'REACQUIRING'}

    def __init__(self, grace_s=.25, max_hold_frames=6):
        self.grace_s = grace_s
        self.max_hold_frames = max_hold_frames
        self.last_time = None
        self.last_pose = None
        self.held_frames = 0
        self.recovery_frames = 0
        self.state = 'INITIALIZING'

    def update(self, status, now, rvec, tvec):
        valid = make_xyz_record(0, 0, status, rvec, tvec)['valid']
        if valid:
            self.last_time = now
            self.last_pose = (rvec.copy(), tvec.copy())
            self.held_frames = 0
            self.recovery_frames += 1
            # Once recovery was visible, require two measured frames to leave
            # it. Initial acquisition still appears immediately.
            if self.state != 'RECOVERING' or self.recovery_frames >= 2:
                self.state = 'TRACKING'
            return self.state, self.last_pose if self.state == 'TRACKING' else None, False
        self.recovery_frames = 0
        self.held_frames += 1
        recent = (self.last_time is not None and now - self.last_time <= self.grace_s
                  and self.held_frames <= self.max_hold_frames)
        if status in self.HARD_LOSS:
            self.last_pose = None
            self.last_time = None
            self.state = 'SEARCHING'
        elif recent and self.state == 'TRACKING':
            return self.state, self.last_pose, True
        else:
            self.state = 'RECOVERING' if self.last_pose is not None else 'INITIALIZING'
        return self.state, None, False


class PoseVerification:
    """Latch repeated detector conflicts until two positive checks agree."""

    def __init__(self):
        self.conflicts = 0
        self.matches = 0
        self.latched = False

    def update(self, matched):
        if matched is None:  # Missing or stale evidence is inconclusive.
            return
        if matched:
            self.matches += 1
            if not self.latched or self.matches >= 2:
                self.conflicts = 0
                self.latched = False
        else:
            self.matches = 0
            self.conflicts += 1
            self.latched = self.latched or self.conflicts >= 2

    def reset(self):
        self.conflicts = self.matches = 0
        self.latched = False


def detection_matches_pose(box, vertices, rvec, tvec, K, dist, image_shape=None):
    """Check a detector box against the current CAD projection."""
    projected = project(vertices, rvec, tvec, K, dist)
    if not np.isfinite(projected).all():
        return False
    predicted = np.r_[projected.min(axis=0), projected.max(axis=0)]
    observed = np.asarray(box, dtype=np.float64).reshape(4)
    if image_shape is not None:
        height, width = image_shape[:2]
        # Compare the detector's visible box to the visible CAD projection.
        # Clipping preserves the same aspect, scale and center checks while
        # allowing a real object to enter through any image edge.
        predicted = np.clip(predicted, [0., 0., 0., 0.],
                             [width - 1., height - 1., width - 1., height - 1.])
    intersection = np.maximum(0, np.minimum(predicted[2:], observed[2:]) -
                              np.maximum(predicted[:2], observed[:2]))
    overlap = float(np.prod(intersection))
    predicted_area = float(np.prod(np.maximum(predicted[2:] - predicted[:2], 0)))
    observed_area = float(np.prod(np.maximum(observed[2:] - observed[:2], 0)))
    union = predicted_area + observed_area - overlap
    if union <= 0 or overlap / union < .18:
        return False
    predicted_size = np.maximum(predicted[2:] - predicted[:2], 1.0)
    observed_size = np.maximum(observed[2:] - observed[:2], 1.0)
    # Overlap alone accepts a vertical CAD box crossing a horizontal target.
    aspect_change = (observed_size[0] / observed_size[1]) / (
        predicted_size[0] / predicted_size[1])
    scale_change = np.sqrt(observed_area / max(predicted_area, 1.0))
    center_error = np.linalg.norm((observed[:2] + observed[2:] -
                                   predicted[:2] - predicted[2:]) * .5)
    return bool(.65 <= aspect_change <= 1.55 and .75 <= scale_change <= 1.33
                and center_error <= max(12.0, float(np.linalg.norm(observed_size)) * .20))


def detection_associates_pose(box, vertices, rvec, tvec, K, dist, image_shape):
    """A hit on another object is inconclusive about this tracked identity.

    Use the same loose IoU association as detector candidate selection, before
    applying the stricter scale, center and axis checks. This is not positive
    pose evidence and must never clear an existing verification conflict.
    """
    if box is None:
        return False
    projected = project(vertices, rvec, tvec, K, dist)
    if not np.isfinite(projected).all():
        return False
    height, width = image_shape[:2]
    predicted = np.clip(np.r_[projected.min(axis=0), projected.max(axis=0)],
                        [0., 0., 0., 0.],
                        [width - 1., height - 1., width - 1., height - 1.])
    return box_overlap(predicted, box)[0] >= .1


def current_detection_agrees(frame, box, vertices, rvec, tvec, K, dist):
    """Only a spatially associated hit can contradict the tracked pose."""
    return not detection_associates_pose(
        box, vertices, rvec, tvec, K, dist, frame.shape) or (detection_matches_pose(
        box, vertices, rvec, tvec, K, dist, image_shape=frame.shape) and
        axis_matches_pose(observe_long_axis(frame, box), rvec, tvec, K, dist))


def tracking_support_quality(model_points, image_points, vertices, rvec, tvec,
                             K, dist):
    """Reject a low-RMS pose supported by a narrow or locally ambiguous patch.

    Optical flow can keep a repeated edge internally consistent after it has
    left the object.  Reprojection RMS alone cannot detect that failure.  The
    accepted points must therefore occupy a meaningful 2-D part of the
    projected object, and the tracked model points must not collapse onto a
    single CAD line.
    """
    model_points = np.asarray(model_points, np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, np.float64).reshape(-1, 2)
    if len(model_points) < 8 or len(model_points) != len(image_points):
        return False
    if not np.isfinite(model_points).all() or not np.isfinite(image_points).all():
        return False
    singular = np.linalg.svd(model_points - model_points.mean(axis=0), compute_uv=False)
    if len(singular) < 2 or singular[1] / max(singular[0], 1e-6) < .035:
        return False
    projected = project(vertices, rvec, tvec, K, dist)
    if not np.isfinite(projected).all():
        return False
    projected_box = np.r_[projected.min(axis=0), projected.max(axis=0)]
    observed_box = np.r_[image_points.min(axis=0), image_points.max(axis=0)]
    projected_size = np.maximum(projected_box[2:] - projected_box[:2], 1.)
    observed_size = np.maximum(observed_box[2:] - observed_box[:2], 1.)
    # At least one substantial span is required; the second span may be small
    # for a strongly foreshortened view, but not effectively zero.
    span = observed_size / projected_size
    if np.max(span) < .16 or np.min(span) < .025:
        return False
    projected_area = float(np.prod(projected_size))
    observed_hull = cv2.contourArea(cv2.convexHull(image_points.astype(np.float32)))
    if observed_hull < max(80., projected_area * .012):
        return False
    projected_center = (projected_box[:2] + projected_box[2:]) * .5
    observed_center = (observed_box[:2] + observed_box[2:]) * .5
    # A visible face can move the feature centroid away from the CAD box
    # center, but a displacement of half the whole projection is suspicious.
    if np.linalg.norm(observed_center - projected_center) > max(28.,
                                                                  .48 * np.linalg.norm(projected_size)):
        return False
    return True


def pose_box_motion_consistent(previous_rvec, previous_tvec, candidate_rvec,
                               candidate_tvec, box_corners, K, dist, flow_vector):
    """Check that a PnP update agrees with the measured image motion."""
    previous = project(box_corners, previous_rvec, previous_tvec, K, dist)
    candidate = project(box_corners, candidate_rvec, candidate_tvec, K, dist)
    if not np.isfinite(previous).all() or not np.isfinite(candidate).all():
        return False
    previous_box = np.r_[previous.min(axis=0), previous.max(axis=0)]
    candidate_box = np.r_[candidate.min(axis=0), candidate.max(axis=0)]
    previous_size = np.maximum(previous_box[2:] - previous_box[:2], 1.)
    candidate_size = np.maximum(candidate_box[2:] - candidate_box[:2], 1.)
    scale = candidate_size / previous_size
    if np.any(scale < .60) or np.any(scale > 1.67):
        return False
    actual_shift = ((candidate_box[:2] + candidate_box[2:]) -
                    (previous_box[:2] + previous_box[2:])) * .5
    measured = np.asarray(flow_vector, np.float64).reshape(2)
    if not np.isfinite(measured).all():
        return False
    return bool(np.linalg.norm(actual_shift - measured) <=
                max(24., .42 * np.linalg.norm(previous_size)))


def box_overlap(first, second):
    first, second = np.asarray(first), np.asarray(second)
    overlap = float(np.prod(np.maximum(0, np.minimum(first[2:], second[2:]) -
                                       np.maximum(first[:2], second[:2]))))
    first_area = float(np.prod(np.maximum(0, first[2:] - first[:2])))
    second_area = float(np.prod(np.maximum(0, second[2:] - second[:2])))
    return overlap / max(first_area + second_area - overlap, 1.), overlap / max(first_area, 1.)


def detect_target_box(detector, frame, fallback_rotation=0, expected_box=None,
                      fast=False):
    if hasattr(detector, 'get_box'):
        return detector.get_box(frame, fallback_rotation=fallback_rotation,
                                expected_box=expected_box, fast=fast)
    for turn in ((0, fallback_rotation) if fallback_rotation else (0,)):
        source = np.ascontiguousarray(np.rot90(frame, turn)) if turn else frame
        kwargs = dict(conf=.20, verbose=False)
        if fast:
            kwargs['imgsz'] = 512
        result = detector.predict(source, **kwargs)[0]
        if len(result.boxes) == 0:
            continue
        boxes = [restore_rotated_box(box, frame.shape, turn)
                 for box in result.boxes.xyxy.cpu().numpy()]
        index = int(result.boxes.conf.argmax().item())
        if expected_box is not None:
            overlaps = [box_overlap(expected_box, box)[0] for box in boxes]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= .1:
                index = best
        return boxes[index]
    return None


def restore_rotated_box(box, frame_shape, quarter_turns):
    """Map a box in np.rot90(frame, k) back into the camera image."""
    x1, y1, x2, y2 = np.asarray(box, np.float64).reshape(4)
    height, width = frame_shape[:2]
    turn = quarter_turns % 4
    if turn == 1:
        return np.array([width - y2, x1, width - y1, x2], np.float32)
    if turn == 2:
        return np.array([width - x2, height - y2, width - x1, height - y1], np.float32)
    if turn == 3:
        return np.array([y1, height - x2, y2, height - x1], np.float32)
    return np.asarray(box, np.float32).copy()


def check_target(frame, detector, recovery, fallback_rotation, expected_box=None,
                 pose_search=None, fallback_search_box=None, fast=False,
                 global_search=False, prefer_roi=False):
    """Search 3D independently; consume optional background YOLO hints."""
    background = hasattr(detector, 'get_box')
    cached_box = (detector.get_box(frame, fallback_rotation=fallback_rotation)
                  if background else None)
    if global_search and pose_search is not None:
        height, width = frame.shape[:2]
        full_box = np.array([0., 0., float(width), float(height)], np.float32)
        # Try the detector hint first when available, but search the full frame
        # in this same worker call if it fails. The full-frame candidate must
        # not inherit a stale detector box as its validation constraint.
        regions = [(full_box, None)]
        if prefer_roi and cached_box is not None:
            regions.insert(0, (cached_box, cached_box))
        for search_box, validation_box in regions:
            try:
                pose = pose_search.estimate(frame, search_box, validation_box=validation_box)
            except TypeError as exc:
                if 'validation_box' not in str(exc):
                    raise
                pose = pose_search.estimate(frame, search_box)
            if pose is not None:
                # The final value is the source-frame validation box used by
                # advance_pose, not the region searched for features. Global
                # search has no detector constraint; using full_box here would
                # reject the handoff because the object cannot fill the image.
                return (validation_box, None, observe_long_axis(frame, validation_box),
                        pose, validation_box)
    box = (cached_box if background else
           detect_target_box(detector, frame, fallback_rotation, expected_box, fast=fast)
           if detector is not None else None)
    try:
        observation = recovery.observe(frame) if recovery is not None else None
    except cv2.error:
        observation = None
    result = (box, observation, observe_long_axis(frame, box))
    if pose_search is None:
        return result
    if global_search:
        # All requested regions were tried above. Keep the detector observation
        # for loss verification, without repeating the expensive global search.
        return (*result, None, full_box)
    search_box = box
    if search_box is None and recovery is not None:
        search_box = recovery.observation_box(observation, frame.shape)
    if search_box is None:
        search_box = fallback_search_box
    validation_box = box
    try:
        pose = pose_search.estimate(frame, search_box, validation_box=validation_box)
    except TypeError as exc:
        # Keep the small search interface used by older/custom providers.
        if 'validation_box' not in str(exc):
            raise
        pose = pose_search.estimate(frame, search_box)
    return (*result, pose, search_box)


def target_check_interval(base, status, flow_median, flow_p90, conflicts=0,
                          recovery_pending=False, pose_pending=False, needs_seed=False):
    """Check promptly during motion/loss without running inference every stable frame."""
    base = max(int(base), 1)
    if (status not in DISPLAY_POSE_STATES or conflicts or recovery_pending
            or pose_pending or needs_seed):
        return min(base, 2)
    # Even slow object/camera motion needs a fresh metric check. A stale
    # background cadence is visible as a lagging CAD contour before optical
    # flow becomes large enough to trigger the old threshold.
    if flow_median > 1.0 or flow_p90 > 3.0:
        return min(base, 4)
    return base


def load_box_display_y_offset(path: Path = BOX_DISPLAY_SETTINGS_PATH) -> float:
    if not path.is_file():
        return BOX_DISPLAY_Y_OFFSET_PX
    settings = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(settings, dict):
        raise ValueError(f'Invalid box display settings in {path}')
    value = settings.get('display_y_offset_px')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not np.isfinite(value) or abs(value) > 100:
        raise ValueError(f'Invalid box display Y offset in {path}')
    return float(value)


@lru_cache(maxsize=4)
def load_model_box(model: Path):
    import trimesh

    mesh = trimesh.load(model, force='mesh', process=False)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or \
            np.any(bounds[1] <= bounds[0]):
        raise RuntimeError(f'Invalid CAD model bounds: {model}')
    lower, upper = bounds
    return np.array([
        [lower[0], lower[1], lower[2]],
        [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]],
        [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]],
        [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]],
        [lower[0], upper[1], upper[2]],
    ], dtype=np.float64)


def roll_neutral_box_points(box_corners, rvec, tvec):
    """Fix cross-section orientation to the viewing direction, ignoring roll."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    axis = rotation[:, 2]
    # One transverse edge always lies in the image plane, perpendicular to
    # the projected long axis. Unlike screen-up projection, this has no
    # reference-axis switch when the object goes from lying down to upright.
    horizontal = np.cross(axis, np.array([0., 0., 1.]))
    length = float(np.linalg.norm(horizontal))
    if length < 1e-6:  # Long axis points into the camera; screen-right is fixed.
        horizontal = np.array([1., 0., 0.])
    else:
        horizontal /= length
    vertical = np.cross(axis, horizontal)
    half_width = float(np.max(np.abs(np.asarray(box_corners)[:, :2])))
    half_length = float(np.max(np.abs(np.asarray(box_corners)[:, 2])))
    return (np.asarray(tvec, np.float64).reshape(3)
            + BOX_SIGNS[:, 0:1] * half_width * horizontal
            + BOX_SIGNS[:, 1:2] * half_width * vertical
            + BOX_SIGNS[:, 2:3] * half_length * axis)


def render_box_layer(frame_shape, box_corners, K, dist, rvec, tvec,
                     display_y_offset_px=BOX_DISPLAY_Y_OFFSET_PX,
                     neutralize_roll=False):
    layer = np.zeros(frame_shape, np.uint8)
    if not np.isfinite(display_y_offset_px):
        raise ValueError('Box display Y offset must be finite')
    if neutralize_roll:
        camera_points = roll_neutral_box_points(box_corners, rvec, tvec)
    else:
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        camera_points = np.asarray(box_corners, np.float64) @ rotation.T + np.asarray(
            tvec, np.float64).reshape(1, 3)
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 1.0):
        return layer
    projected = project(camera_points, np.zeros(3), np.zeros(3), K, dist)
    if not np.isfinite(projected).all() or (np.abs(projected) >= 1e6).any():
        return layer
    projected[:, 1] += display_y_offset_px
    h, w = frame_shape[:2]
    for a, b in BOX_EDGES:
        p1 = tuple(np.rint(projected[a]).astype(int))
        p2 = tuple(np.rint(projected[b]).astype(int))
        visible, q1, q2 = cv2.clipLine((0, 0, w, h), p1, p2)
        if visible:
            cv2.line(layer, q1, q2, (40, 220, 255), 2, cv2.LINE_AA)
    return layer


def make_xyz_record(frame_index, elapsed_s, status, rvec, tvec):
    rotation = np.asarray(rvec, dtype=np.float64).reshape(3)
    center = np.asarray(tvec, dtype=np.float64).reshape(3)
    valid = (status in DISPLAY_POSE_STATES and np.isfinite(rotation).all()
             and np.isfinite(center).all() and center[2] > 0)
    return dict(frame_index=int(frame_index), elapsed_s=round(float(elapsed_s), 4),
                status=status, valid=bool(valid), coordinate_frame='rgb_camera_optical',
                center_xyz_mm=center.tolist() if valid else None,
                rvec_rad=rotation.tolist() if valid else None)


def draw_status(frame, cad_layer, status, inliers, rms, fps, static_detail='',
                center_xyz_mm=None, pose_held=False):
    overlay = cv2.addWeighted(frame, 1.0, cad_layer, .72, 0)
    color = (0, 200, 0) if status == 'TRACKING' else (0, 150, 255)
    text = f'{status} | inliers {inliers} | RMS {rms:.1f}px | {fps:.1f} FPS | R reseed C contour S snapshot Q quit'
    cv2.putText(overlay, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    if static_detail:
        cv2.putText(overlay, static_detail, (8, 46), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (255, 210, 80), 1, cv2.LINE_AA)
    if center_xyz_mm is None:
        xyz_text = 'RGB camera XYZ (mm): unavailable - tracking not valid'
        xyz_color = (80, 180, 255)
    else:
        x, y, z = center_xyz_mm
        xyz_text = f'RGB camera XYZ (mm): X {x:+.1f}  Y {y:+.1f}  Z {z:+.1f}'
        if pose_held:
            xyz_text += ' | last measured'
        xyz_color = (80, 180, 255) if pose_held else (220, 255, 170)
    cv2.putText(overlay, xyz_text, (8, 68), cv2.FONT_HERSHEY_SIMPLEX,
                .43, xyz_color, 1, cv2.LINE_AA)
    return overlay


def make_search_mask(image_shape, vertices, rvec, tvec, K, dist, margin):
    projected = project(vertices, rvec, tvec, K, dist)
    x1, y1, x2, y2 = image_roi(projected, image_shape, margin)
    mask = np.zeros(image_shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def pose_delta(previous_rvec, previous_tvec, candidate_rvec, candidate_tvec):
    previous_R, _ = cv2.Rodrigues(previous_rvec.reshape(3, 1))
    candidate_R, _ = cv2.Rodrigues(candidate_rvec.reshape(3, 1))
    relative = candidate_R @ previous_R.T
    cosine = np.clip((np.trace(relative) - 1.0) * .5, -1.0, 1.0)
    rotation_deg = float(np.rad2deg(np.arccos(cosine)))
    translation_mm = float(np.linalg.norm(candidate_tvec - previous_tvec))
    return rotation_deg, translation_mm


def axis_line_change_deg(reference_rvec, current_rvec):
    reference, _ = cv2.Rodrigues(np.asarray(reference_rvec, np.float64).reshape(3, 1))
    current, _ = cv2.Rodrigues(np.asarray(current_rvec, np.float64).reshape(3, 1))
    cosine = np.clip(abs(float(np.dot(reference[:, 2], current[:, 2]))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def anchor_motion(anchor_points, current_points):
    """Return robust translation and total displacement from a fixed lock anchor."""
    anchor = np.asarray(anchor_points, np.float64).reshape(-1, 2)
    current = np.asarray(current_points, np.float64).reshape(-1, 2)
    if len(anchor) == 0 or anchor.shape != current.shape:
        return np.zeros(2, np.float64), float('inf')
    delta = current - anchor
    valid = np.isfinite(delta).all(axis=1)
    if not valid.any():
        return np.zeros(2, np.float64), float('inf')
    delta = delta[valid]
    return np.median(delta, axis=0), float(np.median(np.linalg.norm(delta, axis=1)))


def resolve_camera_index(camera_name: str, requested_index: int | None):
    if requested_index is not None:
        print(f'Using camera index {requested_index} (manual override)')
        return requested_index, None
    if not camera_name:
        return 0, None
    if sys.platform != 'win32':
        # Linux exposes multiple V4L2 nodes for some cameras. Match the
        # sysfs device label, then probe candidates so metadata/IR nodes are
        # not mistaken for the readable RGB stream.
        nodes = sorted(Path('/sys/class/video4linux').glob('video*'),
                       key=lambda path: int(path.name[5:])
                       if path.name[5:].isdigit() else path.name)
        requested = camera_name.strip().casefold()
        candidates = []
        for node in nodes:
            try:
                label = (node / 'name').read_text(encoding='utf-8').strip()
                index = int(node.name[5:])
            except (OSError, ValueError):
                continue
            if requested in label.casefold() or label.casefold() in requested:
                candidates.append((index, label))
        for index, _label in candidates:
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            try:
                if not cap.isOpened():
                    continue
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    print(f'Using camera {index}: {camera_name} (Linux V4L2)')
                    return index, camera_name
            finally:
                cap.release()
        available = ', '.join(f'{index}: {label}' for index, label in candidates) or 'none'
        raise RuntimeError(
            f'Camera named {camera_name!r} has no readable RGB stream. '
            f'Candidate V4L2 nodes: {available}')
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError as exc:
        raise RuntimeError('Camera-name selection requires pygrabber: pip install pygrabber') from exc
    devices = FilterGraph().get_input_devices()
    matches = [index for index, name in enumerate(devices)
               if name.strip().casefold() == camera_name.strip().casefold()]
    if not matches:
        available = ', '.join(f'{index}: {name}' for index, name in enumerate(devices)) or 'none'
        raise RuntimeError(f'Camera named {camera_name!r} was not found. Available cameras: {available}')
    index = matches[0]
    print(f'Using camera {index}: {devices[index]}')
    return index, devices[index]


def save_pose(path: Path, rvec, tvec, frame=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(units='mm', rvec=rvec.reshape(3).tolist(), tvec_mm=tvec.reshape(3).tolist(),
                   saved_at=datetime.now().isoformat(timespec='seconds'))
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if frame is not None:
        image_path = path.with_suffix('.png')
        ok, encoded = cv2.imencode('.png', frame)
        if ok:
            encoded.tofile(image_path)
    print(f'Saved pose: {path}')


def run(args, capture=None, detector=None, window_name=None, keep_window=False,
        keep_capture_on_reacquire=False):
    if args.camera_fourcc and len(args.camera_fourcc) != 4:
        raise RuntimeError('--camera-fourcc must contain exactly four characters')
    K, dist, size = load_calibration(args.intrinsics)
    vertices, edges, model_points = make_edge_points(args.model, args.max_edges, args.edge_samples)
    box_corners = load_model_box(args.model)
    rvec_initial, tvec_initial = load_pose(args.pose)
    rvec, tvec = rvec_initial.copy(), tvec_initial.copy()
    surface_seeder = None
    if getattr(args, 'surface_seeding', False):
        from .surface_seeding import VisibleSurfaceSeeder
        surface_seeder = VisibleSurfaceSeeder(args.model, K, dist)
    three_d_only_after_init = getattr(args, 'three_d_only_after_init', False)
    # After handoff YOLO only proposes a region for independent metric RGB
    # relocalization and pose verification. It never becomes the displayed
    # tracker or sends this live session back to the 2D search/depth burst.
    recovery = None
    session_reference = None
    recovery_reference = getattr(args, 'rgb_recovery_reference', None)
    if recovery_reference is None and getattr(args, 'rgb_pose_recovery', False):
        recovery_reference = json.loads(args.pose.read_text(encoding='utf-8')).get('rgb_image')
    if recovery_reference is not None and getattr(args, 'rgb_pose_recovery', False):
        reference_path = Path(recovery_reference)
        if not reference_path.is_absolute():
            reference_path = args.pose.parent / reference_path
        reference = cv2.imdecode(np.fromfile(reference_path, np.uint8), cv2.IMREAD_COLOR)
        if reference is None or reference.shape[1::-1] != size:
            raise RuntimeError(f'Invalid RGB recovery reference: {reference_path}')
        session_reference = reference
        try:
            recovery = RgbPoseRecovery(reference, rvec, tvec, vertices, K, dist)
        except RuntimeError as exc:
            print(f'RGB reference recovery unavailable: {exc}')
    owns_detector = detector is None
    recovery_pending = None
    rgb_relocalizer = None
    rgb_initializer = getattr(args, 'rgb_initializer', None)
    if rgb_initializer is not None and hasattr(rgb_initializer, 'new_search'):
        from .rgb_pose_relocalizer import RgbPoseRelocalizer
        rgb_relocalizer = RgbPoseRelocalizer(
            rgb_initializer, session_reference, (rvec_initial, tvec_initial), surface_seeder)
    recovery_stats = dict(checks=0, proposals=0, fast_proposals=0, accepted=0,
                          fast_accepted=0, stale=0, depth_requests=0,
                          total_ms=0.0, axis_conflicts=0, box_conflicts=0,
                          feature_verifications=0, unassociated_detections=0)
    # A live RGB-D session may opt into one more recovery layer after RGB
    # relocalization has been exhausted. Keep this separate from
    # ``three_d_only_after_init``: that flag controls the tracking source,
    # while this opt-in only requests a fresh, depth-verified initialization
    # after sustained loss. Offline/replay callers retain the old behavior
    # unless they explicitly enable the fallback.
    reacquire_on_loss = (getattr(args, 'reacquire_on_loss', False) and
                         (not three_d_only_after_init or
                          getattr(args, 'depth_fallback_on_loss', False)))
    reacquire_model = getattr(args, 'reacquire_model', None)
    if detector is None and reacquire_model is not None:
        from .background_detector import BackgroundDetector
        def load_detector():
            from .live_detector import prepare_detector
            return prepare_detector(reacquire_model)
        detector = BackgroundDetector(load_detector,
                                      lambda model, image, turn: detect_target_box(
                                          model, image, turn, fast=True))
    detector_armed = False
    verification = PoseVerification()
    detector_future = None
    detector_pose = None
    detector_submitted_frame = 0
    detector_submitted_time = 0.0
    detector_frame = None
    detector_checks = 0
    lost_frames = 0
    soft_loss_frames = 0
    reacquire_requested = False
    static_corrector = None
    static_executor = None
    if args.static_correction:
        reference = cv2.imdecode(np.fromfile(args.static_reference_image, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
        if reference is None or reference.shape[1::-1] != size:
            raise RuntimeError(f'Invalid static reference image: {args.static_reference_image}')
        reference_rvec, reference_tvec = load_pose(args.static_reference_pose)
        static_corrector = ReferencePoseCorrector(
            reference, reference_rvec, reference_tvec, vertices, K, dist,
            reference_name=args.static_reference_image.stem)
        static_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='static-pose')
    video = getattr(args, 'video', None)
    headless = getattr(args, 'headless', False)
    session_depth_margin = getattr(args, 'session_depth_margin', 100.0)
    session_depth_gate = getattr(args, 'session_depth_gate', True)
    if video is not None:
        if capture is not None:
            raise RuntimeError('A live capture cannot be combined with --video')
        video = Path(video)
        if not video.is_file():
            raise RuntimeError(f'Video does not exist: {video}')
        camera_index, camera_name = None, None
        try:
            video_timestamps = load_video_timestamps(video)
            session_bounds = load_session_depth_bounds(video, session_depth_margin)
        except Exception:
            if static_executor is not None:
                static_executor.shutdown(wait=False, cancel_futures=True)
            raise
        cap = cv2.VideoCapture(str(video))
    else:
        camera_index, camera_name = (args.camera, getattr(args, 'camera_identity_name', None)) \
            if capture is not None else \
            resolve_camera_index(args.camera_name, args.camera)
        video_timestamps = None
        session_bounds = None
        cap = capture if capture is not None else cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        if static_executor is not None:
            static_executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f'Cannot open {video if video is not None else f"camera {camera_index}"}')
    if video is None:
        if capture is None:
            if args.camera_fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if capture is None and args.camera_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        # Keep the live view close to the newest frame. Some DirectShow drivers
        # ignore this property, but setting it is harmless for test captures.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if video is not None else 0.0
    camera_reported_fps = float(cap.get(cv2.CAP_PROP_FPS)) if video is None else None
    depth_gate_bounds = session_bounds if session_depth_gate else None
    min_depth = (max(args.min_depth, depth_gate_bounds['min_mm'])
                 if depth_gate_bounds else args.min_depth)
    max_depth = (min(args.max_depth, depth_gate_bounds['max_mm'])
                 if depth_gate_bounds else args.max_depth)
    if min_depth >= max_depth or not min_depth <= tvec[2] <= max_depth:
        cap.release()
        if static_executor is not None:
            static_executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError(f'Initial depth {tvec[2]:.1f} mm is outside '
                           f'allowed range [{min_depth:.1f}, {max_depth:.1f}] mm')
    if video_timestamps is not None:
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if reported_frames > 0 and len(video_timestamps) != reported_frames:
            cap.release()
            if static_executor is not None:
                static_executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f'Video timestamp count {len(video_timestamps)} differs from '
                               f'video frame count {reported_frames}: {video}')
    session_corrector = None
    session_relocalization = getattr(args, 'session_relocalization', True)
    session_min_coverage = getattr(args, 'session_min_coverage', .95)
    if session_bounds and session_relocalization:
        pose_metadata = json.loads(args.pose.read_text(encoding='utf-8'))
        expected_source = f"integration/{session_bounds['session_id']}/start"
        if pose_metadata.get('source_sample_id') == expected_source:
            reference_path = Path(session_bounds['reference_image'])
            reference = cv2.imdecode(np.fromfile(reference_path, np.uint8), cv2.IMREAD_COLOR)
            if reference is None or reference.shape[1::-1] != size:
                cap.release()
                if static_executor is not None:
                    static_executor.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(f'Invalid session reference image: {reference_path}')
            session_corrector = ReferencePoseCorrector(
                reference, rvec_initial, tvec_initial, vertices, K, dist,
                reference_name=session_bounds['session_id'])
    window = window_name or 'CAD RGB tracker | C CAD outline | S snapshot | Q quit'
    from .tracking_diagnostics import TrackingDiagnostics
    diagnostic_dir = args.output.parent / ('diagnostics_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    diagnostic_capture = TrackingDiagnostics(
        diagnostic_dir, K, dist, args.model, args.intrinsics,
        record=getattr(args, 'record_diagnostics', False),
        duration_s=getattr(args, 'diagnostic_duration_s', 60.),
        fps=source_fps if video is not None else args.camera_fps)
    cad_outline = getattr(args, 'cad_outline', False)
    outline_seeder = surface_seeder
    if cad_outline and outline_seeder is None:
        from .surface_seeding import VisibleSurfaceSeeder
        outline_seeder = VisibleSurfaceSeeder(args.model, K, dist)
    if diagnostic_capture.enabled:
        print(f'Synchronized diagnostic recording (rolling window {diagnostic_capture.duration_s:g}s): '
              f'{diagnostic_dir.resolve()}', flush=True)
    previous_gray = None
    track_model = np.empty((0, 3), np.float64)
    track_image = np.empty((0, 2), np.float32)
    status, inliers, rms, fps = 'SEEDING', 0, 0.0, 0.0
    needs_seed = True
    reseed_requires_pose = False
    pose_metadata = json.loads(args.pose.read_text(encoding='utf-8'))
    initial_seed = pose_metadata.get('tracking_seed')
    if initial_seed is not None and pose_metadata.get('rgb_pose_quality', {}).get('accepted'):
        points = np.asarray(initial_seed['model_points_mm'], np.float64).reshape(-1, 3)
        pixels = np.asarray(initial_seed['image_points_px'], np.float32).reshape(-1, 2)
        seed_frame = cv2.imdecode(np.fromfile(pose_metadata['rgb_image'], np.uint8),
                                  cv2.IMREAD_GRAYSCALE)
        if (len(points) >= 12 and len(points) == len(pixels) and seed_frame is not None
                and seed_frame.shape == (size[1], size[0])
                and np.isfinite(points).all() and np.isfinite(pixels).all()):
            track_model, track_image = points, pixels
            previous_gray = seed_frame
            needs_seed = False
    flow_median = flow_p90 = 0.0
    static_frames = 0
    last_static_check = -args.static_check_interval
    last_failure_frame = -1000000
    static_future = None
    static_submission = None
    static_pending = None
    static_detail = 'static correction armed' if static_corrector is not None else ''
    static_stats = dict(checks=0, proposals=0, accepted=0, locks=0,
                        rejected=0, stale=0, total_ms=0.0)
    static_lock = False
    static_lock_shift = np.zeros(2, np.float64)
    static_lock_motion = 0.0
    static_lock_anchor_image = np.empty((0, 2), np.float32)
    static_match_failures = 0
    pose_before_frame = rvec.copy()
    session_pending = None
    session_pending_count = 0
    session_loss_latched = False
    session_stats = dict(appearance_checks=0, appearance_verified_frames=0,
                         out_of_view_frames=0, identity_mismatch_frames=0,
                         relocalizations=0)
    start_time = time.perf_counter()
    last_time = start_time
    frame_count = 0
    video_eof = False
    last_seed_frame = -max(args.refresh, 1)
    cad_layer = None
    fps_samples = []
    status_counts = {}
    presentation = PoseDisplayState()
    display_counts = {}
    display_transitions = box_visibility_changes = 0
    previous_display_status = None
    previous_box_visible = None
    timing = dict(capture_ms=0.0, seed_ms=0.0, track_ms=0.0, draw_ms=0.0,
                  seed_calls=0, track_calls=0, draw_calls=0)
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_handle = None
    xyz_handle = None
    xyz_jsonl = getattr(args, 'xyz_jsonl', None)
    try:
        if args.log is None:
            args.log = DATA_ROOT / 'tracking' / f'track_log_{run_stamp}.csv'
        if args.summary is None:
            args.summary = DATA_ROOT / 'tracking' / f'track_summary_{run_stamp}.json'
        args.log.parent.mkdir(parents=True, exist_ok=True)
        if xyz_jsonl is not None:
            xyz_jsonl = Path(xyz_jsonl)
            if xyz_jsonl.resolve() in {args.log.resolve(), args.summary.resolve()}:
                raise RuntimeError('--xyz-jsonl must differ from --log and --summary')
            xyz_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_handle = None if args.no_log else args.log.open('w', newline='', encoding='utf-8')
        xyz_handle = xyz_jsonl.open('w', encoding='utf-8') if xyz_jsonl else None
    except Exception:
        if log_handle is not None:
            log_handle.close()
        cap.release()
        if static_executor is not None:
            static_executor.shutdown(wait=False, cancel_futures=True)
        raise
    log_writer = None if log_handle is None else csv.writer(log_handle)
    if log_writer is not None:
        log_writer.writerow(['frame_index', 'elapsed_s', 'status', 'track_count',
                             'inliers', 'rms_px', 'fps',
                             'rvec_x', 'rvec_y', 'rvec_z', 'tx_mm', 'ty_mm', 'tz_mm',
                             'frame_ms', 'seed_ms', 'track_ms', 'draw_ms',
                             'flow_median_px', 'flow_p90_px', 'flow_dx_px', 'flow_dy_px',
                             'static_lock_shift_px', 'static_lock_motion_px',
                             'static_action', 'static_matches', 'static_inliers',
                             'static_inlier_ratio', 'static_spread_px2',
                             'static_center_error_px', 'static_angle_error_deg',
                             'static_scale_ratio', 'static_ms', 'source_elapsed_s',
                             'projected_coverage', 'appearance_inliers',
                             'appearance_center_error_px', 'session_action',
                             'tracking_status', 'verification_conflicts',
                             'recovery_pending', 'display_status', 'box_visible', 'pose_held',
                             'capture_ms'])
        log_handle.flush()
    summary = dict(started_at=datetime.now().isoformat(timespec='seconds'), frames=0,
                   tracking_frames=0, low_quality_frames=0, max_inliers=0, max_fps=0.0,
                   log=str(args.log.resolve()) if log_writer is not None else None,
                   xyz_jsonl=str(xyz_jsonl.resolve()) if xyz_jsonl is not None else None,
                   settings=dict(camera_index=camera_index, camera_name=camera_name,
                                 camera_device_path=getattr(args, 'camera_device_path', None),
                                 video=str(video.resolve()) if video is not None else None,
                                 source_fps=source_fps if video is not None else None,
                                 timestamp_source=('metadata' if video_timestamps is not None else
                                                   'video' if video is not None else None),
                                 headless=headless,
                                 session_depth_gate=session_depth_gate,
                                 session_depth_bounds=session_bounds,
                                 effective_depth_range_mm=[min_depth, max_depth],
                                 session_relocalization=bool(session_corrector),
                                 session_min_coverage=session_min_coverage,
                                 camera_fps=args.camera_fps, camera_fourcc=args.camera_fourcc,
                                 box_roll_mode=getattr(args, 'box_roll_mode', 'true'),
                                 show_box=getattr(args, 'show_box', False),
                                 pose_axes=getattr(args, 'pose_axes', True),
                                 cad_outline=getattr(args, 'cad_outline', False),
                                 display_y_offset_px=getattr(args, 'box_display_y_offset_px', BOX_DISPLAY_Y_OFFSET_PX),
                                 camera_reported_fps=camera_reported_fps,
                                 max_edges=args.max_edges, edge_samples=args.edge_samples,
                                 max_points=args.max_points, refresh=args.refresh,
                                 reseed_below=args.reseed_below, draw_interval=args.draw_interval,
                                 roi_margin=args.roi_margin, seed_radius=args.seed_radius,
                                 flow_error=args.flow_error, fb_error=args.fb_error,
                                 pnp_error=args.pnp_error, pnp_iterations=args.pnp_iterations,
                                 max_rms=args.max_rms, min_inliers=args.min_inliers,
                                 min_inlier_ratio=args.min_inlier_ratio,
                                 max_rotation_step=args.max_rotation_step,
                                 max_translation_step=args.max_translation_step,
                                 max_axis_change_from_depth_deg=getattr(
                                     args, 'max_axis_change_from_depth_deg', 0.0),
                                 detector_interval=getattr(args, 'detector_interval',
                                                           DEFAULT_DETECTOR_INTERVAL),
                                 rgb_pose_recovery=recovery is not None,
                                 fast_pose_recovery=getattr(args, 'fast_pose_recovery', False),
                                 three_d_only_after_init=three_d_only_after_init,
                                 depth_fallback_on_loss=getattr(
                                     args, 'depth_fallback_on_loss', False),
                                 surface_seeding=surface_seeder is not None,
                                 min_depth=args.min_depth, max_depth=args.max_depth,
                                 static_correction=args.static_correction,
                                 static_min_frames=args.static_min_frames,
                                 static_check_interval=args.static_check_interval,
                                 static_flow_median=args.static_flow_median,
                                 static_flow_p90=args.static_flow_p90,
                                 static_search_margin=args.static_search_margin,
                                 static_gain=args.static_gain,
                                  static_max_center_step=args.static_max_center_step,
                                  static_max_depth_step=args.static_max_depth_step,
                                  static_max_angle_step=args.static_max_angle_step,
                                 static_reference_image=str(args.static_reference_image.resolve())
                                 if args.static_correction else None,
                                 static_reference_pose=str(args.static_reference_pose.resolve())
                                 if args.static_correction else None))
    region_assist = None
    region_executor = None
    region_future = None
    region_provider = getattr(args, 'region_tracker_provider', None)
    if getattr(args, 'region_tracking', False) and region_provider is None:
        from .m3t_live import NativeRegionTracker
        region_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='m3t-prepare')
        region_future = region_executor.submit(NativeRegionTracker, args.model, K, dist, size,
                                                args.summary.parent / 'm3t_stream')
    region_unavailable = False
    current_detection_conflicts = 0
    detector_executor = (ThreadPoolExecutor(max_workers=1, thread_name_prefix='target-check')
                         if detector is not None or recovery is not None or rgb_relocalizer is not None
                         else None)
    try:
        while True:
            loop_started = time.perf_counter()
            if rgb_relocalizer is None:
                provider = getattr(args, 'rgb_initializer_provider', None)
                ready = provider() if provider is not None else None
                if ready is not None and hasattr(ready, 'new_search'):
                    from .rgb_pose_relocalizer import RgbPoseRelocalizer
                    rgb_relocalizer = RgbPoseRelocalizer(
                        ready, session_reference, (rvec_initial, tvec_initial), surface_seeder)
                    rgb_initializer = ready
                    if detector_executor is None:
                        detector_executor = ThreadPoolExecutor(
                            max_workers=1, thread_name_prefix='target-check')
            if getattr(args, 'region_tracking', False) and region_assist is None and not region_unavailable:
                try:
                    ready_region = (region_provider() if region_provider is not None else
                                    region_future.result() if region_future is not None and
                                    region_future.done() else None)
                    if ready_region is not None:
                        from .m3t_live import RegionAssist
                        region_assist = RegionAssist(ready_region)
                except Exception as exc:
                    region_unavailable = True
                    print(f'M3T unavailable: {exc}; continuing RGB recovery', flush=True)
            region_model_points = np.empty((0, 3), np.float64)
            region_image_points = np.empty((0, 2), np.float32)
            pose_before_frame = rvec.copy()
            seed_ms = track_ms = draw_ms = 0.0
            flow_median = flow_p90 = 0.0
            flow_vector = np.zeros(2, np.float64)
            static_action = ''
            static_unlock_shift = None
            static_unlock_motion = None
            static_obs_matches = 0
            static_obs_inliers = 0
            static_obs_ratio = 0.0
            static_obs_spread = 0.0
            static_center_error = static_angle_error = 0.0
            static_scale_ratio = 1.0
            static_elapsed_ms = 0.0
            capture_started = time.perf_counter()
            ok, frame = cap.read()
            capture_ms = (time.perf_counter() - capture_started) * 1000.0
            timing['capture_ms'] += capture_ms
            if not ok:
                if video is not None and frame_count > 0:
                    if video_timestamps is not None and frame_count != len(video_timestamps):
                        raise RuntimeError(f'Video ended after {frame_count} frames, expected '
                                           f'{len(video_timestamps)} from metadata: {video}')
                    video_eof = True
                    break
                raise RuntimeError('Camera frame read failed.')
            if video_timestamps is not None and frame_count >= len(video_timestamps):
                raise RuntimeError(f'Video has more frames than timestamps: {video}')
            if frame.shape[1::-1] != size:
                raise RuntimeError(f'Input returned {frame.shape[1]}x{frame.shape[0]}, '
                                   f'expected {size[0]}x{size[1]}')
            source_elapsed_s = (float(video_timestamps[frame_count])
                                if video_timestamps is not None else
                                float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                                if video is not None else None)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_detection = detector.get_box(frame) if hasattr(detector, 'get_box') else None
            coverage = 1.0
            appearance_inliers = 0
            appearance_center_error = None
            session_action = ''
            visibility_rejected = False
            identity_mismatch = False
            appearance_verified = False
            if session_corrector is not None:
                coverage = projected_coverage(vertices, rvec, tvec, K, dist, frame.shape)
                observation = session_corrector.observe(gray)
                session_stats['appearance_checks'] += 1
                appearance_verified = bool(observation is not None and
                                           observation.inliers >= 8 and
                                           observation.inlier_ratio >= .75 and
                                           observation.spread_px2 >= 600.0)
                if appearance_verified:
                    session_stats['appearance_verified_frames'] += 1
                    appearance_inliers = observation.inliers
                    current_center = project(np.zeros((1, 3), np.float64),
                                             rvec, tvec, K, dist)[0]
                    appearance_center_error = float(np.linalg.norm(
                        observation.control_points[0] - current_center))
                    identity_mismatch = appearance_center_error > 35.0
                    _, reference_angle_error, reference_scale = session_corrector.residual(
                        observation, rvec_initial, tvec_initial)
                    relocalizable = (abs(reference_angle_error) <= 15.0 and
                                     .65 <= reference_scale <= 1.5)
                    if identity_mismatch and relocalizable:
                        strong_match = (observation.inliers >= 20 and
                                        observation.spread_px2 >= 1500.0)
                        if session_corrector.consistent(
                                session_pending, observation,
                                center_tolerance=12.0 if strong_match else 2.0,
                                angle_tolerance=5.0 if strong_match else 1.5,
                                scale_tolerance=.08 if strong_match else .03):
                            session_pending_count += 1
                        else:
                            session_pending_count = 1
                        session_pending = observation
                        required_matches = 2 if strong_match else 3
                        if session_pending_count >= required_matches:
                            candidate_rvec, candidate_tvec, _ = session_corrector.apply(
                                observation, rvec_initial, tvec_initial, gain=1.0,
                                max_center_step_px=1000.0,
                                max_angle_step_deg=15.0,
                                max_depth_step_mm=1000.0)
                            candidate_coverage = projected_coverage(
                                vertices, candidate_rvec, candidate_tvec, K, dist, frame.shape)
                            if min_depth <= candidate_tvec[2] <= max_depth and \
                                    candidate_coverage >= session_min_coverage:
                                rvec, tvec = candidate_rvec, candidate_tvec
                                coverage = candidate_coverage
                                needs_seed = True
                                track_model = np.empty((0, 3), np.float64)
                                track_image = np.empty((0, 2), np.float32)
                                static_lock = False
                                static_pending = None
                                cad_layer = None
                                identity_mismatch = False
                                session_action = 'RELOCALIZED'
                                session_stats['relocalizations'] += 1
                            session_pending = None
                            session_pending_count = 0
                    else:
                        session_pending = None
                        session_pending_count = 0
                else:
                    session_pending = None
                    session_pending_count = 0
                visibility_rejected = not appearance_verified and coverage < session_min_coverage
                # Once the reference appearance says the target left or the
                # identity changed, keep the metric overlay invalid until a
                # fresh appearance/relocalization is confirmed. This prevents
                # the seed branch from publishing alternating stale poses.
                if visibility_rejected or identity_mismatch:
                    session_loss_latched = True
                elif session_loss_latched and appearance_verified:
                    session_loss_latched = False
                if session_loss_latched and not appearance_verified:
                    visibility_rejected = True
                if visibility_rejected:
                    session_action = 'OUT OF VIEW'
                    session_stats['out_of_view_frames'] += 1
                elif identity_mismatch:
                    session_action = 'IDENTITY MISMATCH'
                    session_stats['identity_mismatch_frames'] += 1
            periodic_seed = args.refresh > 0 and frame_count - last_seed_frame >= args.refresh
            # Replenish before the next small loss drops below min_inliers.
            low_points = (previous_gray is not None and args.reseed_below > 0
                          and len(track_model) < max(args.reseed_below, args.min_inliers * 2)
                          and frame_count - last_seed_frame >= 3)
            # Follow existing correspondences before replenishing them. A
            # maintenance seed must not consume a frame without estimating motion.
            force_seed = needs_seed or previous_gray is None or len(track_model) < 8
            if visibility_rejected or identity_mismatch:
                status = session_action
                inliers, rms = 0, 0.0
                needs_seed = True
                track_model = np.empty((0, 3), np.float64)
                track_image = np.empty((0, 2), np.float32)
            elif force_seed and reseed_requires_pose:
                status = 'TARGET LOST'
                inliers, rms = 0, 0.0
                needs_seed = True
                track_model = np.empty((0, 3), np.float64)
                track_image = np.empty((0, 2), np.float32)
            elif force_seed:
                operation_started = time.perf_counter()
                track_model, track_image = seed_points(
                    gray, vertices, edges, model_points, rvec, tvec, K, dist, args.max_points,
                    seed_radius=args.seed_radius, roi_margin=args.roi_margin,
                    surface_seeder=surface_seeder)
                seed_ms = (time.perf_counter() - operation_started) * 1000.0
                timing['seed_ms'] += seed_ms
                timing['seed_calls'] += 1
                status = 'SEEDED' if len(track_model) >= 8 else 'SEED FAILED'
                inliers, rms = len(track_model), 0.0
                needs_seed = len(track_model) < 8
                last_seed_frame = frame_count
                static_frames = 0
                static_pending = None
                last_failure_frame = frame_count
            elif len(track_model) >= 8:
                operation_started = time.perf_counter()
                projected_vertices = project(vertices, rvec, tvec, K, dist)
                x1, y1, x2, y2 = image_roi(projected_vertices, gray.shape, args.roi_margin,
                                            extra_points=track_image)
                inside = (track_image[:, 0] >= x1) & (track_image[:, 0] < x2) & \
                         (track_image[:, 1] >= y1) & (track_image[:, 1] < y2)
                flow_model = track_model[inside]
                flow_image = track_image[inside]
                flow_anchor = None
                if static_lock and len(static_lock_anchor_image) == len(track_image):
                    flow_anchor = static_lock_anchor_image[inside]
                if len(flow_model) >= 8:
                    origin = np.array([x1, y1], np.float32)
                    previous_roi = previous_gray[y1:y2, x1:x2]
                    current_roi = gray[y1:y2, x1:x2]
                    local_image = (flow_image - origin).reshape(-1, 1, 2)
                    next_local, flow_status, flow_error = cv2.calcOpticalFlowPyrLK(
                        previous_roi, current_roi, local_image, None,
                        winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                else:
                    next_local = flow_status = flow_error = None
                if next_local is not None and flow_status is not None:
                    good = flow_status.reshape(-1).astype(bool)
                    if flow_error is not None:
                        good &= flow_error.reshape(-1) < args.flow_error
                    candidates_local = next_local.reshape(-1, 2)
                    good &= np.isfinite(candidates_local).all(axis=1)
                    if args.fb_error > 0:
                        back_local, back_status, _ = cv2.calcOpticalFlowPyrLK(
                            current_roi, previous_roi, next_local, None,
                            winSize=(21, 21), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                        if back_local is None or back_status is None:
                            good[:] = False
                        else:
                            good &= back_status.reshape(-1).astype(bool)
                            backward_error = np.linalg.norm(back_local.reshape(-1, 2) -
                                                            local_image.reshape(-1, 2), axis=1)
                            good &= backward_error <= args.fb_error
                    candidates = candidates_local + origin
                    good &= (candidates[:, 0] >= 0) & (candidates[:, 0] < size[0]) & \
                            (candidates[:, 1] >= 0) & (candidates[:, 1] < size[1])
                    if good.any():
                        flow_delta = candidates_local[good] - local_image.reshape(-1, 2)[good]
                        flow_vector = np.median(flow_delta, axis=0).astype(np.float64)
                        motion = np.linalg.norm(flow_delta, axis=1)
                        flow_median = float(np.median(motion))
                        flow_p90 = float(np.percentile(motion, 90))
                    if good.sum() >= 8:
                        good_model = flow_model[good]
                        good_image = candidates[good]
                        region_model_points, region_image_points = good_model, good_image
                        if static_lock:
                            good_anchor = None if flow_anchor is None else flow_anchor[good]
                            if good_anchor is None:
                                static_lock_shift[:] = 0.0
                                static_lock_motion = float('inf')
                            else:
                                static_lock_shift, static_lock_motion = anchor_motion(
                                    good_anchor, good_image)
                            unlock_median = max(args.static_flow_median * 1.25, .4)
                            unlock_p90 = max(args.static_flow_p90 * 1.25, 1.0)
                            enough_points = len(good_model) >= STATIC_MIN_LOCK_INLIERS
                            still = flow_median <= unlock_median and flow_p90 <= unlock_p90 and \
                                np.linalg.norm(static_lock_shift) <= STATIC_MAX_LOCK_DISPLACEMENT_PX and \
                                static_lock_motion <= STATIC_MAX_LOCK_DISPLACEMENT_PX
                            if enough_points and still:
                                track_image = good_image.astype(np.float32)
                                track_model = good_model
                                static_lock_anchor_image = good_anchor.astype(np.float32)
                                inliers, rms = len(good_model), 0.0
                                status = 'STATIC LOCKED'
                                needs_seed = False
                            else:
                                static_unlock_shift = float(np.linalg.norm(static_lock_shift))
                                static_unlock_motion = static_lock_motion
                                static_lock = False
                                static_lock_shift[:] = 0.0
                                static_lock_motion = 0.0
                                static_lock_anchor_image = np.empty((0, 2), np.float32)
                                static_match_failures = 0
                                needs_seed = True
                                track_model = np.empty((0, 3), np.float64)
                                track_image = np.empty((0, 2), np.float32)
                                status = 'STATIC LOST' if not enough_points else 'MOTION RESEED'
                                inliers, rms = int(good.sum()), 0.0
                        else:
                            # Fast local solve first, then an independent current-frame
                            # solve on the same measured correspondences. The fallback
                            # is what breaks pose lock when motion is larger than the
                            # previous-frame prediction, while the local solve keeps
                            # ordinary tracking stable and preserves out-of-view gating.
                            attempts = [((rvec.copy(), tvec.copy(), True),
                                         cv2.SOLVEPNP_ITERATIVE)]
                            # When the appearance/session gate already says the
                            # object is absent, a free solve can fit background
                            # texture and delay the explicit OUT OF VIEW state.
                            if not identity_mismatch and not visibility_rejected:
                                attempts.append(((None, None, False), cv2.SOLVEPNP_EPNP))
                            accepted_candidates = []
                            rejection = 'PNP FAILED'
                            best_metrics = None
                            for attempt_index, ((guess_rvec, guess_tvec, use_guess), pnp_flag) in enumerate(attempts):
                                solved, candidate_rvec, candidate_tvec, mask = cv2.solvePnPRansac(
                                    good_model.astype(np.float32), good_image.astype(np.float32), K, dist,
                                    guess_rvec, guess_tvec, useExtrinsicGuess=use_guess,
                                    iterationsCount=args.pnp_iterations, reprojectionError=args.pnp_error,
                                    confidence=.995, flags=pnp_flag)
                                if not solved or mask is None or len(mask) < 6:
                                    rejection = 'PNP FAILED'
                                    continue
                                candidate_rvec = candidate_rvec.reshape(3)
                                candidate_tvec = candidate_tvec.reshape(3)
                                inlier_indices = mask.reshape(-1).astype(np.int64)
                                candidate_inliers = len(inlier_indices)
                                candidate_ratio = candidate_inliers / len(good_model)
                                projected = project(good_model, candidate_rvec, candidate_tvec, K, dist)
                                residual = np.linalg.norm(projected - good_image, axis=1)
                                candidate_rms = float(np.sqrt(np.mean(residual[inlier_indices] ** 2)))
                                finite_pose = np.isfinite(candidate_rvec).all() and \
                                    np.isfinite(candidate_tvec).all()
                                if finite_pose:
                                    rotation_step = axis_line_change_deg(rvec, candidate_rvec)
                                    translation_step = float(np.linalg.norm(candidate_tvec - tvec))
                                else:
                                    rotation_step = translation_step = float('inf')
                                quality_ok = candidate_inliers >= args.min_inliers and \
                                    candidate_ratio >= args.min_inlier_ratio and candidate_rms <= args.max_rms
                                support_ok = finite_pose and tracking_support_quality(
                                    good_model[inlier_indices], good_image[inlier_indices],
                                    vertices, candidate_rvec, candidate_tvec, K, dist)
                                motion_support_ok = (support_ok and
                                    pose_box_motion_consistent(
                                        rvec, tvec, candidate_rvec, candidate_tvec,
                                        box_corners, K, dist, flow_vector))
                                depth_ok = min_depth <= candidate_tvec[2] <= max_depth
                                motion_ok = rotation_step <= args.max_rotation_step and \
                                    translation_step <= args.max_translation_step
                                best_metrics = (candidate_inliers, candidate_rms)
                                if finite_pose and quality_ok and support_ok and \
                                        motion_support_ok and depth_ok and motion_ok:
                                    accepted_candidates.append({
                                        'rvec': candidate_rvec,
                                        'tvec': candidate_tvec,
                                        'inliers': inlier_indices,
                                        'count': candidate_inliers,
                                        'rms': candidate_rms,
                                        'rotation_step': rotation_step,
                                        'translation_step': translation_step,
                                        'attempt': attempt_index,
                                    })
                                    # Keep testing the independent current-frame
                                    # EPNP branch. Selection happens after all
                                    # candidates have passed the same gates, so a
                                    # failed EPNP result can never overwrite the
                                    # accepted local solution or its metrics.
                                    continue
                                rejection = ('REJECTED SUPPORT' if not support_ok or
                                             not motion_support_ok else
                                             'REJECTED QUALITY' if not quality_ok else
                                             'REJECTED DEPTH' if not depth_ok or not finite_pose else
                                             'REJECTED JUMP')
                                # A depth/support failure is evidence that the
                                # current correspondences are no longer on the
                                # object. Do not let a free EPNP branch turn
                                # that into a plausible-looking false pose.
                                if (attempt_index == 0 and rejection not in
                                        {'REJECTED JUMP', 'REJECTED QUALITY'}):
                                    break
                            if accepted_candidates:
                                # The previous-pose ITERATIVE result is the
                                # default because it is temporally stable. Let
                                # the independent EPNP result take over only
                                # when the current-frame evidence is materially
                                # better; this removes stale-pose lag without
                                # turning normal tracking into frame-to-frame
                                # EPNP jitter.
                                selected = accepted_candidates[0]
                                for candidate in accepted_candidates[1:]:
                                    rms_gain = selected['rms'] - candidate['rms']
                                    inlier_gain = candidate['count'] - selected['count']
                                    materially_better = (
                                        rms_gain >= .25 or
                                        (inlier_gain >= max(4, int(round(selected['count'] * .12))) and
                                         candidate['rms'] <= selected['rms'] * .95))
                                    if materially_better:
                                        selected = candidate
                                candidate_rvec = selected['rvec']
                                candidate_tvec = selected['tvec']
                                inlier_indices = selected['inliers']
                                rvec, tvec = candidate_rvec, candidate_tvec
                                status = 'TRACKING'
                                track_image = good_image[inlier_indices].astype(np.float32)
                                track_model = good_model[inlier_indices]
                                inliers, rms = selected['count'], selected['rms']
                                needs_seed = False
                            else:
                                status = rejection
                                inliers, rms = best_metrics if best_metrics is not None else (0, 0.0)
                                needs_seed = True
                                if status == 'REJECTED JUMP':
                                    status = 'TRACKING HOLD'
                    else:
                        status = 'FLOW LOST'
                        inliers, rms = int(good.sum()), 0.0
                        needs_seed = True
                else:
                    status = 'FLOW LOST'
                    inliers, rms = 0, 0.0
                    needs_seed = True
                track_ms = (time.perf_counter() - operation_started) * 1000.0
                timing['track_ms'] += track_ms
                timing['track_calls'] += 1
            else:
                status = 'SEED FAILED'
                needs_seed = True

            # In 3D mode, do not turn one transient flow/PnP failure into a
            # hard loss. A hand-held target can blur or lose a few corners for
            # a short burst; allowing a bounded reseed from the last pose lets
            # the next frame rebuild correspondences. Once the bounded window
            # is exhausted, require an independent RGB/depth recovery pose so
            # background texture cannot be accepted as metric evidence.
            # A fresh seed only rebuilds candidate correspondences; it is not
            # independent pose evidence. Clear the transient-loss counter only
            # after an actual tracked/relocalized pose is accepted.
            if status in DISPLAY_POSE_STATES and not needs_seed:
                soft_loss_frames = 0
            else:
                soft_loss_frames += 1
            if three_d_only_after_init:
                reseed_requires_pose = soft_loss_frames >= MAX_SOFT_LOSS_FRAMES

            if region_assist is not None and not visibility_rejected and not identity_mismatch:
                if outline_seeder is None:
                    from .surface_seeding import VisibleSurfaceSeeder
                    outline_seeder = VisibleSurfaceSeeder(args.model, K, dist)
                certified = (status in DISPLAY_POSE_STATES and status != 'REGION TRACKED' and
                             not verification.conflicts) or (frame_count == 0 and status == 'SEEDED')
                # A metric relocalization can jump farther than a local region
                # optimizer can follow. Rebuild its histograms at that pose.
                if certified and region_assist.previous is not None:
                    angle, shift = pose_delta(*region_assist.previous, rvec, tvec)
                    if angle > 25. or shift > 40.:
                        region_assist.initialized = False
                region_pose = region_assist.update(
                    frame, rvec, tvec, certified, vertices, K, dist, outline_seeder,
                    region_model_points, region_image_points, current_detection, min_depth, max_depth)
                # Independently gated region evidence may correct a locally
                # consistent PnP solution before its correspondences collapse.
                if region_pose is not None:
                    region_assist.stats['applied'] += 1
                    rvec, tvec = region_pose.rvec.copy(), region_pose.tvec.copy()
                    reseed_requires_pose = False
                    residual = (np.linalg.norm(project(region_model_points, rvec, tvec, K, dist) -
                                              region_image_points, axis=1)
                                if len(region_model_points) else np.empty(0))
                    keep = residual <= 3.5
                    if np.count_nonzero(keep) >= 12:
                        track_model = region_model_points[keep].copy()
                        track_image = region_image_points[keep].astype(np.float32)
                        needs_seed = False
                        inliers, rms = len(track_model), float(np.sqrt(np.mean(residual[keep]**2)))
                    else:
                        track_model, track_image = seed_points(
                            gray, vertices, edges, model_points, rvec, tvec, K, dist, args.max_points,
                            seed_radius=args.seed_radius, roi_margin=args.roi_margin,
                            surface_seeder=outline_seeder)
                        needs_seed = len(track_model) < 8
                        inliers, rms = len(track_model), 0.
                    status = 'REGION TRACKED'
                    cad_layer = None
                    # Flow and contour agreement cannot erase independent
                    # detector conflicts. Clear them only with a fresh hit.
                    if (region_assist.last_quality or {}).get('yolo_verified'):
                        verification.reset()
                    static_lock = False

            if (not force_seed and not needs_seed and not static_lock
                    and (periodic_seed or low_points)
                    and status in DISPLAY_POSE_STATES):
                operation_started = time.perf_counter()
                extra_model, extra_image = seed_points(
                    gray, vertices, edges, model_points, rvec, tvec, K, dist, args.max_points,
                    seed_radius=args.seed_radius, roi_margin=args.roi_margin,
                    surface_seeder=surface_seeder)
                for model_point, image_point in zip(extra_model, extra_image):
                    if len(track_model) >= args.max_points:
                        break
                    if len(track_image) and np.min(np.linalg.norm(
                            track_image - image_point, axis=1)) < 4.:
                        continue
                    track_model = np.vstack((track_model, model_point))
                    track_image = np.vstack((track_image, image_point)).astype(np.float32)
                seed_ms = (time.perf_counter() - operation_started) * 1000.
                timing['seed_ms'] += seed_ms
                timing['seed_calls'] += 1
                last_seed_frame = frame_count

            tracking_status = status
            if detector_future is not None and detector_future.done():
                detector_result = detector_future.result()
                relocalization_box = None
                if len(detector_result) == 3:
                    detected_box, recovery_observation, observed_axis = detector_result
                    proposed_rgb_pose = None
                elif len(detector_result) == 4:
                    (detected_box, recovery_observation, observed_axis,
                     proposed_rgb_pose) = detector_result
                else:
                    (detected_box, recovery_observation, observed_axis,
                     proposed_rgb_pose, relocalization_box) = detector_result
                # The search region may be the full image after hard loss.
                # Only a real detector box is valid for the delayed optical
                # flow handoff; otherwise let the PnP pose remain box-free.
                if relocalization_box is None:
                    relocalization_box = detected_box
                if relocalization_box is not None:
                    search_width = relocalization_box[2] - relocalization_box[0]
                    search_height = relocalization_box[3] - relocalization_box[1]
                    if (search_width >= frame.shape[1] * .95 and
                            search_height >= frame.shape[0] * .95):
                        relocalization_box = None
                checked_rvec, checked_tvec = detector_pose
                submission_translation = float(np.linalg.norm(tvec - checked_tvec))
                stale = (time.perf_counter() - detector_submitted_time > .5 or
                         frame_count - detector_submitted_frame > 12 or
                         submission_translation > 20.0 or
                         axis_line_change_deg(checked_rvec, rvec) > 10.0)
                associated = detection_associates_pose(
                    detected_box, vertices, checked_rvec, checked_tvec, K, dist, frame.shape)
                matched = associated and detection_matches_pose(
                    detected_box, vertices, checked_rvec, checked_tvec, K, dist,
                    image_shape=frame.shape)
                axis_ok = axis_matches_pose(observed_axis, checked_rvec, checked_tvec, K, dist)
                feature_verified = False
                if (not stale and not matched and associated and axis_ok
                        and recovery is not None):
                    predicted = project(vertices, checked_rvec, checked_tvec, K, dist)
                    expected_box = np.r_[predicted.min(axis=0), predicted.max(axis=0)]
                    _, contained = box_overlap(expected_box, detected_box)
                    feature_verified = (contained >= .7 and recovery.verifies_pose(
                        recovery_observation, checked_rvec, checked_tvec))
                    recovery_stats['feature_verifications'] += int(feature_verified)
                if not stale and detected_box is not None and not associated:
                    recovery_stats['unassociated_detections'] += 1
                if not stale and associated:
                    recovery_stats['box_conflicts'] += int(not matched)
                    recovery_stats['axis_conflicts'] += int(not axis_ok)
                matched = (matched or feature_verified) and axis_ok
                detector_future = None
                detector_checks += 1
                if stale:
                    pass
                elif matched:
                    detector_armed = True
                    verification.update(True)
                elif associated:
                    detector_armed = True
                    verification.update(False)
                # A missed or unassociated detection is inconclusive; do not erase an
                # already observed scale, center or orientation conflict.
                if recovery is not None:
                    recovery_stats['checks'] += 1
                    if recovery_observation is not None:
                        recovery_stats['total_ms'] += recovery_observation.elapsed_ms
                    proposal = None if stale else recovery.proposal(
                        recovery_observation, rvec, tvec, detected_box)
                    if stale:
                        recovery_stats['stale'] += 1
                    if proposal is None:
                        recovery_pending = None
                    else:
                        recovery_stats['proposals'] += 1
                        fast_accept = (getattr(args, 'fast_pose_recovery', False) and
                                       recovery.fast_recovery_ready(proposal))
                        recovery_stats['fast_proposals'] += int(fast_accept)
                        if fast_accept or recovery.consistent(recovery_pending, proposal):
                            candidate_rvec, candidate_tvec = recovery.recover(proposal)
                            candidate_box_ok = detected_box is None or detection_matches_pose(
                                detected_box, vertices, candidate_rvec, candidate_tvec, K, dist,
                                image_shape=frame.shape)
                            candidate_box_ok = candidate_box_ok and axis_matches_pose(
                                observed_axis, candidate_rvec, candidate_tvec, K, dist)
                            if min_depth <= candidate_tvec[2] <= max_depth and candidate_box_ok:
                                rvec, tvec = candidate_rvec, candidate_tvec
                                reseed_requires_pose = False
                                needs_seed = True
                                track_model = np.empty((0, 3), np.float64)
                                track_image = np.empty((0, 2), np.float32)
                                cad_layer = None
                                status = ('POSE CORRECTED' if fast_accept and
                                          three_d_only_after_init else 'RGB RECOVERED')
                                static_lock = False
                                static_pending = None
                                static_lock_anchor_image = np.empty((0, 2), np.float32)
                                verification.reset()
                                recovery_stats['accepted'] += 1
                                recovery_stats['fast_accepted'] += int(fast_accept)
                            else:
                                recovery_stats['stale'] += 1
                            recovery_pending = None
                        else:
                            recovery_pending = proposal
                if rgb_relocalizer is not None:
                    # Fresh metric PnP does not inherit the old tracked pose's
                    # motion-based stale gate. Its own pixels must reach this
                    # current image successfully, within the age/frame limit.
                    age_ok = (time.perf_counter() - detector_submitted_time <= .5
                              and frame_count - detector_submitted_frame <= 12)
                    fast_recovery = status not in DISPLAY_POSE_STATES or needs_seed
                    candidate = rgb_relocalizer.confirm(
                        proposed_rgb_pose if age_ok else None, detector_frame, frame,
                        relocalization_box, time.perf_counter(), rvec, tvec,
                        status in DISPLAY_POSE_STATES and not verification.conflicts,
                        allow_single_frame=fast_recovery)
                    if candidate is not None and min_depth <= candidate.tvec[2] <= max_depth:
                        rvec, tvec = candidate.rvec.copy(), candidate.tvec.copy()
                        # These are measured correspondences in the CURRENT
                        # frame; use them on the next frame without reseeding.
                        track_model, track_image = candidate.points.copy(), candidate.pixels.copy()
                        reseed_requires_pose = False
                        needs_seed = False
                        status = ('POSE RELOCALIZED' if three_d_only_after_init
                                  else 'RGB RECOVERED')
                        inliers, rms = len(track_model), candidate.rms_px
                        static_lock = False
                        static_pending = None
                        static_lock_anchor_image = np.empty((0, 2), np.float32)
                        recovery_pending = None
                        verification.reset()
                        rgb_relocalizer.stats['accepted'] += 1
                        rgb_relocalizer.pending = None
            detector_interval = target_check_interval(
                getattr(args, 'detector_interval', DEFAULT_DETECTOR_INTERVAL),
                status, flow_median, flow_p90, verification.conflicts,
                recovery_pending is not None,
                rgb_relocalizer is not None and rgb_relocalizer.pending is not None,
                needs_seed)
            if detector_executor is not None and detector_future is None and \
                    frame_count % detector_interval == 0:
                detector_pose = (rvec.copy(), tvec.copy())
                detector_submitted_frame = frame_count
                detector_submitted_time = time.perf_counter()
                detector_frame = frame.copy()
                predicted = project(vertices, rvec, tvec, K, dist)
                expected_box = np.r_[predicted.min(axis=0), predicted.max(axis=0)]
                hard_loss = status not in DISPLAY_POSE_STATES or needs_seed
                if hard_loss and detector_checks % 6 != 5:
                    fallback_rotation = 0
                else:
                    fallback_rotation = ((detector_checks % 3 + 1)
                                         if getattr(args, 'detector_rotation', True) else 0)
                detector_future = detector_executor.submit(
                    check_target, frame, detector, recovery,
                    fallback_rotation, expected_box, rgb_relocalizer,
                    expected_box if hard_loss else None, hard_loss,
                    hard_loss and rgb_relocalizer is not None,
                    hard_loss)
            if (status in DISPLAY_POSE_STATES and current_detection is not None and
                    not current_detection_agrees(frame, current_detection, vertices,
                                                  rvec, tvec, K, dist)):
                current_detection_conflicts += 1
                status = 'VERIFYING POSE'
            if three_d_only_after_init and recovery_pending is not None and status != 'VERIFYING POSE':
                status = 'POSE CHECK'
            elif verification.latched or (recovery_pending is not None and
                                           status not in DISPLAY_POSE_STATES):
                status = 'VERIFYING POSE'
            if detector_armed and verification.conflicts >= 3:
                reseed_requires_pose = three_d_only_after_init
                status = 'TARGET LOST'
                needs_seed = True
                track_model = np.empty((0, 3), np.float64)
                track_image = np.empty((0, 2), np.float32)
                reacquire_requested = reacquire_on_loss
                recovery_stats['depth_requests'] += int(reacquire_on_loss)
                if reacquire_on_loss:
                    failure_image = args.summary.with_name('pose_verification_failed.png')
                    failure_image.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode('.png', frame)[1].tofile(failure_image)
                    summary['verification_failure_image'] = str(failure_image.resolve())

            max_axis_change = getattr(args, 'max_axis_change_from_depth_deg', 0.0)
            if (reacquire_on_loss and max_axis_change > 0 and
                    status in DISPLAY_POSE_STATES and
                    axis_line_change_deg(pose_before_frame, rvec) >= max_axis_change):
                status = 'REACQUIRE DEPTH'
                needs_seed = True
                track_model = np.empty((0, 3), np.float64)
                track_image = np.empty((0, 2), np.float32)
                reacquire_requested = True

            if status in DISPLAY_POSE_STATES:
                lost_frames = 0
            else:
                lost_frames += 1
            if reacquire_on_loss and lost_frames >= getattr(args, 'reacquire_lost_frames', 15):
                status = 'TARGET LOST'
                reacquire_requested = True

            if static_lock and status not in {'STATIC LOCKED', 'STATIC CORRECTED'}:
                if status == 'FLOW LOST':
                    status = 'STATIC LOST'
                static_lock = False
                static_lock_shift[:] = 0.0
                static_lock_motion = 0.0
                static_lock_anchor_image = np.empty((0, 2), np.float32)
                static_match_failures = 0

            static_ready_inliers = (STATIC_MIN_LOCK_INLIERS if static_lock else
                                    STATIC_MIN_READY_INLIERS)
            static_ready = status in {'TRACKING', 'STATIC LOCKED'} and \
                inliers >= static_ready_inliers and \
                flow_median <= args.static_flow_median and flow_p90 <= args.static_flow_p90 and \
                frame_count - last_failure_frame >= args.static_min_frames
            if static_ready:
                static_frames += 1
            else:
                static_frames = 0
                if status not in {'TRACKING', 'SEEDED', 'STATIC CORRECTED', 'STATIC LOCKED'}:
                    last_failure_frame = frame_count
                if static_future is None:
                    static_pending = None

            if static_future is not None and static_future.done():
                try:
                    observation = static_future.result()
                except (cv2.error, RuntimeError, ValueError):
                    observation = None
                submitted_rvec, submitted_tvec = static_submission
                static_future = None
                static_submission = None
                if observation is None:
                    static_stats['rejected'] += 1
                    static_pending = None
                    if static_lock:
                        static_match_failures += 1
                    static_action = 'STATIC NO MATCH'
                    static_detail = 'static: no confirmed-reference match'
                else:
                    static_obs_matches = observation.match_count
                    static_obs_inliers = observation.inliers
                    static_obs_ratio = observation.inlier_ratio
                    static_obs_spread = observation.spread_px2
                    static_elapsed_ms = observation.elapsed_ms
                    static_stats['total_ms'] += observation.elapsed_ms
                    submitted_rotation, submitted_translation = pose_delta(
                        submitted_rvec, submitted_tvec, rvec, tvec)
                    stale = not static_ready or submitted_rotation > 1.0 or submitted_translation > 2.0
                    center_delta, angle_delta, scale_ratio = static_corrector.residual(
                        observation, rvec, tvec)
                    static_center_error = float(np.linalg.norm(center_delta))
                    static_angle_error = float(angle_delta)
                    static_scale_ratio = float(scale_ratio)
                    confidence_ok = observation.inliers >= STATIC_MIN_MATCH_INLIERS and \
                        observation.inlier_ratio >= STATIC_MIN_INLIER_RATIO and \
                        observation.spread_px2 >= STATIC_MIN_SPREAD_PX2
                    residual_ok = static_center_error <= STATIC_MAX_CENTER_ERROR_PX and \
                        abs(static_angle_error) <= STATIC_MAX_ANGLE_ERROR_DEG and \
                        abs(static_scale_ratio - 1.0) <= STATIC_MAX_SCALE_ERROR
                    if stale:
                        static_stats['stale'] += 1
                        static_pending = None
                        static_action = 'STATIC STALE'
                        static_detail = 'static: proposal discarded after motion'
                    elif not confidence_ok or not residual_ok:
                        static_stats['rejected'] += 1
                        static_pending = None
                        if static_lock:
                            static_match_failures += 1
                        static_action = 'STATIC REJECTED'
                        static_detail = f'static: rejected ({observation.inliers} matches)'
                    elif static_center_error < .75 and abs(static_angle_error) < .25 and \
                            abs(static_scale_ratio - 1.0) < .0075:
                        static_match_failures = 0
                        if static_corrector.consistent(static_pending, observation):
                            was_locked = static_lock
                            static_pending = None
                            static_lock = True
                            static_lock_shift[:] = 0.0
                            static_lock_motion = 0.0
                            static_lock_anchor_image = track_image.copy()
                            status = 'STATIC LOCKED'
                            if was_locked:
                                static_action = 'STATIC ALIGNED'
                                static_detail = 'static: alignment confirmed'
                            else:
                                static_stats['locks'] += 1
                                static_action = 'STATIC LOCKED'
                                static_detail = 'static: aligned and locked'
                        else:
                            static_pending = observation
                            static_stats['proposals'] += 1
                            static_action = 'STATIC PROPOSAL'
                            static_detail = f'static: confirming alignment ({observation.inliers} matches)'
                    elif static_corrector.consistent(static_pending, observation):
                        static_match_failures = 0
                        old_rvec, old_tvec = rvec.copy(), tvec.copy()
                        rvec, tvec, diagnostics = static_corrector.apply(
                            observation, rvec, tvec, gain=args.static_gain,
                            max_center_step_px=args.static_max_center_step,
                            max_angle_step_deg=args.static_max_angle_step,
                            max_depth_step_mm=args.static_max_depth_step)
                        if not min_depth <= tvec[2] <= max_depth:
                            rvec, tvec = old_rvec, old_tvec
                            static_stats['rejected'] += 1
                            static_action = 'STATIC REJECTED'
                            static_detail = 'static: depth correction rejected'
                        else:
                            static_lock = True
                            static_match_failures = 0
                            static_lock_shift[:] = 0.0
                            static_lock_motion = 0.0
                            static_lock_anchor_image = track_image.copy()
                            needs_seed = False
                            cad_layer = None
                            status = 'STATIC CORRECTED'
                            static_stats['accepted'] += 1
                            static_action = 'STATIC CORRECTED'
                            static_detail = (f"static: corrected {diagnostics['center_step_px']:.1f}px / "
                                             f"{diagnostics['angle_step_deg']:.1f}deg / "
                                             f"{diagnostics['depth_step_mm']:.1f}mm")
                            static_frames = 0
                        static_pending = None
                    else:
                        static_match_failures = 0
                        static_pending = observation
                        static_stats['proposals'] += 1
                        static_action = 'STATIC PROPOSAL'
                        static_detail = f'static: confirming reference ({observation.inliers} matches)'

                if static_lock and static_match_failures >= 2:
                    static_lock = False
                    static_lock_shift[:] = 0.0
                    static_lock_motion = 0.0
                    static_lock_anchor_image = np.empty((0, 2), np.float32)
                    static_match_failures = 0
                    static_pending = None
                    needs_seed = True
                    track_model = np.empty((0, 3), np.float64)
                    track_image = np.empty((0, 2), np.float32)
                    cad_layer = None
                    status = 'STATIC LOST'
                    static_action = 'STATIC LOST'
                    static_detail = 'static: reference lost, reseeding'
                    static_frames = 0
                    last_failure_frame = frame_count

            if static_corrector is not None and static_future is None and static_ready and \
                    static_frames >= args.static_min_frames and \
                    frame_count - last_static_check >= args.static_check_interval:
                search_mask = make_search_mask(gray.shape, vertices, rvec, tvec, K, dist,
                                               args.static_search_margin)
                static_future = static_executor.submit(static_corrector.observe, gray.copy(), search_mask)
                static_submission = (rvec.copy(), tvec.copy())
                last_static_check = frame_count
                static_stats['checks'] += 1
                static_action = 'STATIC CHECK'
                static_detail = 'static: checking confirmed reference'

            if static_corrector is not None and not static_action:
                if static_frames:
                    static_detail = f'static: holding {min(static_frames, args.static_min_frames)}/{args.static_min_frames}'
                else:
                    static_detail = 'static: waiting for still object'
            xyz_record = make_xyz_record(frame_count, time.perf_counter() - start_time,
                                         status, rvec, tvec)
            display_status, display_pose, pose_held = presentation.update(
                status, time.perf_counter() - start_time, rvec, tvec)
            box_visible = display_pose is not None
            display_transitions += int(previous_display_status is not None and
                                       display_status != previous_display_status)
            box_visibility_changes += int(previous_box_visible is not None and
                                          box_visible != previous_box_visible)
            previous_display_status, previous_box_visible = display_status, box_visible
            display_counts[display_status] = display_counts.get(display_status, 0) + 1
            display = None
            key = 255
            if not headless:
                draw_started = time.perf_counter()
                if not box_visible:
                    cad_layer = None
                else:
                    display_rvec, display_tvec = display_pose
                    display_offset = getattr(args, 'box_display_y_offset_px',
                                             BOX_DISPLAY_Y_OFFSET_PX)
                    cad_layer = np.zeros_like(frame)
                    if getattr(args, 'show_box', False):
                        cad_layer = render_box_layer(
                            frame.shape, box_corners, K, dist, display_rvec, display_tvec,
                            display_offset,
                            neutralize_roll=getattr(args, 'box_roll_mode', 'true') == 'neutral')
                    if cad_outline and outline_seeder is not None:
                        contour_layer = render_cad_pose_layer(
                            frame.shape, vertices, K, dist, display_rvec, display_tvec,
                            outline_seeder, display_offset,
                            show_axes=getattr(args, 'pose_axes', True))
                        cad_layer = cv2.add(cad_layer, contour_layer)
                    timing['draw_calls'] += 1
                visible_layer = cad_layer if cad_layer is not None else np.zeros_like(frame)
                display = draw_status(frame, visible_layer, display_status, inliers, rms, fps,
                                      static_detail,
                                      display_pose[1] if box_visible else None, pose_held)
                draw_ms = (time.perf_counter() - draw_started) * 1000.0
                timing['draw_ms'] += draw_ms
                if (getattr(args, 'show_box', False) and
                        getattr(args, 'box_roll_mode', 'true') == 'neutral'):
                    cv2.putText(display, 'DISPLAY: roll neutral (not measured roll)',
                                (10, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                .45, (40, 220, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, collection_display(cap, display))
                key = collection_key(cap, cv2.waitKey(1) & 0xFF)
            diagnostic_record = dict(
                xyz_record, rvec_rad=np.asarray(rvec).reshape(3).tolist(),
                tvec_mm=np.asarray(tvec).reshape(3).tolist(),
                inliers=int(inliers), rms_px=float(rms) if np.isfinite(rms) else None,
                display_status=display_status, pose_held=bool(pose_held),
                display_pose=(dict(rvec_rad=np.asarray(display_pose[0]).reshape(3).tolist(),
                                   tvec_mm=np.asarray(display_pose[1]).reshape(3).tolist())
                              if display_pose is not None else None),
                box_roll_mode=getattr(args, 'box_roll_mode', 'true'),
                show_box=getattr(args, 'show_box', False),
                pose_axes=getattr(args, 'pose_axes', True),
                display_y_offset_px=getattr(args, 'box_display_y_offset_px', BOX_DISPLAY_Y_OFFSET_PX),
                box_display_y_offset_px=getattr(args, 'box_display_y_offset_px', BOX_DISPLAY_Y_OFFSET_PX),
                cad_outline=bool(cad_outline), source_elapsed_s=source_elapsed_s)
            diagnostic_capture.write(frame, diagnostic_record)
            collection = getattr(cap, 'collection', None)
            if collection is not None:
                collection.pose(diagnostic_record)
            now = time.perf_counter()
            frame_ms = (now - loop_started) * 1000.0
            instant = 1.0 / max(now - last_time, 1e-6)
            fps = .9 * fps + .1 * instant if fps else instant
            last_time = now
            fps_samples.append(fps)
            if log_writer is not None:
                log_writer.writerow([frame_count, round(now - start_time, 4),
                                     status, len(track_model), inliers,
                                     round(rms, 4), round(fps, 3), *np.round(rvec, 8).tolist(),
                                     *np.round(tvec, 5).tolist(), round(frame_ms, 3), round(seed_ms, 3),
                                     round(track_ms, 3), round(draw_ms, 3), round(flow_median, 4),
                                     round(flow_p90, 4), round(float(flow_vector[0]), 4),
                                     round(float(flow_vector[1]), 4),
                                     round(static_unlock_shift if static_unlock_shift is not None else
                                           float(np.linalg.norm(static_lock_shift)), 4),
                                     round(static_unlock_motion if static_unlock_motion is not None else
                                           static_lock_motion, 4),
                                     static_action, static_obs_matches, static_obs_inliers,
                                     round(static_obs_ratio, 4), round(static_obs_spread, 2),
                                     round(static_center_error, 4), round(static_angle_error, 4),
                                     round(static_scale_ratio, 6), round(static_elapsed_ms, 3),
                                     round(source_elapsed_s, 6) if source_elapsed_s is not None else '',
                                     round(coverage, 4), appearance_inliers,
                                     round(appearance_center_error, 3)
                                      if appearance_center_error is not None else '', session_action,
                                      tracking_status, verification.conflicts,
                                      recovery_pending is not None, display_status,
                                      box_visible, pose_held, round(capture_ms, 3)])
                if frame_count % 30 == 0:
                    log_handle.flush()
            if xyz_handle is not None:
                xyz_handle.write(json.dumps(xyz_record, allow_nan=False) + '\n')
                if frame_count % 30 == 0:
                    xyz_handle.flush()
                if video is None and xyz_record['valid'] and frame_count % 30 == 0:
                    x, y, z = xyz_record['center_xyz_mm']
                    print(f'frame {frame_count:06d} | RGB camera XYZ (mm): '
                          f'X={x:+.1f} Y={y:+.1f} Z={z:+.1f}', flush=True)
            summary['frames'] = frame_count + 1
            summary['tracking_frames'] += int(status in DISPLAY_POSE_STATES)
            if source_elapsed_s is not None:
                summary['source_duration_s'] = round(source_elapsed_s, 6)
            summary['low_quality_frames'] += int(
                status not in DISPLAY_POSE_STATES and status != 'SEEDED')
            summary['max_inliers'] = max(summary['max_inliers'], int(inliers))
            summary['max_fps'] = max(summary['max_fps'], float(fps))
            status_counts[status] = status_counts.get(status, 0) + 1
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                needs_seed = True
                static_lock = False
                static_lock_shift[:] = 0.0
                static_lock_motion = 0.0
                static_lock_anchor_image = np.empty((0, 2), np.float32)
                static_match_failures = 0
                static_pending = None
                track_model = np.empty((0, 3), np.float64)
                status = 'RESEEDING'
            if key in (ord('d'), ord('D')) and reacquire_on_loss:
                reacquire_requested = True
                status = 'REACQUIRING'
            if key in (ord('c'), ord('C')):
                if outline_seeder is None:
                    from .surface_seeding import VisibleSurfaceSeeder
                    outline_seeder = VisibleSurfaceSeeder(args.model, K, dist)
                cad_outline = not cad_outline
            if key in (ord('s'), ord('S')):
                snapshot = diagnostic_capture.snapshot(frame, diagnostic_record, display)
                print(f'Saved synchronized snapshot: {snapshot}', flush=True)
                if xyz_record['valid']:
                    save_pose(args.output, rvec, tvec, display)
                else:
                    print('No valid pose to save; tracking is unavailable.')
            previous_gray = gray
            frame_count += 1
            if reacquire_requested:
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            if not headless and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if region_executor is not None:
            region_executor.shutdown(wait=True, cancel_futures=True)
            if region_future.done() and not region_future.cancelled() and region_future.exception() is None:
                region_future.result().close()
        if detector_executor is not None:
            detector_executor.shutdown(wait=True, cancel_futures=True)
        if static_executor is not None:
            static_executor.shutdown(wait=True, cancel_futures=True)
        if owns_detector and detector is not None and hasattr(detector, 'close'):
            detector.close()
        if not (keep_capture_on_reacquire and reacquire_requested):
            cap.release()
        if not headless and not keep_window:
            cv2.destroyAllWindows()
        ended = time.perf_counter()
        duration = max(ended - start_time, 1e-6)
        summary['ended_at'] = datetime.now().isoformat(timespec='seconds')
        summary['duration_s'] = round(duration, 3)
        summary['average_fps'] = round(summary['frames'] / duration, 3)
        summary['median_smoothed_fps'] = round(float(np.median(fps_samples)), 3) if fps_samples else 0.0
        summary['status_counts'] = status_counts
        summary['display'] = dict(status_counts=display_counts,
                                  state_changes=display_transitions,
                                  box_visibility_changes=box_visibility_changes,
                                  hold_max_s=presentation.grace_s,
                                  hold_max_frames=presentation.max_hold_frames)
        summary['reacquire_requested'] = reacquire_requested
        summary['m3t_region'] = region_assist.stats if region_assist is not None else None
        summary['current_detection_conflicts'] = current_detection_conflicts
        summary['detector_armed'] = detector_armed
        summary['background_yolo'] = (dict(**detector.stats, error=detector.error)
                                      if hasattr(detector, 'get_box') else None)
        summary['rgb_recovery'] = (dict(
            **recovery_stats,
            mean_ms=round(recovery_stats['total_ms'] / max(recovery_stats['checks'], 1), 3))
            if recovery is not None else None)
        summary['rgb_relocalization'] = (rgb_relocalizer.stats if rgb_relocalizer is not None
                                          else None)
        summary['session_appearance'] = session_stats if session_corrector is not None else None
        summary['video_reached_eof'] = video_eof if video is not None else None
        end_depth_error = (abs(float(tvec[2]) - session_bounds['end_nominal_mm'])
                           if session_bounds and video_eof else None)
        summary['end_nominal_depth_error_mm'] = (round(end_depth_error, 3)
                                                 if end_depth_error is not None else None)
        summary['end_nominal_depth_check_passed'] = (
            end_depth_error <= session_bounds['margin_mm']
            if end_depth_error is not None else None)
        diagnostic_capture.close()
        summary['diagnostics'] = dict(directory=str(diagnostic_dir.resolve()),
                                      recorded_frames=len(diagnostic_capture.timestamps))
        summary['final_pose'] = dict(rvec=rvec.tolist(), tvec_mm=tvec.tolist(),
                                     status=status if summary['frames'] else None,
                                     tracking_valid=bool(summary['frames'] and status in
                                                         DISPLAY_POSE_STATES))
        summary['static_correction'] = dict(
            **static_stats,
            mean_ms=round(static_stats['total_ms'] / max(static_stats['checks'], 1), 3))
        summary['timing_ms'] = dict(
            mean_capture=round(timing['capture_ms'] / max(summary['frames'], 1), 3),
            mean_seed=round(timing['seed_ms'] / max(timing['seed_calls'], 1), 3),
            mean_track=round(timing['track_ms'] / max(timing['track_calls'], 1), 3),
            mean_display=round(timing['draw_ms'] / max(summary['frames'], 1), 3),
            seed_calls=timing['seed_calls'], track_calls=timing['track_calls'],
            cad_layer_updates=timing['draw_calls'])
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
        if xyz_handle is not None:
            xyz_handle.flush()
            xyz_handle.close()
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return REACQUIRE_EXIT_CODE if reacquire_requested else 0


def validate_args(parser, args):
    checks = [
        (not np.isfinite(args.diagnostic_duration_s) or args.diagnostic_duration_s <= 0,
         '--diagnostic-duration-s must be finite and positive'),
        (args.video is not None and args.camera is not None,
         '--video and --camera cannot be used together'),
        (args.session_depth_margin < 0, '--session-depth-margin must be non-negative'),
        (not np.isfinite(args.box_display_y_offset_px) or
         abs(args.box_display_y_offset_px) > 100,
         '--box-display-y-offset-px must be finite and within +/-100'),
        (not 0 <= args.session_min_coverage <= 1,
         '--session-min-coverage must be between 0 and 1'),
        (args.camera is not None and args.camera < 0, '--camera must be non-negative'),
        (args.camera_fps < 0, '--camera-fps must be non-negative'),
        (args.camera_fourcc and len(args.camera_fourcc) != 4,
         '--camera-fourcc must contain exactly four characters'),
        (args.max_frames < 0, '--max-frames must be non-negative'),
        (args.max_edges < 1, '--max-edges must be at least 1'),
        (args.edge_samples < 2, '--edge-samples must be at least 2'),
        (args.max_points < 8, '--max-points must be at least 8'),
        (args.refresh < 0, '--refresh must be non-negative'),
        (args.reseed_below < 0 or args.reseed_below > args.max_points,
         '--reseed-below must be between 0 and --max-points'),
        (args.reacquire_lost_frames < 1,
         '--reacquire-lost-frames must be at least 1'),
        (args.detector_interval < 1,
         '--detector-interval must be at least 1'),
        (not 0 <= args.max_axis_change_from_depth_deg <= 90,
         '--max-axis-change-from-depth-deg must be in [0, 90]'),
        (args.draw_interval < 1, '--draw-interval must be at least 1'),
        (args.roi_margin < 0, '--roi-margin must be non-negative'),
        (args.seed_radius <= 0, '--seed-radius must be positive'),
        (args.flow_error <= 0, '--flow-error must be positive'),
        (args.fb_error < 0, '--fb-error must be non-negative'),
        (args.pnp_error <= 0, '--pnp-error must be positive'),
        (args.pnp_iterations < 1, '--pnp-iterations must be at least 1'),
        (args.max_rms <= 0, '--max-rms must be positive'),
        (args.min_inliers < 6 or args.min_inliers > args.max_points,
         '--min-inliers must be between 6 and --max-points'),
        (not 0 <= args.min_inlier_ratio <= 1,
         '--min-inlier-ratio must be between 0 and 1'),
        (args.max_rotation_step <= 0 or args.max_rotation_step > 180,
         '--max-rotation-step must be in (0, 180]'),
        (args.max_translation_step <= 0, '--max-translation-step must be positive'),
        (args.min_depth <= 0 or args.max_depth <= args.min_depth,
         '--min-depth must be positive and less than --max-depth'),
        (args.static_min_frames < 1, '--static-min-frames must be at least 1'),
        (args.static_check_interval < 1, '--static-check-interval must be at least 1'),
        (args.static_flow_median < 0 or args.static_flow_p90 < args.static_flow_median,
         '--static flow thresholds must be non-negative and p90 must be at least the median'),
        (args.static_search_margin < 0, '--static-search-margin must be non-negative'),
        (not 0 < args.static_gain <= 1, '--static-gain must be in (0, 1]'),
        (args.static_max_center_step <= 0, '--static-max-center-step must be positive'),
        (args.static_max_depth_step <= 0, '--static-max-depth-step must be positive'),
        (args.static_max_angle_step <= 0, '--static-max-angle-step must be positive'),
    ]
    errors = [message for failed, message in checks if failed]
    if errors:
        parser.error('; '.join(errors))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path,
                        help='Replay a recorded RGB video instead of opening the camera')
    parser.add_argument('--region-tracking', action=argparse.BooleanOptionalAction, default=False,
                        help='Validate persistent M3T contour tracking against current image evidence')
    parser.add_argument('--headless', action='store_true',
                        help='Process without a display window; useful for recorded clips')
    parser.add_argument('--session-depth-margin', type=float, default=100.0,
                        help='Depth tolerance around accepted paired RGB-D anchor distances, in mm')
    parser.add_argument('--box-display-y-offset-px', type=float,
                        help='Display-only downward shift of the contour, axes and optional box; '
                             'defaults to saved setting')
    parser.add_argument('--box-roll-mode', choices=('neutral', 'true'),
                        default='true',
                        help='true displays measured 6DoF; neutral is a cosmetic roll override')
    parser.add_argument('--cad-outline', action=argparse.BooleanOptionalAction, default=True,
                        help='Draw the CAD silhouette in green; C toggles it and its axes')
    parser.add_argument('--show-box', action=argparse.BooleanOptionalAction, default=False,
                        help='Also show the optional yellow model bounding box')
    parser.add_argument('--pose-axes', action=argparse.BooleanOptionalAction, default=True,
                        help='Draw model X/Y/Z axes with the contour; Z follows the long axis')
    parser.add_argument('--record-diagnostics', action='store_true',
                        help='Save bounded raw RGB video and synchronized poses, including losses')
    parser.add_argument('--diagnostic-duration-s', type=float, default=60.)
    parser.add_argument('--session-depth-gate', action=argparse.BooleanOptionalAction,
                        default=True, help='Limit drift using accepted paired RGB-D anchors')
    parser.add_argument('--session-relocalization', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Check the paired start RGB view and relocalize on a confident match')
    parser.add_argument('--session-min-coverage', type=float, default=.95,
                        help='Minimum CAD projection coverage when appearance is unverified')
    parser.add_argument('--reacquire-on-loss', action='store_true',
                        help='Request a new depth initialization after sustained tracking loss; D requests it manually')
    parser.add_argument('--depth-fallback-on-loss', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Allow the live RGB-D wrapper to request depth again after RGB relocalization fails')
    parser.add_argument('--three-d-only-after-init', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='After the initial pose handoff, use only 3D CAD tracking and never fall back to YOLO')
    parser.add_argument('--reacquire-lost-frames', type=int, default=15)
    parser.add_argument('--max-axis-change-from-depth-deg', type=float, default=0.0,
                        help='Reinitialize depth when the tracked long-axis line changes this much; 0 disables')
    parser.add_argument('--reacquire-model', type=Path,
                        help='Optional object detector used to invalidate a stale RGB pose')
    parser.add_argument('--detector-interval', type=int, default=DEFAULT_DETECTOR_INTERVAL,
                        help='Run the background detector every N RGB frames')
    parser.add_argument('--rgb-pose-recovery', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Use the depth initialization RGB image to recover a drifted pose')
    parser.add_argument('--fast-pose-recovery', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Accept one-frame RGB correction only when reference matches are strong')
    parser.add_argument('--rgb-recovery-reference', type=Path,
                        help='RGB image paired with --pose; defaults to pose JSON rgb_image')
    parser.add_argument('--detector-rotation', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Cycle rotated RGB views when the normal detector misses')
    parser.add_argument('--camera', type=int, default=None,
                        help='Manual camera index; when set, overrides --camera-name')
    parser.add_argument('--camera-name', type=str, default='USB Camera',
                        help='Exact DirectShow camera name; empty string uses --camera index')
    parser.add_argument('--camera-fps', type=float, default=30.0)
    parser.add_argument('--camera-fourcc', type=str, default='MJPG',
                        help='FourCC requested before resolution setup; empty string leaves the driver default')
    parser.add_argument('--pose', type=Path, default=DEFAULT_POSE,
                        help='Initial CAD pose JSON, including a depth_pose_handoff RGB pose')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--intrinsics', type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'tracking/last_pose.json')
    parser.add_argument('--log', type=Path, default=None,
                        help='CSV pose/status log')
    parser.add_argument('--xyz-jsonl', type=Path,
                        help='Write one RGB camera XYZ pose record per frame; invalid poses are null')
    parser.add_argument('--summary', type=Path, default=None)
    parser.add_argument('--no-log', action='store_true', help='Disable automatic CSV logging')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Stop after N frames; 0 runs until video EOF or the window is closed')
    parser.add_argument('--max-edges', type=int, default=700)
    parser.add_argument('--edge-samples', type=int, default=3)
    parser.add_argument('--max-points', type=int, default=100)
    parser.add_argument('--refresh', type=int, default=0,
                        help='Maximum frames between reseeds; 0 disables periodic reseeding')
    parser.add_argument('--reseed-below', type=int, default=12,
                        help='Reseed when fewer than this many correspondences remain')
    parser.add_argument('--draw-interval', type=int, default=2,
                        help='Compatibility setting; the lightweight box now updates every frame')
    parser.add_argument('--roi-margin', type=int, default=40,
                        help='Pixel margin around the projected CAD tracking ROI')
    parser.add_argument('--seed-radius', type=float, default=8.0,
                        help='Maximum pixels between a seed corner and projected CAD edge')
    parser.add_argument('--surface-seeding', action=argparse.BooleanOptionalAction,
                        default=False, help='Bind features to the visible CAD surface using ray casting')
    parser.add_argument('--flow-error', type=float, default=20.0)
    parser.add_argument('--fb-error', type=float, default=1.5,
                        help='Maximum forward-backward optical-flow error; 0 disables the check')
    parser.add_argument('--pnp-error', type=float, default=5.0)
    parser.add_argument('--pnp-iterations', type=int, default=60)
    parser.add_argument('--max-rms', type=float, default=3.5,
                        help='Reject a PnP update when inlier RMS exceeds this many pixels')
    parser.add_argument('--min-inliers', type=int, default=12)
    parser.add_argument('--min-inlier-ratio', type=float, default=.30)
    parser.add_argument('--max-rotation-step', type=float, default=12.0,
                        help='Reject a single-frame long-axis change above this many degrees; ignores equivalent roll')
    parser.add_argument('--max-translation-step', type=float, default=15.0,
                        help='Reject a single-frame translation jump above this many millimetres')
    parser.add_argument('--min-depth', type=float, default=80.0)
    parser.add_argument('--max-depth', type=float, default=2400.0)
    parser.add_argument('--static-correction', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable conservative corrections from the confirmed reference view')
    parser.add_argument('--static-reference-image', type=Path,
                        default=DEFAULT_STATIC_REFERENCE_IMAGE)
    parser.add_argument('--static-reference-pose', type=Path,
                        default=DEFAULT_STATIC_REFERENCE_POSE)
    parser.add_argument('--static-min-frames', type=int, default=20)
    parser.add_argument('--static-check-interval', type=int, default=15)
    parser.add_argument('--static-flow-median', type=float, default=.35)
    parser.add_argument('--static-flow-p90', type=float, default=.8)
    parser.add_argument('--static-search-margin', type=int, default=80)
    parser.add_argument('--static-gain', type=float, default=.25)
    parser.add_argument('--static-max-center-step', type=float, default=5.0)
    parser.add_argument('--static-max-depth-step', type=float, default=5.0)
    parser.add_argument('--static-max-angle-step', type=float, default=1.0)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.box_display_y_offset_px is None:
        try:
            args.box_display_y_offset_px = load_box_display_y_offset()
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    validate_args(parser, args)
    return args


def main():
    args = parse_args()
    try:
        return run(args)
    except (RuntimeError, OSError, cv2.error) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc


if __name__ == '__main__':
    raise SystemExit(main())
