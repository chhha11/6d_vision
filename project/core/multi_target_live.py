"""Multi-instance RGB detection with independent, verified RGB/CAD poses."""
from __future__ import annotations

import colorsys
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .background_detector import BackgroundDetector, advance_box
from .cad_video_tracker import BOX_SIGNS, box_overlap, detection_matches_pose, load_video_timestamps
from .ego_motion import IdentityEgoMotion
from .multi_target_detection import detect_target_boxes, detect_live_target_boxes
from .multi_target_tracking import replenish_pose, needs_replenishment, advance_pose_trail, reference_improves_pose
from .frame_trail import FrameTrail
from .pose_visualization import (display_edge_distance, estimate_display_affine,
                                  render_cad_pose_layer)
from .rgb_pose_recovery import (directed_long_axis_jump_deg,
                                    pose_has_continuous_long_axis)
from .surface_seeding import VisibleSurfaceSeeder
from .tracking_diagnostics import TrackingDiagnostics


@dataclass
class Target:
    id: int
    box: np.ndarray
    last_seen: float
    visible: bool = True
    pose: object = None
    search: object = None
    detected: bool = True
    gap_frames: int = 0
    last_good_pose: object = None
    pose_source: object = None
    pose_box: object = None
    pose_time: float = -float('inf')
    display_pose: object = None
    # Most recent detector extent in the current frame. It is retained as an
    # association/diagnostic hint and never changes the metric pose.
    observed_box: object = None
    # Keep detector measurement, optical-flow prediction and the box used by
    # association/display as separate state. ``box`` remains the legacy
    # compatibility alias for the current filtered/association box.
    raw_detection_box: object = None
    predicted_box: object = None
    filtered_box: object = None
    # CAD projection in the current image.  This is deliberately separate
    # from ``filtered_box``: the latter is the detector/flow association ROI.
    pose_projection_box: object = None
    box_source: str = "detection"
    pose_status: str = "detected"
    pose_prediction_frames: int = 0
    pose_prediction_time: float = -float('inf')
    pose_frame: str = "camera"
    ego_motion_source: str = "identity"
    display_affine: object = None
    display_refit_at: int = -20
    display_motion_px: float = 0.
    display_rotation_deg: float = 0.
    display_held: bool = False
    # A display-only edge fit is allowed to change while moving, but a
    # stationary target must see the same correction repeatedly before it is
    # allowed to replace the locked contour alignment.
    display_candidate_affine: object = None
    display_candidate_hits: int = 0
    failed_frames: int = 0
    tracked_frames: int = 0
    reference_checked_at: int = 0
    seeded_at: int = -20
    seeded_rvec: object = None
    tracking_failure: str | None = None
    initialization_failure: str | None = None
    # Directed axis continuity is a metric-pose diagnostic.  A value near
    # 180 degrees means the latest candidate came from the opposite PnP
    # branch, even when its reprojection RMS is small.
    long_axis_jump_deg: object = None

    def clear_pose(self):
        self.pose = self.last_good_pose = self.display_pose = None
        self.pose_source = self.pose_box = self.search = None
        self.observed_box = None
        self.pose_projection_box = None
        self.display_affine = None
        self.display_refit_at = -20
        self.display_motion_px = 0.
        self.display_rotation_deg = 0.
        self.pose_time = -float('inf')
        self.pose_prediction_time = -float('inf')
        self.pose_prediction_frames = 0
        self.pose_status = 'detected'
        self.display_held = False
        self.display_candidate_affine = None
        self.display_candidate_hits = 0
        self.failed_frames = 0
        self.tracked_frames, self.seeded_at = 0, -20
        self.reference_checked_at = 0
        self.seeded_rvec = None
        self.initialization_failure = None
        self.long_axis_jump_deg = None

    @property
    def pose_camera(self):
        """Explicit alias for the current CAD pose in the RGB camera frame."""
        return self.pose

    @pose_camera.setter
    def pose_camera(self, value):
        self.pose = value


def _copy_box(box):
    return None if box is None else np.asarray(box, np.float32).reshape(4).copy()


def _box_diagnostics(observed, projected):
    """Return comparable detector/CAD box errors in image pixels.

    These values are diagnostic only.  They do not alter the metric pose or
    silently apply a screen-space correction to the CAD projection.
    """
    if observed is None or projected is None:
        return None
    observed = np.asarray(observed, np.float64).reshape(4)
    projected = np.asarray(projected, np.float64).reshape(4)
    if not np.isfinite(observed).all() or not np.isfinite(projected).all():
        return None
    observed_size = np.maximum(observed[2:] - observed[:2], 0.)
    projected_size = np.maximum(projected[2:] - projected[:2], 0.)
    intersection = np.maximum(0., np.minimum(observed[2:], projected[2:]) -
                              np.maximum(observed[:2], projected[:2]))
    overlap = float(np.prod(intersection))
    observed_area = float(np.prod(observed_size))
    projected_area = float(np.prod(projected_size))
    union = observed_area + projected_area - overlap
    observed_center = (observed[:2] + observed[2:]) * .5
    projected_center = (projected[:2] + projected[2:]) * .5
    scale_ratio = float(np.sqrt(observed_area / projected_area)) if projected_area > 1. else None
    aspect_ratio = None
    if projected_size[0] > 1. and projected_size[1] > 1.:
        aspect_ratio = float((observed_size[0] / observed_size[1]) /
                             (projected_size[0] / projected_size[1]))
    return dict(iou=(overlap / union if union > 1. else 0.),
                center_error_px=float(np.linalg.norm(observed_center - projected_center)),
                scale_ratio=scale_ratio, aspect_ratio=aspect_ratio)


def _box_filter(previous, measurement, prediction=None, alpha=.7):
    """Fuse a detector extent with checked flow without accepting a jump.

    The detector remains the measurement source.  The flow prediction only
    supplies the low-frequency prior, so a one-frame multi-view box excursion
    cannot move the displayed extent by its full amount.
    """
    measurement = _copy_box(measurement)
    if measurement is None:
        return _copy_box(prediction if prediction is not None else previous)
    prior = _copy_box(prediction if prediction is not None else previous)
    if prior is None:
        return measurement
    prior_size = np.maximum(prior[2:] - prior[:2], 1.)
    measured_size = np.maximum(measurement[2:] - measurement[:2], 1.)
    prior_center = (prior[:2] + prior[2:]) * .5
    measured_center = (measurement[:2] + measurement[2:]) * .5
    center_limit = max(12., .45 * float(np.linalg.norm(prior_size)))
    if np.linalg.norm(measured_center - prior_center) > center_limit:
        return measurement
    size_ratio = measured_size / prior_size
    if np.any(size_ratio < .65) or np.any(size_ratio > 1.55):
        return measurement
    return (prior + float(alpha) * (measurement - prior)).astype(np.float32)


def _predict_pose_from_box(pose, previous_box, current_box, K):
    """Create a short-lived image-motion pose prediction.

    A failed LK/PnP update must not make a verified contour blink out for one
    frame.  This fallback follows image translation and only a checked,
    nearly-uniform box scale.  A real depth change expands or contracts both
    axes together; a horizontal target's edge mask often changes only one axis
    or changes the two axes by very different amounts.  The latter is treated
    as extent noise so it cannot make the CAD contour jump in size.  It is
    explicitly marked as predicted and is never used as a new metric
    measurement.
    """
    if pose is None or previous_box is None or current_box is None:
        return None
    previous_box = np.asarray(previous_box, np.float64).reshape(4)
    current_box = np.asarray(current_box, np.float64).reshape(4)
    if not np.isfinite(previous_box).all() or not np.isfinite(current_box).all():
        return None
    old_size = previous_box[2:] - previous_box[:2]
    new_size = current_box[2:] - current_box[:2]
    if np.any(old_size < 20.) or np.any(new_size < 20.):
        return None
    old_center = (previous_box[:2] + previous_box[2:]) * .5
    new_center = (current_box[:2] + current_box[2:]) * .5
    scale_xy = new_size / np.maximum(old_size, 1e-6)
    if not np.isfinite(scale_xy).all() or np.any(scale_xy <= 0.):
        return None
    # Use extent as a depth cue only when both axes agree.  A single-axis
    # change is common when a horizontal object rotates slightly or an edge is
    # segmented differently.  Damping keeps a genuine slow approach/recede
    # responsive while preventing one noisy box from changing depth abruptly.
    log_scale_xy = np.log(scale_xy)
    uniform_scale = (np.sign(log_scale_xy[0]) == np.sign(log_scale_xy[1]) and
                     abs(float(log_scale_xy[0] - log_scale_xy[1])) <= .18)
    log_scale = float(np.mean(log_scale_xy)) if uniform_scale else 0.
    log_scale = float(np.clip(.55 * log_scale, np.log(.86), np.log(1.16)))
    scale = float(np.exp(log_scale))
    tvec = np.asarray(pose.tvec, np.float64).reshape(3).copy()
    if not np.isfinite(tvec).all() or tvec[2] <= 1.:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx <= 1. or fy <= 1.:
        return None
    # Only the uniform, damped component changes depth.  The detector extent is
    # still not a full depth sensor, so this remains a short-lived prediction
    # and never refreshes the measured pose timestamp.
    tvec[2] /= scale
    tvec[0] += (new_center[0] - old_center[0]) * tvec[2] / fx
    tvec[1] += (new_center[1] - old_center[1]) * tvec[2] / fy
    pose_pixels = getattr(pose, 'pixels', None)
    if pose_pixels is None:
        return None
    pixels = np.asarray(pose_pixels, np.float32).reshape(-1, 2)
    if len(pixels) and np.isfinite(pixels).all():
        pixels = ((pixels - old_center.astype(np.float32)) * scale +
                  new_center.astype(np.float32)).astype(np.float32)
    return replace(pose, tvec=tvec, pixels=pixels)





def lock_long_axis_twist(pose, reference_pose):
    """Keep the previous X/Y orientation while following the current Z axis.

    The metric pose remains untouched; this returns a display-only copy. The
    model Z axis is the object's long axis. A minimum rotation maps the
    previous displayed Z axis to the current measured Z axis, then applies that
    swing to the previous display pose. Measured roll, including roll already
    accumulated in earlier noisy frames, is excluded from the result.
    """
    if pose is None or reference_pose is None:
        return pose
    try:
        previous_rotation, _ = cv2.Rodrigues(
            np.asarray(reference_pose.rvec, np.float64).reshape(3))
        current_rotation, _ = cv2.Rodrigues(
            np.asarray(pose.rvec, np.float64).reshape(3))
        previous_axis = previous_rotation[:, 2].astype(np.float64)
        current_axis = current_rotation[:, 2].astype(np.float64)
        previous_axis /= np.linalg.norm(previous_axis)
        current_axis /= np.linalg.norm(current_axis)
        # PnP can return the same elongated object with its local long axis
        # reversed. That is a 180-degree sign ambiguity for this display
        # purpose, not a real change that should flip the displayed X/Y axes.
        # Unwrap the axis direction against the previous displayed pose before
        # computing the swing, so roll noise and sign flips are both ignored.
        if np.dot(previous_axis, current_axis) < 0.:
            current_axis = -current_axis
        cross = np.cross(previous_axis, current_axis)
        sine = np.linalg.norm(cross)
        cosine = float(np.clip(np.dot(previous_axis, current_axis), -1., 1.))
        if sine <= 1e-8:
            if cosine >= 0.:
                swing_rotation = np.eye(3, dtype=np.float64)
            else:
                # For the 180-degree case, any axis perpendicular to the old
                # long axis is valid; the previous X axis gives a stable choice.
                swing_rvec = previous_rotation[:, 0] * np.pi
                swing_rotation, _ = cv2.Rodrigues(swing_rvec)
        else:
            swing_rvec = cross / sine * np.arctan2(sine, cosine)
            swing_rotation, _ = cv2.Rodrigues(swing_rvec)
        locked_rotation = swing_rotation @ previous_rotation
        locked_rvec, _ = cv2.Rodrigues(locked_rotation)
        return replace(pose, rvec=locked_rvec.reshape(3).astype(np.float64))
    except (cv2.error, TypeError, ValueError, FloatingPointError):
        return pose


def association_overlap(target, box):
    """Do not let a merged/background detection consume a measured track.

    ``target.box`` is the detector/flow association extent.  CAD projection is
    kept separately so a new detector result cannot overwrite the tracking ROI.
    Use the same scale/aspect envelope as pose verification when assigning
    detector hints; rejected hints remain eligible for independent
    initialization.
    """
    overlap = box_overlap(target.box, box)[0]
    if target.last_good_pose is None and (
            encloses_smaller_candidate(target.box, np.asarray(box))
            or encloses_smaller_candidate(np.asarray(box), target.box)):
        # A new broad proposal must not replace a compact search ROI (or vice
        # versa) and reset its feature evidence on every scan orientation.
        return 0.
    if target.last_good_pose is not None:
        predicted = np.maximum(target.box[2:] - target.box[:2], 1.)
        observed = np.maximum(np.asarray(box)[2:] - np.asarray(box)[:2], 1.)
        aspect = (observed[0] / observed[1]) / (predicted[0] / predicted[1])
        scale = np.sqrt(np.prod(observed) / np.prod(predicted))
        if not (.65 <= aspect <= 1.55 and .75 <= scale <= 1.33):
            return 0.
    return overlap


def encloses_smaller_candidate(box, other):
    area = float(np.prod(np.maximum(box[2:] - box[:2], 0.)))
    other_area = float(np.prod(np.maximum(other[2:] - other[:2], 0.)))
    intersection = float(np.prod(np.maximum(
        np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]), 0.)))
    return other_area > 0 and area >= 2.5 * other_area and intersection >= .9 * other_area


def compact_search_candidates(targets):
    """Prefer contained ROIs without deleting possibly overlapping objects."""
    return [target for target in targets if not any(
        other is not target and encloses_smaller_candidate(target.box, other.box)
        for other in targets)]


def duplicate_candidate_boxes(first, second):
    """Return true for a broad/compact proposal pair for the same object."""
    return (encloses_smaller_candidate(np.asarray(first), np.asarray(second))
            or encloses_smaller_candidate(np.asarray(second), np.asarray(first)))


def similar_candidate_boxes(first, second, *, min_iou=.55):
    """Return true for two similarly sized, strongly overlapping boxes."""
    first = np.asarray(first, np.float64).reshape(4)
    second = np.asarray(second, np.float64).reshape(4)
    first_size = np.maximum(first[2:] - first[:2], 0.)
    second_size = np.maximum(second[2:] - second[:2], 0.)
    first_area = float(np.prod(first_size))
    second_area = float(np.prod(second_size))
    if min(first_area, second_area) <= 0.:
        return False
    intersection_size = np.maximum(
        np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0.)
    intersection = float(np.prod(intersection_size))
    union = first_area + second_area - intersection
    iou = intersection / max(union, 1e-6)
    area_ratio = max(first_area, second_area) / min(first_area, second_area)
    return iou >= min_iou and area_ratio <= 1.8



class TargetSet:
    """One-to-one image association; IDs are session-local, not physical identity."""

    def __init__(self, max_missing_s=1., min_iou=.15, detection_grace_s=.3, search_grace_s=0.):
        self.max_missing_s = max_missing_s
        self.min_iou = min_iou
        self.detection_grace_s = detection_grace_s
        self.search_grace_s = search_grace_s
        self.targets = {}
        self.next_id = 1
        self.previous = None

    def update(self, frame, boxes, now):
        # Expire absent tracks, not objects still present in the preceding
        # frame. A slow initialization/render must not reset every ID/color
        # before the current detections get a chance to match.
        self.targets = {key: target for key, target in self.targets.items()
                        if target.visible or now - target.last_seen <= self.max_missing_s}
        previous = list(self.targets.values())
        # Preserve broad ROIs while the target is still uninitialized; startup
        # search intentionally uses them as a fallback. Once a measured pose
        # exists, a near-contained proposal is a duplicate and must not create
        # or maintain a second track for the same object.
        measured_previous = [target for target in previous
                             if target.visible and target.pose is not None]
        propagated = set()
        for target in previous:
            # A prediction is valid only for this source/current frame pair.
            # Do not let a successful flow estimate from an older frame become
            # the prior for a later detector measurement after flow fails.
            target.predicted_box = None
            if target.visible and self.previous is not None:
                flow_seed = target.filtered_box if target.filtered_box is not None else target.box
                moved = advance_box(self.previous, frame, flow_seed)
                if moved is not None:
                    target.predicted_box = _copy_box(moved)
                    target.filtered_box = _copy_box(moved)
                    target.box = moved
                    target.box_source = 'optical_flow'
                    propagated.add(target.id)
            target.visible = False
            target.detected = False
        matched = set()
        if previous and len(boxes):
            # A duplicate 2D candidate must not steal a compatible detection
            # from a measured 6DoF track as the object's silhouette changes.
            # Match measured tracks first, still one-to-one within each pass.
            for group in ([t for t in previous if t.pose is not None],
                          [t for t in previous if t.pose is None]):
                available = [i for i in range(len(boxes)) if i not in matched]
                if group and group[0].pose is None and measured_previous:
                    available = [i for i in available if not any(
                        duplicate_candidate_boxes(boxes[i], measured.box)
                        for measured in measured_previous)]
                if not group or not available:
                    continue
                overlaps = np.array([[association_overlap(t, boxes[i]) for i in available]
                                     for t in group])
                costs = np.where(overlaps >= self.min_iou, 1. - overlaps, 1e6)
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    if overlaps[row, column] < self.min_iou:
                        continue
                    target = group[row]
                    index = available[column]
                    measurement = _copy_box(boxes[index])
                    target.raw_detection_box = measurement.copy()
                    target.filtered_box = _box_filter(
                        target.filtered_box if target.filtered_box is not None else target.box,
                        measurement, target.predicted_box,
                        alpha=(1. if target.pose is not None else .7))
                    target.box = target.filtered_box.copy()
                    target.box_source = ('pose_association' if target.pose is not None
                                         else 'filtered_detection')
                    target.observed_box = measurement.copy()
                    target.last_seen = now
                    target.visible = target.detected = True
                    target.gap_frames = 0
                    matched.add(index)
        for column, box in enumerate(boxes):
            if column not in matched:
                if any(duplicate_candidate_boxes(box, measured.box)
                       for measured in measured_previous):
                    continue
                measurement = _copy_box(box)
                target = Target(self.next_id, measurement.copy(), now,
                                observed_box=measurement.copy(),
                                raw_detection_box=measurement.copy(),
                                filtered_box=measurement.copy(),
                                box_source='detection')
                self.targets[target.id] = target
                self.next_id += 1
        for target in self.targets.values():
            if not target.visible:
                target.gap_frames += 1
                # Bridge only an established pose track with checked current-image
                # flow. Neither propagation nor display hold renews detector age.
                # A recent measured 6DoF track gets a chance to follow the
                # current frame even if detector-box flow failed. Successful
                # PnP measurements, never a held contour, renew this evidence.
                measured = (target.pose is not None and target.pose_source is not None
                            and now - target.pose_time <= self.detection_grace_s)
                gap = (target.last_good_pose is not None and target.id in propagated
                       and now - target.last_seen <= self.detection_grace_s
                       and target.gap_frames <= 6)
                # Progressive live scans revisit diagonal views over three rounds.
                # Preserve a search only while current-image flow follows its ROI;
                # this is never a pose or permission to draw a contour.
                searching = (target.last_good_pose is None and target.id in propagated
                             and now - target.last_seen <= self.search_grace_s)
                bridge = ((measured or gap or searching)
                          and all(association_overlap(target, box) < self.min_iou
                                  or (searching and encloses_smaller_candidate(
                                      np.asarray(box), target.box)) for box in boxes))
                if bridge:
                    target.visible = True
                else:
                    target.clear_pose()
        self.previous = frame.copy()
        return [target for target in self.targets.values() if target.visible]


class IndependentPoses:
    """Bounded background search, one target at a time in round-robin order."""

    def __init__(self, loader, image_size=None, grace_s=.25, max_hold_frames=6,
                 ego_motion=None):
        self.image_size = image_size
        self.grace_s, self.max_hold_frames = grace_s, max_hold_frames
        self.ego_motion = ego_motion or IdentityEgoMotion()
        if not callable(getattr(self.ego_motion, 'pose_at', None)):
            raise TypeError('ego_motion must provide pose_at(timestamp)')
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='multi-pose')
        self.loading = self.executor.submit(loader)
        self.pending = None
        self.pending_trail = None
        self.pending_refresh = False
        self.stats = dict(submitted=0, no_reference_pose=0, target_lost=0, expired=0,
                          history_lost=0, handoff_failed=0, detector_conflict=0,
                          long_axis_branch=0, superseded=0, accepted=0, corrected=0)
        self.last_id = 0
        self.search_round = 0
        self.last_recovery_id = 0
        self.previous = None
        self.error = None

    def update(self, targets, frame, *, schedule=True, timestamp=None):
        from .rgb_global_initializer import advance_pose, RgbPoseSearch

        now = time.monotonic() if timestamp is None else timestamp
        frame_source = frame.copy()
        new_frame = self.previous is None or not np.array_equal(self.previous, frame)
        if self.pending_trail is not None:
            self.pending_trail.append(frame)

        if self.image_size is not None and tuple(frame.shape[1::-1]) != self.image_size:
            self.error = f'RGB size {frame.shape[1::-1]} differs from calibrated {self.image_size}'
            for target in targets:
                target.clear_pose()
            return
        if not self.loading.done():
            return
        try:
            initializer = self.loading.result()
        except Exception as exc:
            self.error = str(exc)
            return
        def accept(target, pose, *, reference=False):
            previous_pose = target.pose if target.pose is not None else target.last_good_pose
            target.long_axis_jump_deg = None
            if previous_pose is not None:
                target.long_axis_jump_deg = directed_long_axis_jump_deg(
                    previous_pose.rvec, previous_pose.tvec,
                    pose.rvec, pose.tvec, initializer.K, initializer.dist)
            if not pose_has_continuous_long_axis(
                    previous_pose, pose, initializer.K, initializer.dist):
                # A low RMS is insufficient for an elongated/symmetric target:
                # PnP can fit a second 180-degree pose branch just as well. Keep
                # the last verified pose and let temporal tracking or a later
                # reference search recover through intermediate directions.
                self.stats['long_axis_branch'] += 1
                target.initialization_failure = 'long_axis_branch'
                return False
            if new_frame:
                target.tracked_frames += 1
            if previous_pose is not None:
                try:
                    old_center, _ = cv2.projectPoints(
                        np.zeros((1, 3), np.float64), previous_pose.rvec,
                        previous_pose.tvec, initializer.K, initializer.dist)
                    new_center, _ = cv2.projectPoints(
                        np.zeros((1, 3), np.float64), pose.rvec, pose.tvec,
                        initializer.K, initializer.dist)
                    target.display_motion_px = float(np.linalg.norm(
                        new_center.reshape(2) - old_center.reshape(2)))
                    old_rotation, _ = cv2.Rodrigues(previous_pose.rvec)
                    new_rotation, _ = cv2.Rodrigues(pose.rvec)
                    cosine = np.clip((np.trace(new_rotation @ old_rotation.T) - 1.) / 2.,
                                     -1., 1.)
                    target.display_rotation_deg = float(np.rad2deg(np.arccos(cosine)))
                except (cv2.error, TypeError, ValueError):
                    target.display_motion_px = 0.
                    target.display_rotation_deg = 0.
            else:
                target.display_motion_px = 0.
                target.display_rotation_deg = 0.
            # Replenish a verified track before feature attrition reaches the
            # PnP minimum. Limit raycasting cost per target; never seed a hold.
            age = target.tracked_frames - target.seeded_at
            if (getattr(initializer, 'tracking_seeder', None) is not None
                    and (reference or needs_replenishment(
                        pose, target.last_good_pose, target.seeded_rvec, age))):
                pose = replenish_pose(pose, frame, initializer)
                target.seeded_at = target.tracked_frames
                target.seeded_rvec = pose.rvec.copy()
            if reference:
                target.reference_checked_at = target.tracked_frames
            # A temporal pose update also carries the median displacement of
            # the real image correspondences.  If the detector and its box
            # flow both missed this frame, use that measured displacement to
            # advance the association ROI.  A box already advanced by checked
            # ROI flow must not be shifted a second time, and the CAD
            # projection is never allowed to drive the ROI.
            motion_shift = getattr(pose, 'motion_shift_px', None)
            if (not target.detected and target.predicted_box is None
                    and motion_shift is not None and target.box is not None):
                motion_shift = np.asarray(motion_shift, np.float32).reshape(2)
                if np.isfinite(motion_shift).all():
                    target.box = (np.asarray(target.box, np.float32).reshape(4)
                                  + np.tile(motion_shift, 2)).astype(np.float32)
                    target.filtered_box = target.box.copy()
                    target.predicted_box = target.box.copy()
                    target.box_source = 'optical_flow'
            target.pose = target.last_good_pose = pose
            target.pose_status = 'pose_verified'
            target.pose_prediction_frames = 0
            target.pose_prediction_time = -float('inf')
            from .cad_video_tracker import project
            pixels = project(initializer.vertices, pose.rvec, pose.tvec,
                             initializer.K, initializer.dist)
            height, width = frame.shape[:2]
            bounds = np.clip(np.r_[pixels.min(0), pixels.max(0)],
                             [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
            if np.isfinite(bounds).all() and np.min(bounds[2:] - bounds[:2]) >= 20.:
                target.pose_projection_box = bounds.astype(np.float32)
                # Keep the association box independent from the metric CAD
                # projection. In a detector gap, writing this projection back
                # into ``target.box`` creates a positive-feedback loop: a
                # slightly wrong pose moves the ROI, the moved ROI selects a
                # different background edge, and the next pose drifts farther.
                # The detector/flow box remains the only association prior;
                # the CAD projection is used separately for pose validation and
                # diagnostics.
            # The temporal source box belongs to detector/flow association.
            # Keep it separate from the CAD projection so a pose error cannot
            # feed back into the next image search.
            target.pose_source, target.pose_box = frame_source, target.box.copy()
            target.pose_time = now
            target.pose_frame = 'camera'
            target.ego_motion_source = getattr(self.ego_motion, 'source', 'identity')
            target.failed_frames = 0
            target.tracking_failure = None
            target.initialization_failure = None
            return True

        for target in targets:
            # This is a per-frame diagnostic.  ``accept`` fills it when a new
            # measured candidate is evaluated; leaving it empty avoids making
            # an old branch jump look like a current-frame event.
            target.long_axis_jump_deg = None
            # A predicted pose carries the latest image motion forward.  A
            # measured pose always starts from the last verified measurement.
            if target.pose_status == 'pose_predicted' and target.pose is not None:
                base = target.pose
            else:
                base = (target.last_good_pose if target.last_good_pose is not None
                        else target.pose)
            source = target.pose_source if target.pose_source is not None else self.previous
            can_retry = target.pose is not None or now - target.pose_time <= self.grace_s
            if base is not None and source is not None and can_retry:
                if np.array_equal(source, frame):
                    pose = base
                else:
                    # Retry from the last measured frame, never the failed frame:
                    # base.pixels are coordinates in source, not self.previous.
                    # Keep the last measured CAD extent in the flow/PnP solve.
                    # Passing ``None`` here disabled the only geometric guard
                    # on temporal updates, so a locally consistent background
                    # edge could produce a large but apparently valid pose jump.
                    # ``pose_box`` is expressed in ``source``; advance_pose
                    # moves it with the measured flow before validating the
                    # candidate in ``current``.
                    pose = advance_pose(base, source, frame, initializer.K,
                                        initializer.dist, initializer.vertices,
                                        target.pose_box, retry_pnp=True)
                validation_box = (target.raw_detection_box if target.detected
                                  and target.raw_detection_box is not None else
                                  target.predicted_box)
                if (validation_box is None and pose is not None
                        and target.box is not None):
                    motion_shift = getattr(pose, 'motion_shift_px', None)
                    if motion_shift is not None:
                        motion_shift = np.asarray(motion_shift, np.float32).reshape(2)
                        if np.isfinite(motion_shift).all():
                            # This is a temporary validation box for the
                            # current frame. ``accept`` commits the same shift
                            # only after the pose passes all metric gates.
                            validation_box = (np.asarray(target.box, np.float32)
                                              + np.tile(motion_shift, 2))
                # Detector gaps still have an independent image-space prior when
                # the detector ROI was advanced by checked optical flow. If that
                # flow failed, the measured point motion supplies a temporary
                # box for validation. Never use the CAD projection itself.
                box_is_valid = (pose is not None and (validation_box is None or
                    detection_matches_pose(validation_box, initializer.vertices,
                                           pose.rvec, pose.tvec, initializer.K,
                                           initializer.dist, frame.shape)))
                if pose is not None and box_is_valid:
                    if accept(target, pose):
                        continue
                    reason = 'long_axis_branch'
                elif pose is not None:
                    reason = 'detector_conflict'
                else:
                    reason = 'flow_or_pnp'
                if reason == 'long_axis_branch':
                    target.pose = None
                    target.pose_status = 'detected'
                    target.tracking_failure = reason
                    continue
                if reason == 'detector_conflict' or reason == 'flow_or_pnp':
                    predicted_box = None
                    if target.pose_box is not None and source is not None:
                        predicted_box = advance_box(source, frame, target.pose_box)
                    predicted_pose = _predict_pose_from_box(
                        base, target.pose_box, predicted_box, initializer.K)
                    predicted_ok = (predicted_pose is not None and
                                    target.pose_prediction_frames < 3 and
                                    (not target.detected or detection_matches_pose(
                                        validation_box, initializer.vertices,
                                        predicted_pose.rvec, predicted_pose.tvec,
                                        initializer.K, initializer.dist, frame.shape)))
                    if predicted_ok:
                        target.pose = predicted_pose
                        target.pose_status = 'pose_predicted'
                        target.pose_prediction_frames += 1
                        target.pose_prediction_time = now
                        target.pose_source = frame_source
                        target.pose_box = predicted_box.astype(np.float32)
                        # Keep the association ROI from checked box flow, but
                        # derive the diagnostic CAD box from the predicted
                        # metric pose. The two can differ during scale/depth
                        # prediction and must not be conflated.
                        projection = project_target_box(target, initializer)
                        if projection is not None:
                            target.pose_projection_box = np.asarray(
                                np.r_[projection[1].min(0), projection[1].max(0)],
                                np.float32)
                        else:
                            target.pose_projection_box = np.asarray(
                                predicted_box, np.float32).copy()
                        target.tracking_failure = reason
                    else:
                        target.pose = None
                        target.pose_status = 'detected'
                        target.tracking_failure = reason
            elif base is not None:
                target.pose = None
                target.pose_status = 'detected'
                target.tracking_failure = target.tracking_failure or 'recovery_expired'
        if self.pending is not None and self.pending.done():
            target_id, source, captured_at, search = self.request
            try:
                pose = self.pending.result()
                target = next((t for t in targets if t.id == target_id and t.search is search), None)
                reason = None
                if target is None:
                    reason = 'target_lost'
                elif target.pose is not None and not self.pending_refresh:
                    # A newer current-frame measurement already won. A delayed
                    # reference must not replace it and make the contour jump.
                    reason = 'superseded'
                elif pose is None:
                    reason = 'no_reference_pose'
                elif time.monotonic() - captured_at > 1.:
                    reason = 'expired'
                elif self.pending_trail is None or not self.pending_trail.complete:
                    reason = 'history_lost'
                else:
                    if len(self.pending_trail.frames) > 1:
                        pose = advance_pose_trail(pose, self.pending_trail.frames, initializer,
                                                  replenish=not self.pending_refresh)
                    if pose is None:
                        reason = 'handoff_failed'
                    else:
                        validation_box = (target.raw_detection_box if target.detected
                                          and target.raw_detection_box is not None else target.box)
                        if not detection_matches_pose(
                                validation_box, initializer.vertices, pose.rvec, pose.tvec,
                                initializer.K, initializer.dist, frame.shape):
                            reason = 'detector_conflict'
                        elif (pose is not None and not pose_has_continuous_long_axis(
                                target.pose if target.pose is not None else target.last_good_pose,
                                pose, initializer.K, initializer.dist)):
                            reason = 'long_axis_branch'
                        elif target.pose is not None and not reference_improves_pose(
                                target.pose, pose, initializer):
                            reason = 'superseded'
                        else:
                            if target.pose is not None:
                                self.stats['corrected'] += 1
                            if accept(target, pose, reference=True):
                                self.stats['accepted'] += 1
                            else:
                                reason = 'long_axis_branch'
                if reason is not None:
                    self.stats[reason] += 1
                    if target is not None and reason != 'superseded':
                        target.initialization_failure = reason
            except Exception as exc:
                self.error = str(exc)
            self.pending = None
            self.pending_trail = None
        for target in targets:
            target.display_held = False
            if target.pose is not None:
                # Keep the metric pose untouched for robot output.  The
                # display pose unwraps the sign-ambiguous long axis so a
                # symmetric elongated target does not make the overlay arrow
                # reverse when PnP visits the equivalent branch.
                reference_display = (target.display_pose if target.display_pose is not None
                                      else target.last_good_pose)
                target.display_pose = lock_long_axis_twist(target.pose, reference_display)
                continue
            target.display_pose = None
            if new_frame:
                target.failed_frames += 1
            base = target.last_good_pose
            if (base is None or now - target.pose_time > self.grace_s
                    or target.failed_frames > self.max_hold_frames):
                continue
            moved = advance_box(target.pose_source, frame, target.pose_box)
            # Hold only nearly stationary, still-supported objects. Actual motion
            # needs a newly solved pose; a stale contour must not trail the hand.
            validation_box = (target.raw_detection_box if target.detected
                              and target.raw_detection_box is not None else target.box)
            if (moved is not None and np.max(np.abs(moved - target.pose_box)) <= 4.
                    and detection_matches_pose(validation_box, initializer.vertices,
                        base.rvec, base.tvec, initializer.K, initializer.dist, frame.shape)):
                target.display_pose = base
                target.display_rotation_deg = 0.
                target.display_held = True
        self.previous = frame.copy()
        if schedule and targets and self.pending is None:
            # Give recently lost measured tracks a recovery slot between normal
            # round-robin slots. Spurious/new 2D candidates cannot delay every
            # recovery, and a persistently lost target cannot starve new ones.
            ordered = sorted((t for t in targets if t.pose is None), key=lambda t: t.id)
            # Large arm/background proposals often contain a much tighter unit.
            # Give compact ROIs three of four rounds, but still search enclosing
            # candidates: containment alone cannot disprove a physical object.
            if self.search_round % 4 != 3:
                compact = compact_search_candidates(ordered)
                if compact:
                    ordered = compact
            audits = sorted((t for t in targets if t.pose is not None
                             and t.tracked_frames - t.reference_checked_at >= 30),
                            key=lambda t: (-(t.tracked_frames - t.reference_checked_at), t.id))
            # Reserve at most one in four busy search slots for an independent
            # reference audit. Otherwise self-seeded tracks can drift forever.
            if audits and (not ordered or self.search_round % 4 == 3):
                target = audits[0]
                target.reference_checked_at = target.tracked_frames
            elif ordered:
                recovering = [t for t in ordered if t.last_good_pose is not None]
                recovery_slot = bool(recovering) and self.search_round % 2 == 0
                pool = recovering if recovery_slot else ordered
                cursor = self.last_recovery_id if recovery_slot else self.last_id
                target = next((t for t in pool if t.id > cursor), pool[0])
                if recovery_slot:
                    self.last_recovery_id = target.id
                else:
                    self.last_id = target.id
            else:
                return
            self.search_round += 1
            if target.search is None:
                target.search = initializer.new_search()
                if isinstance(target.search, RgbPoseSearch):
                    target.search.cross_family_support = True
            source = frame.copy()
            box = target.box.copy()
            self.request = target.id, source, time.monotonic(), target.search
            self.pending_trail = FrameTrail(source)
            self.pending_refresh = target.pose is not None
            self.stats['submitted'] += 1
            prior_pose = (target.last_good_pose if target.last_good_pose is not None
                          else target.pose)
            # Keep compatibility with lightweight/custom search objects used by
            # older callers.  The built-in RgbPoseSearch accepts the temporal
            # prior; a legacy callable without that parameter still receives
            # the same validation box and remains usable.
            estimate_kwargs = dict(validation_box=box)
            try:
                parameters = inspect.signature(target.search.estimate).parameters
                if ('prior_pose' in parameters or
                        any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters.values())):
                    estimate_kwargs['prior_pose'] = prior_pose
            except (TypeError, ValueError):
                estimate_kwargs['prior_pose'] = prior_pose
            self.pending = self.executor.submit(
                target.search.estimate, source, box, **estimate_kwargs)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def project_target_box(target, initializer, *, display=False):
    """Project the metric CAD bounds using this instance's measured R and t.

    No screen-space extrusion, neutralized roll or cosmetic position offset is
    applied. Both returned arrays use the same corner ordering as BOX_EDGES.
    """
    pose = ((target.display_pose if target.display_pose is not None else target.pose)
            if display else target.pose)
    if pose is None or initializer is None:
        return None
    rvec = np.asarray(pose.rvec, np.float64).reshape(3)
    tvec = np.asarray(pose.tvec, np.float64).reshape(3)
    if not np.isfinite(np.r_[rvec, tvec]).all():
        return None
    lower = np.min(initializer.vertices, axis=0)
    upper = np.max(initializer.vertices, axis=0)
    corners = (lower + upper) * .5 + BOX_SIGNS * (upper - lower) * .5
    rotation, _ = cv2.Rodrigues(rvec)
    camera_corners = corners @ rotation.T + tvec
    if not np.isfinite(camera_corners).all() or np.any(camera_corners[:, 2] <= 1.):
        return None
    pixels, _ = cv2.projectPoints(corners, rvec, tvec, initializer.K, initializer.dist)
    pixels = pixels.reshape(8, 2)
    if not np.isfinite(pixels).all() or np.any(np.abs(pixels) >= 1e6):
        return None
    return camera_corners, pixels


def target_record(target, initializer=None):
    geometry = project_target_box(target, initializer)
    pose = target.pose
    if initializer is not None and geometry is None:
        pose = None
    box = target.filtered_box if target.filtered_box is not None else target.box
    pose_box = target.pose_projection_box
    raw_box = target.raw_detection_box
    predicted_box = target.predicted_box
    box_diag = _box_diagnostics(raw_box, pose_box) or {}
    status = (target.pose_status if pose is not None else 'detected')
    if pose is not None and status == 'detected':
        # Tests and external callers may construct a Target with a verified
        # pose directly, without going through IndependentPoses.accept().
        status = 'pose_verified'
    # Predictions retain the latest verified PnP diagnostics. They are useful
    # for judging the age of the prediction, but are not new measurements.
    measured_pose = (target.last_good_pose
                     if status == 'pose_predicted' and target.last_good_pose is not None
                     else pose)
    return dict(id=target.id, box_xyxy=np.asarray(box).tolist(),
                raw_detection_box_xyxy=(np.asarray(raw_box).tolist()
                                        if raw_box is not None else None),
                predicted_box_xyxy=(np.asarray(predicted_box).tolist()
                                    if predicted_box is not None else None),
                filtered_box_xyxy=np.asarray(box).tolist(),
                cad_box_xyxy=(np.asarray(pose_box).tolist()
                              if pose_box is not None else None),
                association_box_xyxy=np.asarray(target.box).tolist(),
                box_source=target.box_source,
                detected=target.detected,
                tracking_failure=target.tracking_failure,
                initialization_failure=target.initialization_failure,
                pose_status=status,
                pose_age_frames=(target.pose_prediction_frames if pose is not None else None),
                pose_rms_px=(getattr(measured_pose, 'rms_px', None)
                             if measured_pose is not None else None),
                pose_inlier_ratio=(getattr(measured_pose, 'inlier_ratio', None)
                                   if measured_pose is not None else None),
                pose_reference_support=(getattr(measured_pose, 'reference_support', None)
                                        if measured_pose is not None else None),
                long_axis_jump_deg=target.long_axis_jump_deg,
                cad_detection_iou=box_diag.get('iou'),
                cad_detection_center_error_px=box_diag.get('center_error_px'),
                cad_detection_scale_ratio=box_diag.get('scale_ratio'),
                cad_detection_aspect_ratio=box_diag.get('aspect_ratio'),
                tracking_points=(len(pose.points) if pose is not None and hasattr(pose, 'points') else 0),
                display_held=target.display_held,
                display_visible=project_target_box(target, initializer, display=True) is not None,
                state=status,
                pose_frame=(target.pose_frame if pose is not None else None),
                pose_timestamp=(target.pose_time if pose is not None else None),
                pose_prediction_timestamp=(target.pose_prediction_time
                                           if pose is not None and
                                           status == 'pose_predicted'
                                           else None),
                ego_motion_source=(target.ego_motion_source if pose is not None else None),
                rvec=pose.rvec.reshape(3).tolist() if pose is not None else None,
                tvec_mm=pose.tvec.reshape(3).tolist() if pose is not None else None,
                box_corners_camera_mm=geometry[0].tolist() if geometry is not None else None,
                box_corners_image_px=geometry[1].tolist() if geometry is not None else None)


def target_color(target_id):
    """Stable, saturated BGR color; independent of frame/detection ordering."""
    hue = (1. / 3. + (target_id - 1) * .61803398875) % 1.
    rgb = colorsys.hsv_to_rgb(hue, .75, 1.)
    return tuple(round(channel * 255) for channel in reversed(rgb))


def verified_duplicate_targets(first, second, initializer=None):
    """Detect duplicate verified tracks even when their boxes have similar size."""
    first_pose = first.pose if first.pose is not None else first.display_pose
    second_pose = second.pose if second.pose is not None else second.display_pose
    if first_pose is None or second_pose is None:
        return False
    if duplicate_candidate_boxes(first.box, second.box):
        return True
    if similar_candidate_boxes(first.box, second.box):
        return True
    if initializer is None:
        return False
    first_geometry = project_target_box(first, initializer, display=first.pose is None)
    second_geometry = project_target_box(second, initializer, display=second.pose is None)
    if first_geometry is None or second_geometry is None:
        return False
    first_pixels = first_geometry[1]
    second_pixels = second_geometry[1]
    first_box = np.r_[first_pixels.min(axis=0), first_pixels.max(axis=0)]
    second_box = np.r_[second_pixels.min(axis=0), second_pixels.max(axis=0)]
    return similar_candidate_boxes(first_box, second_box)


def display_target_subset(targets, initializer=None):
    """Keep one contour when two verified tracks are duplicate proposals."""
    selected = []
    for target in targets:
        if target.pose is None and target.display_pose is None:
            continue
        duplicate_index = next((index for index, other in enumerate(selected)
                                if verified_duplicate_targets(target, other, initializer)),
                               None)
        if duplicate_index is None:
            selected.append(target)
            continue
        other = selected[duplicate_index]
        contained = duplicate_candidate_boxes(target.box, other.box)
        target_area = float(np.prod(np.maximum(target.box[2:] - target.box[:2], 0.)))
        other_area = float(np.prod(np.maximum(other.box[2:] - other.box[:2], 0.)))
        target_quality = (int(target.detected), target.tracked_frames,
                          -target.gap_frames, -target.id)
        other_quality = (int(other.detected), other.tracked_frames,
                         -other.gap_frames, -other.id)
        if ((contained and target_area < other_area)
                or (not contained and target_quality > other_quality)):
            selected[duplicate_index] = target
    return selected


def draw_targets(frame, targets, initializer=None, show_axes=True, show_box=False,
                 outline_seeder=None, show_outline=True, display_y_offset_px=0.,
                 display_edge_correction=False):
    # Keep show_box for existing callers; the multi-target preview is contour-only.
    canvas = frame.copy()
    edge_distance = None
    edge_distance_ready = False

    # A detector-only proposal has no metric geometry yet. Keep the preview
    # clean until a verified 6DoF pose (or a short hold of one) exists; the
    # detector box is still retained internally for association and recovery.

    for target in display_target_subset(targets, initializer):
        # A measured/predicted CAD pose is the only geometry source for a
        # live contour. ``display_pose`` is still useful for a short hold after
        # a measurement gap. A stale display copy must not override a current
        # measurement or prediction.
        metric_pose = target.pose if target.pose is not None else target.display_pose
        pose = target.display_pose if target.display_pose is not None else metric_pose
        if pose is None:
            continue
        if project_target_box(target, initializer, display=True) is None:
            continue
        if show_outline and outline_seeder is not None:
            if not display_edge_correction:
                # Background edges are not a valid measurement of a CAD pose.
                # Keep the field clear so a correction from an older frame or
                # caller cannot silently shift the live contour.
                target.display_affine = None
                target.display_candidate_affine = None
                target.display_candidate_hits = 0
            # Experimental edge fitting is restricted to stationary frames.
            # Repeated background edges can still pass consensus, so it must
            # remain opt-in and must never update the exported metric pose.
            moving = (target.display_motion_px >= 8. or
                      target.display_rotation_deg >= 2.)
            refit_interval = 8 if target.display_affine is None else 24
            if moving:
                target.display_candidate_affine = None
                target.display_candidate_hits = 0
            if target.display_affine is not None:
                # Discard corrections produced by older builds or callers.
                # Only a bounded translation is valid for this optional
                # display experiment; scale and shear are rejected.
                affine = np.asarray(target.display_affine, np.float32)
                if (affine.shape != (2, 3) or not np.isfinite(affine).all()
                        or np.max(np.abs(affine[:, :2] - np.eye(2, dtype=np.float32))) > 1e-4
                        or np.max(np.abs(affine[:, 2])) > 24.):
                    target.display_affine = None
                    target.display_candidate_affine = None
                    target.display_candidate_hits = 0
            if (display_edge_correction and not moving and
                    (target.display_affine is None or
                     target.tracked_frames - target.display_refit_at >= refit_interval)):
                if not edge_distance_ready:
                    edge_distance = display_edge_distance(frame)
                    edge_distance_ready = True
                candidate = estimate_display_affine(
                    frame, initializer.vertices, initializer.K, initializer.dist,
                    pose.rvec, pose.tvec, outline_seeder,
                    display_box=(target.observed_box if target.detected else None),
                    max_shift_px=8.,
                    edge_distance=edge_distance,
                    min_improvement_px=.8,
                    allow_shape_adjustment=False)
                if candidate is not None:
                    candidate = np.asarray(candidate, np.float32)
                    current = (None if target.display_affine is None else
                               np.asarray(target.display_affine, np.float32))
                    candidate_translation = candidate[:, 2]
                    current_translation = (None if current is None else
                                           current[:, 2])
                    # Small changes are below the useful display precision;
                    # ignoring them prevents a stationary contour from
                    # random-walking one pixel at a time.
                    close_to_current = (current_translation is not None and
                                        np.linalg.norm(candidate_translation -
                                                       current_translation) <= 1.5)
                    if close_to_current:
                        target.display_candidate_affine = None
                        target.display_candidate_hits = 0
                    else:
                        pending = target.display_candidate_affine
                        same_as_pending = (pending is not None and
                                           np.linalg.norm(candidate_translation -
                                                          np.asarray(pending)[:, 2]) <= 2.5)
                        if same_as_pending:
                            target.display_candidate_hits += 1
                        else:
                            target.display_candidate_affine = candidate.copy()
                            target.display_candidate_hits = 1
                        needed_hits = 2 if current is None else 3
                        if target.display_candidate_hits >= needed_hits:
                            if current is None:
                                target.display_affine = candidate.copy()
                            else:
                                # Apply a confirmed change conservatively;
                                # the confirmation already rejected one-off
                                # background edges, and this avoids a jump.
                                target.display_affine = (.35 * candidate +
                                                         .65 * current)
                            target.display_candidate_affine = None
                            target.display_candidate_hits = 0
                target.display_refit_at = target.tracked_frames
            layer = render_cad_pose_layer(
                frame.shape, initializer.vertices, initializer.K, initializer.dist,
                metric_pose.rvec, metric_pose.tvec, outline_seeder,
                display_y_offset_px=display_y_offset_px, show_axes=show_axes,
                axis_rvec=pose.rvec, axis_tvec=pose.tvec,
                contour_color=target_color(target.id),
                display_affine=(target.display_affine
                                if display_edge_correction else None))
            # Preserve the ID's color rather than adding it to background pixels.
            mask = np.any(layer != 0, axis=2)
            canvas[mask] = layer[mask]
        elif show_axes:
            cv2.drawFrameAxes(canvas, initializer.K, initializer.dist,
                              pose.rvec, pose.tvec, 30., 2)
    return canvas


def run_multi_target(args):
    from . import live_depth_6dof as live

    cv2.setNumThreads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = detector = poses = diagnostics = None
    outline_seeder = None
    timings = {}
    window = 'Energy units - multi target'
    offline = args.video is not None
    video_timestamps = load_video_timestamps(args.video) if offline else None
    video_fps = 30.
    started = offline or not args.manual_start or args.headless
    targets = TargetSet(search_grace_s=0. if offline else .75)
    frame_index = 0
    started_at = time.monotonic()
    try:
        if offline:
            video_path = Path(args.video).expanduser()
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                if not video_path.is_file():
                    raise RuntimeError(
                        f'Video file does not exist: {video_path}. '
                        'Replace the example path after --video with the absolute path '
                        'of a real video, or omit --video to use the camera.'
                    )
                raise RuntimeError(f'Cannot open video: {video_path}')
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            video_fps = reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 30.
            detector = live.prepare_live_detector(args.reacquire_model, timings)
        else:
            camera_index, camera_name = live.resolve_camera_index(args.camera_name, args.camera)
            identity = (live.PinnedCamera.bind(camera_index, camera_name)
                        if live.sys.platform == 'win32' else None)
            cap = (identity.open(live.open_camera, *live.IMAGE_SIZE,
                                 args.camera_fourcc, args.camera_fps) if identity is not None
                   else live.open_camera(camera_index, *live.IMAGE_SIZE,
                                         args.camera_fourcc, args.camera_fps))
            detector = BackgroundDetector(
                lambda: live.prepare_live_detector(args.reacquire_model, timings),
                detect_live_target_boxes, max_age_s=1., multiple=True)
        if args.rgb_first:
            _, _, image_size = live.load_calibration(args.intrinsics)
            def load_initializer():
                initializer = live.prepare_rgb_initializer(args, timings)
                initializer.tracking_seeder = VisibleSurfaceSeeder(
                    args.model, initializer.K, initializer.dist)
                return initializer
            poses = IndependentPoses(load_initializer, image_size)
            if offline:
                # Offline frames have no camera clock; finish loading before consuming the clip.
                try:
                    poses.loading.result()
                except Exception as exc:
                    poses.error = str(exc)
        if not offline and args.record_diagnostics:
            K, dist, _ = live.load_calibration(args.intrinsics)
            diagnostics = TrackingDiagnostics(
                args.output_dir / 'diagnostics', K, dist, args.model, args.intrinsics,
                record=True, duration_s=args.diagnostic_duration_s, fps=args.camera_fps)
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            if not started:
                print('Press SPACE or S to start; Q or Esc to quit.')
        with (args.output_dir / 'multi_targets.jsonl').open('w', encoding='utf-8') as log:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if offline:
                        break
                    raise RuntimeError('RGB camera stopped delivering frames')
                now = time.monotonic()
                boxes = detector.get_boxes(frame) if started else []
                error = getattr(detector, 'error', None)
                if error:
                    raise RuntimeError(f'Multi-target detector failed: {error}')
                if video_timestamps is not None and frame_index >= len(video_timestamps):
                    raise RuntimeError('Video has more frames than its timestamp metadata')
                # Recorded motion follows capture time, never replay CPU time.
                tracking_time = (float(video_timestamps[frame_index]) if video_timestamps is not None
                                 else frame_index/video_fps if offline else now)
                visible = targets.update(frame, boxes, tracking_time)
                if poses is not None and started:
                    poses.update(visible, frame, timestamp=tracking_time)
                    if offline and poses.pending is not None:
                        try:
                            poses.pending.result()
                        except Exception:
                            pass  # update records the error and keeps detection available.
                        poses.update(visible, frame, schedule=False, timestamp=tracking_time)
                initializer = (poses.loading.result() if poses is not None and poses.loading.done()
                               and poses.loading.exception() is None else None)
                if (outline_seeder is None and initializer is not None and args.cad_outline
                        and any(target.pose is not None for target in visible)):
                    outline_seeder = getattr(initializer, 'tracking_seeder', None)
                    if outline_seeder is None:
                        outline_seeder = VisibleSurfaceSeeder(args.model, initializer.K, initializer.dist)
                canvas = draw_targets(
                    frame, visible, initializer, args.pose_axes, args.show_box,
                    outline_seeder=outline_seeder, show_outline=args.cad_outline,
                    display_y_offset_px=args.box_display_y_offset_px or 0.,
                    display_edge_correction=getattr(args, 'display_edge_correction', False))
                if not started and not args.headless:
                    cv2.putText(canvas, 'Press SPACE or S to start detection',
                                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, .9,
                                (0, 255, 255), 2, cv2.LINE_AA)
                records = [target_record(t, initializer) for t in visible]
                record = dict(frame=frame_index, elapsed_s=now-started_at,
                    source_elapsed_s=tracking_time if offline else now-started_at,
                    count=len(visible),
                    pose_verified_count=sum(r['state'] == 'pose_verified' for r in records),
                    display_count=sum(r['display_visible'] for r in records),
                    held_count=sum(r['display_held'] for r in records),
                    targets=records,
                    pose_error=poses.error if poses is not None else None,
                    pose_search_stats=poses.stats.copy() if poses is not None else None,
                    detector_stats=getattr(detector, 'stats', {}).copy())
                log.write(json.dumps(record) + '\n')
                if diagnostics is not None and started:
                    diagnostics.write(frame, record)
                frame_index += 1
                if not args.headless:
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) & 0xff
                    if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key in (32, ord('s')):
                        started = True
                if args.max_frames and frame_index >= args.max_frames:
                    break
        return 0
    finally:
        if diagnostics is not None:
            diagnostics.close()
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        if poses is not None:
            poses.close()
        if not args.headless:
            cv2.destroyAllWindows()
