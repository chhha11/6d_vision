"""Multi-instance RGB detection with independent, verified RGB/CAD poses."""
from __future__ import annotations

import colorsys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .background_detector import BackgroundDetector, advance_box
from .cad_video_tracker import BOX_SIGNS, box_overlap, detection_matches_pose, load_video_timestamps
from .multi_target_detection import detect_target_boxes, detect_live_target_boxes
from .multi_target_tracking import replenish_pose, needs_replenishment, advance_pose_trail, reference_improves_pose
from .frame_trail import FrameTrail
from .pose_visualization import (display_edge_distance, estimate_display_affine,
                                  render_cad_pose_layer)
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
    # Most recent detector extent in the current frame. This is used only to
    # align the rendered CAD contour; it never changes the metric pose.
    observed_box: object = None
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

    def clear_pose(self):
        self.pose = self.last_good_pose = self.display_pose = None
        self.pose_source = self.pose_box = self.search = None
        self.observed_box = None
        self.display_affine = None
        self.display_refit_at = -20
        self.display_motion_px = 0.
        self.display_rotation_deg = 0.
        self.pose_time = -float('inf')
        self.display_held = False
        self.display_candidate_affine = None
        self.display_candidate_hits = 0
        self.failed_frames = 0
        self.tracked_frames, self.seeded_at = 0, -20
        self.reference_checked_at = 0
        self.seeded_rvec = None
        self.initialization_failure = None





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

    After a pose measurement target.box is the current CAD projection. Use the
    same scale/aspect envelope as pose verification when assigning detector
    hints; rejected hints remain eligible for independent initialization.
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
            if target.visible and self.previous is not None:
                moved = advance_box(self.previous, frame, target.box)
                if moved is not None:
                    target.box = moved
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
                    target.box = np.asarray(boxes[index], np.float32).copy()
                    target.observed_box = target.box.copy()
                    target.last_seen = now
                    target.visible = target.detected = True
                    target.gap_frames = 0
                    matched.add(index)
        for column, box in enumerate(boxes):
            if column not in matched:
                if any(duplicate_candidate_boxes(box, measured.box)
                       for measured in measured_previous):
                    continue
                target = Target(self.next_id, np.asarray(box, np.float32).copy(), now,
                                observed_box=np.asarray(box, np.float32).copy())
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

    def __init__(self, loader, image_size=None, grace_s=.25, max_hold_frames=6):
        self.image_size = image_size
        self.grace_s, self.max_hold_frames = grace_s, max_hold_frames
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='multi-pose')
        self.loading = self.executor.submit(loader)
        self.pending = None
        self.pending_trail = None
        self.pending_refresh = False
        self.stats = dict(submitted=0, no_reference_pose=0, target_lost=0, expired=0,
                          history_lost=0, handoff_failed=0, detector_conflict=0,
                          superseded=0, accepted=0, corrected=0)
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
            if new_frame:
                target.tracked_frames += 1
            previous_pose = target.pose if target.pose is not None else target.last_good_pose
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
            target.pose = target.last_good_pose = pose
            from .cad_video_tracker import project
            pixels = project(initializer.vertices, pose.rvec, pose.tvec,
                             initializer.K, initializer.dist)
            height, width = frame.shape[:2]
            bounds = np.clip(np.r_[pixels.min(0), pixels.max(0)],
                             [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
            if np.isfinite(bounds).all() and np.min(bounds[2:] - bounds[:2]) >= 20.:
                target.box = bounds.astype(np.float32)
            target.pose_source, target.pose_box = frame_source, target.box.copy()
            target.pose_time = now
            target.failed_frames = 0
            target.tracking_failure = None
            target.initialization_failure = None

        for target in targets:
            base = target.last_good_pose if target.last_good_pose is not None else target.pose
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
                if pose is not None and (not target.detected or detection_matches_pose(
                        target.box, initializer.vertices, pose.rvec, pose.tvec,
                        initializer.K, initializer.dist, frame.shape)):
                    accept(target, pose)
                else:
                    target.pose = None
                    target.tracking_failure = ('flow_or_pnp' if pose is None else 'detector_conflict')
            elif base is not None:
                target.pose = None
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
                    elif not detection_matches_pose(
                            target.box, initializer.vertices, pose.rvec, pose.tvec,
                            initializer.K, initializer.dist, frame.shape):
                        reason = 'detector_conflict'
                    elif target.pose is not None and not reference_improves_pose(
                            target.pose, pose, initializer):
                        reason = 'superseded'
                    else:
                        if target.pose is not None:
                            self.stats['corrected'] += 1
                        accept(target, pose, reference=True)
                        self.stats['accepted'] += 1
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
            previous_display_pose = target.display_pose
            if target.pose is not None:
                # Roll around the model's long (local Z) axis is deliberately
                # ignored in the rendered overlay. Keep the measured pose for
                # tracking and metrics, but stabilize the displayed X/Y axes.
                target.display_pose = lock_long_axis_twist(
                    target.pose, previous_display_pose)
                if previous_display_pose is None:
                    target.display_rotation_deg = 0.
                else:
                    try:
                        old_rotation, _ = cv2.Rodrigues(
                            np.asarray(previous_display_pose.rvec,
                                       np.float64).reshape(3))
                        new_rotation, _ = cv2.Rodrigues(
                            np.asarray(target.display_pose.rvec,
                                       np.float64).reshape(3))
                        cosine = np.clip(
                            (np.trace(new_rotation @ old_rotation.T) - 1.) / 2.,
                            -1., 1.)
                        target.display_rotation_deg = float(
                            np.rad2deg(np.arccos(cosine)))
                    except (cv2.error, TypeError, ValueError):
                        target.display_rotation_deg = 0.
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
            if (moved is not None and np.max(np.abs(moved - target.pose_box)) <= 4.
                    and detection_matches_pose(target.box, initializer.vertices,
                        base.rvec, base.tvec, initializer.K, initializer.dist, frame.shape)):
                target.display_pose = lock_long_axis_twist(
                    base, previous_display_pose)
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
            self.pending = self.executor.submit(target.search.estimate, source, box,
                                                validation_box=box)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def project_target_box(target, initializer, *, display=False):
    """Project the metric CAD bounds using this instance's measured R and t.

    No screen-space extrusion, neutralized roll or cosmetic position offset is
    applied. Both returned arrays use the same corner ordering as BOX_EDGES.
    """
    pose = target.display_pose if display and target.display_pose is not None else target.pose
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
    return dict(id=target.id, box_xyxy=target.box.tolist(),
                detected=target.detected,
                tracking_failure=target.tracking_failure,
                initialization_failure=target.initialization_failure,
                tracking_points=(len(pose.points) if pose is not None and hasattr(pose, 'points') else 0),
                display_held=target.display_held,
                display_visible=project_target_box(target, initializer, display=True) is not None,
                state='pose_verified' if pose is not None else 'detected',
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
    first_pose = first.display_pose if first.display_pose is not None else first.pose
    second_pose = second.display_pose if second.display_pose is not None else second.pose
    if first_pose is None or second_pose is None:
        return False
    if duplicate_candidate_boxes(first.box, second.box):
        return True
    if similar_candidate_boxes(first.box, second.box):
        return True
    if initializer is None:
        return False
    first_geometry = project_target_box(first, initializer, display=True)
    second_geometry = project_target_box(second, initializer, display=True)
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
                 outline_seeder=None, show_outline=True, display_y_offset_px=0.):
    # Keep show_box for existing callers; the multi-target preview is contour-only.
    canvas = frame.copy()
    edge_distance = None
    edge_distance_ready = False
    for target in display_target_subset(targets, initializer):
        if project_target_box(target, initializer, display=True) is None:
            continue
        pose = target.display_pose if target.display_pose is not None else target.pose
        if show_outline and outline_seeder is not None:
            # Refit more often while the target is moving. A long interval
            # leaves a display-only translation behind the current pose during
            # fast hand motion; a short, bounded interval keeps the contour
            # responsive without running the edge search on every target on
            # every frame. A detector extent is only an ROI hint.
            moving = (target.display_motion_px >= 8. or
                      target.display_rotation_deg >= 2.)
            # During a hold, the metric pose can move by a few pixels because
            # of depth/feature noise. Re-running the edge fit every few frames
            # lets an unrelated background edge pull the contour around. Keep
            # the locked correction for a while and use consensus before any
            # stationary update. Motion still gets the short refit cadence.
            refit_interval = (2 if moving else
                              (8 if target.display_affine is None else 24))
            if moving:
                target.display_candidate_affine = None
                target.display_candidate_hits = 0
            if target.display_affine is not None:
                # Discard corrections produced by older builds or callers.
                # Only an identity linear part is valid for live display;
                # translation may be carried for at most the short refit
                # interval while the current pose is being updated.
                affine = np.asarray(target.display_affine, np.float32)
                if (affine.shape != (2, 3) or not np.isfinite(affine).all()
                        or np.max(np.abs(affine[:, :2] - np.eye(2, dtype=np.float32))) > 1e-4
                        or np.max(np.abs(affine[:, 2])) > 24.):
                    target.display_affine = None
                    target.display_candidate_affine = None
                    target.display_candidate_hits = 0
            if (target.display_affine is None or
                    target.tracked_frames - target.display_refit_at >= refit_interval):
                if not edge_distance_ready:
                    edge_distance = display_edge_distance(frame)
                    edge_distance_ready = True
                candidate = estimate_display_affine(
                    frame, initializer.vertices, initializer.K, initializer.dist,
                    pose.rvec, pose.tvec, outline_seeder,
                    display_box=(target.observed_box if target.detected else None),
                    # Recent RGB-D frames show a stable 15-20 px pose/display
                    # offset. Keep the correction translation-only, but let the
                    # edge fit cover that measured range instead of leaving the
                    # contour visibly above the object. Moving targets use a
                    # smaller bound so a blurred background edge cannot pull the
                    # overlay too far in one frame.
                    max_shift_px=(16. if moving else 20.),
                    edge_distance=edge_distance,
                    min_improvement_px=(.85 if moving else .55),
                    allow_shape_adjustment=False)
                if candidate is not None:
                    if moving:
                        # Motion needs immediate feedback. The correction is
                        # still translation-only and bounded by the estimator.
                        if target.display_affine is None:
                            target.display_affine = candidate
                        else:
                            target.display_affine = (.55 * candidate +
                                                     .45 * target.display_affine)
                    else:
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
                pose.rvec, pose.tvec, outline_seeder,
                display_y_offset_px=display_y_offset_px, show_axes=show_axes,
                contour_color=target_color(target.id),
                display_affine=target.display_affine)
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
            cap = cv2.VideoCapture(str(args.video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video: {args.video}')
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
                    display_y_offset_px=args.box_display_y_offset_px or 0.)
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
