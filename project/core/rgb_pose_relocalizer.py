"""Confirm fresh metric RGB poses before replacing drifted flow correspondences."""
from __future__ import annotations

import numpy as np
import cv2

from .rgb_global_initializer import advance_pose


class RgbPoseRelocalizer:
    def __init__(self, initializer, reference_frame=None, reference_pose=None,
                 surface_seeder=None):
        if (reference_frame is not None and reference_pose is not None
                and surface_seeder is not None and hasattr(initializer, 'with_reference')):
            initializer = initializer.with_reference(
                reference_frame, *reference_pose, surface_seeder)
        self.initializer = initializer
        self.search = initializer.new_search()
        self.global_search = initializer.new_search()
        self.pending = None
        self.pending_at = 0.
        self.pending_frame = None
        self.pending_box = None
        self.stats = dict(checks=0, candidates=0, accepted=0, rejected_current=0,
                          unchanged=0, single_frame_accepted=0,
                          current_frame_advances=0, fresh_correction_proposals=0)

    def estimate(self, frame, box, validation_box=None):
        # Separate temporal banks keep alternating global/YOLO searches from
        # erasing each other's accumulated correspondences.
        h, w = frame.shape[:2]
        full = (box is None or (box[2]-box[0] >= w*.95 and box[3]-box[1] >= h*.95))
        search = self.global_search if full else self.search
        if box is None:
            box = np.array([0., 0., float(w), float(h)], np.float32)
        try:
            return search.estimate(frame, box, validation_box=validation_box)
        except TypeError as exc:
            if 'validation_box' not in str(exc):
                raise
            return search.estimate(frame, box)
        except (cv2.error, RuntimeError, ValueError):
            search.clear()
            return None

    @staticmethod
    def consistent(first, second):
        a, _ = cv2.Rodrigues(first.rvec)
        b, _ = cv2.Rodrigues(second.rvec)
        return bool(np.linalg.norm(first.tvec - second.tvec) <= 25.
                    and abs(float(np.dot(a[:, 2], b[:, 2]))) >= np.cos(np.deg2rad(12.)))

    def _clear_pending(self):
        self.pending = None
        self.pending_frame = None
        self.pending_box = None

    def confirm(self, pose, source, current, box, now, rvec, tvec, tracking_valid,
                allow_single_frame=False):
        """Confirm a metric RGB pose against the newest camera frame.

        Every candidate is first advanced to ``current``. If confirmation
        takes two observations, the older pending candidate is advanced to the
        same current image before it is compared. This keeps a moving target
        from being rejected merely because the two confirmations were captured
        at different times.
        """
        self.stats['checks'] += 1
        if pose is None:
            self._clear_pending()
            return None
        self.stats['candidates'] += 1
        initializer = self.initializer
        try:
            advanced = advance_pose(pose, source, current, initializer.K, initializer.dist,
                                    initializer.vertices, box)
        except (cv2.error, RuntimeError, ValueError):
            advanced = None
        if advanced is None:
            self._clear_pending()
            self.stats['rejected_current'] += 1
            return None
        self.stats['current_frame_advances'] += 1
        rotation, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
        candidate_rotation, _ = cv2.Rodrigues(advanced.rvec)
        axis_error = np.degrees(np.arccos(np.clip(abs(float(np.dot(
            rotation[:, 2], candidate_rotation[:, 2]))), 0., 1.)))
        center_error = np.linalg.norm(np.asarray(tvec) - advanced.tvec)
        # A fresh, high-quality current-frame solution should be allowed to
        # correct small but visible drift. The old 6 deg / 12 mm deadband made
        # slow motion remain attached to the previous pose.
        fresh_quality = (len(getattr(advanced, 'points', ())) >= 16 and
                         advanced.rms_px <= 2.4 and
                         advanced.inlier_ratio >= .55)
        strong_single = (allow_single_frame and not tracking_valid and
                         # Two independent reference views plus current-frame
                         # PnP evidence are enough to restore immediately. The
                         # old 20/1.8/.75 gate made a valid reappearance wait
                         # for another detector cycle, which is visible as a
                         # long SEARCHING interval after occlusion.
                         len(getattr(advanced, 'points', ())) >= 18 and
                         getattr(advanced, 'reference_support', 1) >= 2 and
                         advanced.rms_px <= 2.0 and
                         advanced.inlier_ratio >= .72)
        if strong_single:
            self._clear_pending()
            self.stats['single_frame_accepted'] += 1
            return advanced

        # During valid tracking, a fresh metric solution can correct slow
        # optical-flow drift immediately. Require distributed evidence and a
        # bounded correction so this path fixes lag without becoming a free
        # frame-to-frame pose switch.
        fresh_correction = (tracking_valid and fresh_quality and
                            (getattr(advanced, 'reference_support', 1) >= 2 or
                             len(getattr(advanced, 'points', ())) >= 24) and
                            (axis_error >= .8 or center_error >= 2.5) and
                            axis_error <= 8.0 and center_error <= 20.0)
        if fresh_correction:
            self._clear_pending()
            self.stats['fresh_correction_proposals'] += 1
            return advanced

        # Keep the wider deadband for weaker evidence; strong bounded
        # corrections above must be considered before this noise filter.
        if tracking_valid and axis_error <= 1.8 and center_error <= 4.5:
            self.stats['unchanged'] += 1
            self._clear_pending()
            return None

        comparison = self.pending
        if (comparison is not None and self.pending_frame is not None and
                not np.array_equal(self.pending_frame, current)):
            try:
                comparison = advance_pose(
                    comparison, self.pending_frame, current, initializer.K, initializer.dist,
                    initializer.vertices, self.pending_box)
            except (cv2.error, RuntimeError, ValueError):
                comparison = None
            if comparison is None:
                self._clear_pending()
        confirmed = (comparison is not None and now - self.pending_at <= .6 and
                     self.consistent(comparison, advanced))
        self.pending = advanced
        self.pending_at = now
        self.pending_frame = current.copy()
        self.pending_box = None if box is None else np.asarray(box, np.float32).copy()
        return advanced if confirmed else None

