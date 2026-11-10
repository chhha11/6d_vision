"""Verify a tracked pose against object features from its depth initialization."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .reference_pose_corrector import (CONTROL_POINTS, ReferencePoseCorrector,
                                       StaticObservation, axis_angle, project_points)


@dataclass(frozen=True)
class PoseRecoveryProposal:
    observation: StaticObservation
    center_error_px: float
    axis_error_deg: float
    scale_ratio: float


def axis_angle_difference(first, second):
    """Signed angle between undirected image axes, in [-90, 90)."""
    return (float(first) - float(second) + 90.0) % 180.0 - 90.0


def observe_long_axis(frame, box):
    """Independent image-axis evidence from long parallel lines inside a detection.

    This only validates a pose; it cannot determine out-of-plane tilt or depth.
    """
    if box is None:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if min(x2 - x1, y2 - y1) < 20:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if lines is None:
        return None
    vectors = lines.reshape(-1, 4)[:, 2:] - lines.reshape(-1, 4)[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths >= max(18., max(gray.shape) * .25)
    if np.count_nonzero(keep) < 3:
        return None
    lengths, vectors = lengths[keep], vectors[keep]
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180.
    differences = np.abs((angles[:, None] - angles[None, :] + 90.) % 180. - 90.)
    support = differences <= 10.
    weights = lengths ** 2
    best = int(np.argmax(support @ weights))
    cluster = support[best]
    if (np.count_nonzero(cluster) < 3 or weights[cluster].sum() < weights.sum() * .65
            or lengths[cluster].max() < max(gray.shape) * .40):
        return None
    doubled = np.radians(angles[cluster] * 2.)
    return float(np.degrees(np.arctan2(np.sum(weights[cluster] * np.sin(doubled)),
                                        np.sum(weights[cluster] * np.cos(doubled)))) * .5)


def axis_matches_pose(observed_angle, rvec, tvec, K, dist):
    if observed_angle is None:
        return True
    controls = project_points(CONTROL_POINTS, rvec, tvec, K, dist)
    if np.linalg.norm(controls[1] - controls[2]) < 25.:
        return True
    return abs(axis_angle_difference(observed_angle, axis_angle(controls))) <= 18.


def directed_long_axis_vector(rvec, tvec, K, dist, min_length_px=20.):
    """Return the image vector from the model origin toward local ``+Z``.

    ``axis_matches_pose`` intentionally treats the detected line as undirected:
    a line detector cannot tell ``+Z`` from ``-Z``.  That is useful for a
    standalone validation, but it also admits the 180-degree PnP branch for a
    symmetric elongated target.  A tracked pose has a previous directed axis,
    so this helper exposes the sign needed by the temporal continuity gate.
    When the axis is nearly foreshortened, its sign is not observable and the
    caller should leave the candidate unconstrained.
    """
    controls = project_points(CONTROL_POINTS, rvec, tvec, K, dist)
    vector = np.asarray(controls[1] - controls[0], np.float64).reshape(2)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or length < float(min_length_px):
        return None
    return vector / length


def directed_long_axis_jump_deg(previous_rvec, previous_tvec, rvec, tvec,
                                K, dist, min_length_px=20.):
    """Measure the signed image-axis change between two metric poses.

    The returned value is in ``[0, 180]``.  Unlike ``axis_angle_difference`` it
    preserves the distinction between a continuous turn and a sudden reversal
    of the model's local long-axis direction.
    """
    previous = directed_long_axis_vector(previous_rvec, previous_tvec, K, dist,
                                         min_length_px)
    current = directed_long_axis_vector(rvec, tvec, K, dist, min_length_px)
    if previous is None or current is None:
        return None
    cosine = float(np.clip(np.dot(previous, current), -1., 1.))
    return float(np.degrees(np.arccos(cosine)))


def pose_has_continuous_long_axis(previous_pose, pose, K, dist,
                                  max_jump_deg=120., min_length_px=20.):
    """Reject a sudden 180-degree branch switch in a live pose track.

    A real target can eventually rotate through 180 degrees, but at camera
    frame rates it must pass through intermediate directions.  Returning true
    when either projected axis is too short keeps the gate from inventing a
    sign while the object is viewed end-on.
    """
    if previous_pose is None or pose is None:
        return True
    jump = directed_long_axis_jump_deg(
        previous_pose.rvec, previous_pose.tvec, pose.rvec, pose.tvec,
        K, dist, min_length_px)
    return jump is None or jump <= float(max_jump_deg)


class RgbPoseRecovery:
    """Use a confirmed object reference to detect correspondence drift.

    Only image-plane rotation, center and apparent scale are recovered here.
    Unobserved changes in out-of-plane tilt still need a new depth pose.
    """

    def __init__(self, frame, rvec, tvec, vertices, K, dist):
        self.corrector = ReferencePoseCorrector(
            frame, rvec, tvec, vertices, K, dist, reference_name='depth-initialization')
        self.reference_rvec = np.asarray(rvec, np.float64).reshape(3).copy()
        self.reference_tvec = np.asarray(tvec, np.float64).reshape(3).copy()
        self.reference_axis_length = float(np.linalg.norm(
            self.corrector.reference_controls[1] - self.corrector.reference_controls[2]))
        self.reference_area = float(cv2.contourArea(cv2.convexHull(
            project_points(vertices, rvec, tvec, K, dist).astype(np.float32))))
        projected = project_points(vertices, rvec, tvec, K, dist)
        self.reference_box_size = np.maximum(
            projected.max(axis=0) - projected.min(axis=0), 20.0)

    def observe(self, frame):
        return self.corrector.observe(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def observation_box(self, observation, image_shape, padding=.75):
        """Estimate a generous current-frame box from independent reference features."""
        if observation is None:
            return None
        controls = np.asarray(observation.control_points, np.float64).reshape(-1, 2)
        if controls.shape != (3, 2) or not np.isfinite(controls).all():
            return None
        observed_length = float(np.linalg.norm(controls[1] - controls[2]))
        if observed_length < 15.0 or self.reference_axis_length < 15.0:
            return None
        scale = np.clip(observed_length / self.reference_axis_length, .35, 2.5)
        size = self.reference_box_size * scale * (1.0 + 2.0 * padding)
        center = controls[0]
        height, width = image_shape[:2]
        box = np.r_[center - size * .5, center + size * .5]
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            return None
        return box.astype(np.float32)

    def verifies_pose(self, observation, rvec, tvec):
        """Strong object appearance can disambiguate an oversized YOLO box.

        This only corroborates an existing measured pose; it never estimates a
        new depth or overrides independent long-axis evidence.
        """
        if (observation is None or observation.inliers < 20 or
                observation.inlier_ratio < .75 or
                observation.spread_px2 < max(300., self.reference_area * .18)):
            return False
        center, angle, scale = self.corrector.residual(observation, rvec, tvec)
        return bool(np.isfinite(center).all() and np.isfinite(angle) and
                    np.isfinite(scale) and scale > 0 and np.linalg.norm(center) <= 6.
                    and abs(angle) <= 6. and abs(math.log(scale)) <= .08)

    def fast_recovery_ready(self, proposal):
        """Allow one-frame correction only for strong, distributed evidence."""
        if proposal is None:
            return False
        observation = proposal.observation
        return bool(observation.inliers >= 28 and observation.inlier_ratio >= .80 and
                    observation.spread_px2 >= max(1000., self.reference_area * .18) and
                    (proposal.center_error_px >= 18.0 or proposal.axis_error_deg >= 14.0 or
                     abs(math.log(max(proposal.scale_ratio, 1e-6))) >= .18))

    def proposal(self, observation, rvec, tvec, detected_box=None):
        if observation is None or self.reference_axis_length < 20.0:
            return None
        # Repeated rails can match each other: require support over the object,
        # rather than accepting a small patch with a low reprojection error.
        if (observation.inliers < 12 or observation.inlier_ratio < .65 or
                observation.spread_px2 < max(150.0, self.reference_area * .10)):
            return None
        controls = observation.control_points
        if not np.isfinite(controls).all():
            return None
        observed_length = float(np.linalg.norm(controls[1] - controls[2]))
        reference_scale = observed_length / self.reference_axis_length
        if not .55 <= reference_scale <= 1.8:
            return None
        if detected_box is not None:
            box = np.asarray(detected_box, np.float64).reshape(4)
            margin = .15 * np.maximum(box[2:] - box[:2], 1.0)
            if np.any(controls[0] < box[:2] - margin) or np.any(controls[0] > box[2:] + margin):
                return None
        current = project_points(CONTROL_POINTS, rvec, tvec,
                                 self.corrector.K, self.corrector.dist)
        current_length = float(np.linalg.norm(current[1] - current[2]))
        if current_length < 15.0:
            return None
        center_error = float(np.linalg.norm(controls[0] - current[0]))
        angle_error = abs(axis_angle_difference(axis_angle(controls), axis_angle(current)))
        scale_ratio = observed_length / current_length
        if center_error <= 12.0 and angle_error <= 12.0 and abs(math.log(scale_ratio)) <= .15:
            return None
        return PoseRecoveryProposal(observation, center_error, angle_error, scale_ratio)

    @staticmethod
    def consistent(first, second):
        if first is None or second is None:
            return False
        a, b = first.observation.control_points, second.observation.control_points
        length_a = float(np.linalg.norm(a[1] - a[2]))
        length_b = float(np.linalg.norm(b[1] - b[2]))
        return bool(np.linalg.norm(a[0] - b[0]) <= max(15.0, length_b * .2) and
                    abs(axis_angle_difference(axis_angle(a), axis_angle(b))) <= 10.0 and
                    abs(math.log(length_b / max(length_a, 1e-6))) <= .15)

    def recover(self, proposal):
        # Always use the depth reference. Applying a partial correction to the
        # drifted PnP pose would preserve its incorrect out-of-plane orientation.
        return self.corrector.apply(
            proposal.observation, self.reference_rvec, self.reference_tvec,
            gain=1.0, max_center_step_px=1000.0, max_angle_step_deg=180.0,
            max_depth_step_mm=1000.0)[:2]
