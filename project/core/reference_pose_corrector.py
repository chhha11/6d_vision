"""Conservative static pose corrections from a confirmed RGB reference view."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


CONTROL_POINTS = np.array([[0, 0, 0], [0, 0, 75], [0, 0, -75]], np.float64)


def project_points(points, rvec, tvec, K, dist):
    uv, _ = cv2.projectPoints(np.asarray(points, np.float64), np.asarray(rvec).reshape(3, 1),
                              np.asarray(tvec).reshape(3, 1), K, dist)
    return uv.reshape(-1, 2)


def wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def axis_angle(points):
    vector = points[1] - points[2]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def make_projection_mask(shape, vertices, rvec, tvec, K, dist, erode_px=9):
    projected = project_points(vertices, rvec, tvec, K, dist).astype(np.float32)
    valid = np.isfinite(projected).all(axis=1) & (np.abs(projected) < 1e5).all(axis=1)
    mask = np.zeros(shape[:2], np.uint8)
    if valid.sum() < 3:
        return mask
    hull = cv2.convexHull(projected[valid]).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    if erode_px > 1:
        mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
    return mask


@dataclass(frozen=True)
class StaticObservation:
    reference_name: str
    control_points: np.ndarray
    inliers: int
    match_count: int
    inlier_ratio: float
    spread_px2: float
    elapsed_ms: float


class ReferencePoseCorrector:
    """Match one manually confirmed object view and return image-space controls."""

    def __init__(self, reference_bgr, reference_rvec, reference_tvec, vertices, K, dist,
                 reference_name='confirmed', ratio_test=.74):
        self.K = np.asarray(K, np.float64)
        self.dist = np.asarray(dist, np.float64)
        self.reference_name = reference_name
        self.ratio_test = float(ratio_test)
        self.detector = cv2.AKAZE_create(threshold=.0001)
        gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        mask = make_projection_mask(gray.shape, vertices, reference_rvec, reference_tvec,
                                    self.K, self.dist)
        self.reference_keypoints, self.reference_descriptors = self.detector.detectAndCompute(
            gray, mask)
        if self.reference_descriptors is None or len(self.reference_keypoints) < 12:
            raise RuntimeError('Confirmed reference image has too few AKAZE features')
        self.reference_controls = project_points(CONTROL_POINTS, reference_rvec, reference_tvec,
                                                 self.K, self.dist).astype(np.float32)

    def observe(self, gray, search_mask=None):
        started = time.perf_counter()
        keypoints, descriptors = self.detector.detectAndCompute(gray, search_mask)
        if descriptors is None or len(keypoints) < 4:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(self.reference_descriptors, descriptors, k=2)
        matches = [first for first, second in pairs
                   if first.distance < self.ratio_test * second.distance]
        unique_targets = {}
        for match in matches:
            previous = unique_targets.get(match.trainIdx)
            if previous is None or match.distance < previous.distance:
                unique_targets[match.trainIdx] = match
        matches = list(unique_targets.values())
        if len(matches) < 4:
            return None
        source = np.float32([self.reference_keypoints[m.queryIdx].pt for m in matches])
        target = np.float32([keypoints[m.trainIdx].pt for m in matches])
        affine, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0,
            maxIters=2000, confidence=.995, refineIters=10)
        if affine is None or mask is None:
            return None
        inlier_mask = mask.reshape(-1).astype(bool)
        inliers = int(inlier_mask.sum())
        if inliers < 4:
            return None
        spread = float(cv2.contourArea(cv2.convexHull(source[inlier_mask]))) if inliers >= 3 else 0.0
        controls = cv2.transform(self.reference_controls[None], affine)[0]
        if not np.isfinite(controls).all():
            return None
        return StaticObservation(
            reference_name=self.reference_name,
            control_points=controls.astype(np.float64),
            inliers=inliers,
            match_count=len(matches),
            inlier_ratio=inliers / len(matches),
            spread_px2=spread,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def consistent(first, second, center_tolerance=2.0, angle_tolerance=1.0,
                   scale_tolerance=.02):
        if first is None or second is None or first.reference_name != second.reference_name:
            return False
        center_error = np.linalg.norm(first.control_points[0] - second.control_points[0])
        angle_error = abs(wrap_degrees(axis_angle(first.control_points) -
                                       axis_angle(second.control_points)))
        first_length = np.linalg.norm(first.control_points[1] - first.control_points[2])
        second_length = np.linalg.norm(second.control_points[1] - second.control_points[2])
        scale_error = abs(second_length / max(first_length, 1e-6) - 1.0)
        return center_error <= center_tolerance and angle_error <= angle_tolerance and \
            scale_error <= scale_tolerance

    def residual(self, observation, rvec, tvec):
        current = project_points(CONTROL_POINTS, rvec, tvec, self.K, self.dist)
        center_delta = observation.control_points[0] - current[0]
        angle_delta = wrap_degrees(axis_angle(observation.control_points) - axis_angle(current))
        observed_length = np.linalg.norm(observation.control_points[1] - observation.control_points[2])
        current_length = np.linalg.norm(current[1] - current[2])
        scale_ratio = observed_length / max(current_length, 1e-6)
        return center_delta, angle_delta, float(scale_ratio)

    def apply(self, observation, rvec, tvec, gain=.25, max_center_step_px=2.0,
              max_angle_step_deg=.5, max_depth_step_mm=1.0):
        center_delta, angle_delta, scale_ratio = self.residual(observation, rvec, tvec)
        center_step = center_delta * gain
        length = float(np.linalg.norm(center_step))
        if length > max_center_step_px:
            center_step *= max_center_step_px / length
        angle_step = float(np.clip(angle_delta * gain, -max_angle_step_deg, max_angle_step_deg))
        target_depth = float(tvec[2]) / max(scale_ratio, 1e-6)
        depth_step = float(np.clip((target_depth - float(tvec[2])) * gain,
                                   -max_depth_step_mm, max_depth_step_mm))
        new_depth = float(tvec[2]) + depth_step

        current_center = project_points(CONTROL_POINTS[:1], rvec, tvec, self.K, self.dist)[0]
        desired_center = current_center + center_step
        normalized = cv2.undistortPoints(desired_center.reshape(1, 1, 2), self.K,
                                         self.dist).reshape(2)
        new_tvec = np.array([normalized[0] * new_depth, normalized[1] * new_depth,
                             new_depth], np.float64)
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        delta_rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(angle_step)]))
        new_rvec, _ = cv2.Rodrigues(delta_rotation @ rotation)
        diagnostics = dict(center_error_px=float(np.linalg.norm(center_delta)),
                           angle_error_deg=float(angle_delta), scale_ratio=scale_ratio,
                           center_step_px=float(np.linalg.norm(center_step)),
                           angle_step_deg=angle_step, depth_step_mm=depth_step)
        return new_rvec.reshape(3), new_tvec, diagnostics
