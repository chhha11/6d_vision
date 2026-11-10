"""Timestamped camera ego-motion interface for future vehicle integration.

The live 6DoF tracker currently reports object poses in the RGB camera frame.
This module keeps the camera-motion boundary explicit so wheel odometry, IMU,
visual odometry, or SLAM can be connected later without changing the object
tracker's public pose representation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_EPS = 1e-6


def validate_transform(matrix) -> np.ndarray:
    """Return a finite 4x4 homogeneous transform or raise ``ValueError``."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("ego-motion pose must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0., 0., 0., 1.], atol=_EPS):
        raise ValueError("ego-motion pose must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("ego-motion rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.) > 1e-4:
        raise ValueError("ego-motion rotation must have determinant +1")
    return transform.copy()


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in a world/reference frame at one capture timestamp."""

    timestamp: float
    matrix: np.ndarray
    frame: str = "world"

    def __post_init__(self):
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "matrix", validate_transform(self.matrix))


class EgoMotionProvider:
    """Interface for timestamped camera motion.

    ``pose_at(t)`` returns :math:`T_{world\leftarrow camera}(t)`.  The default
    implementation is identity, which is correct for the current bench setup
    where no wheel/IMU/SLAM source is available.
    """

    source = "identity"

    def pose_at(self, timestamp: float) -> np.ndarray:
        del timestamp
        return np.eye(4, dtype=np.float64)

    def relative_camera_transform(self, from_timestamp: float,
                                  to_timestamp: float) -> np.ndarray:
        """Return :math:`T_{camera(to)\leftarrow camera(from)}`."""
        old_camera_in_world = validate_transform(self.pose_at(from_timestamp))
        new_camera_in_world = validate_transform(self.pose_at(to_timestamp))
        return validate_transform(np.linalg.inv(new_camera_in_world)
                                  @ old_camera_in_world)


class IdentityEgoMotion(EgoMotionProvider):
    """Explicit no-motion provider used by the current camera/target bench."""

    source = "identity"


class SampledEgoMotion(EgoMotionProvider):
    """Small timestamped provider useful for replay and synthetic tests.

    Samples are camera poses in a common world frame.  Linear interpolation is
    used for translation and the nearest sample is used for rotation; this is
    deliberately conservative until a real odometry/SLAM source is connected.
    """

    source = "sampled"

    def __init__(self, samples):
        ordered = sorted((CameraPose(t, matrix) for t, matrix in samples),
                         key=lambda sample: sample.timestamp)
        if not ordered:
            raise ValueError("at least one ego-motion sample is required")
        self._samples = ordered

    def pose_at(self, timestamp: float) -> np.ndarray:
        timestamp = float(timestamp)
        if timestamp <= self._samples[0].timestamp:
            return self._samples[0].matrix.copy()
        if timestamp >= self._samples[-1].timestamp:
            return self._samples[-1].matrix.copy()
        right = next(index for index, sample in enumerate(self._samples)
                     if sample.timestamp >= timestamp)
        left = self._samples[right - 1]
        upper = self._samples[right]
        ratio = ((timestamp - left.timestamp) /
                 max(upper.timestamp - left.timestamp, _EPS))
        matrix = left.matrix.copy()
        matrix[:3, 3] = ((1. - ratio) * left.matrix[:3, 3] +
                          ratio * upper.matrix[:3, 3])
        return matrix
