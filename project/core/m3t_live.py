"""Camera-free native region proposals, independently gated before display.

Native API success only means an optimizer ran. A proposal needs fresh image
support, plausible motion and CAD silhouette agreement to enter metric output.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import select
import struct
import subprocess
import time

import cv2
import numpy as np

from .paths import DATA_ROOT, PROJECT_ROOT

BINARY = PROJECT_ROOT.parent / 'build/m3t/m3t_stream'


@dataclass(frozen=True)
class RegionPose:
    rvec: np.ndarray
    tvec: np.ndarray
    processing_ms: float


class NativeRegionTracker:
    def __init__(self, model, K, dist, size, output, timeout_s=.25):
        from .evaluate_m3t import prepare_mesh
        if not BINARY.is_file():
            raise RuntimeError('M3T stream executable is unavailable; run scripts/setup_m3t_offline.py')
        cache = DATA_ROOT / 'tracking/m3t_cache'
        cache.mkdir(parents=True, exist_ok=True)
        mesh, digest = prepare_mesh(model, cache)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        config = output / 'config.yml'
        storage = cv2.FileStorage(str(config), cv2.FILE_STORAGE_WRITE)
        try:
            storage.write('mesh', str(mesh))
            storage.write('model_cache', str(cache / f'region_{digest[:12]}_d3_i1000_p200.bin'))
            storage.write('camera_matrix', np.asarray(K, np.float64))
            storage.write('distortion', np.asarray(dist, np.float64))
            storage.write('width', int(size[0]))
            storage.write('height', int(size[1]))
        finally:
            storage.release()
        self.size, self.timeout_s, self.sequence = tuple(size), timeout_s, 0
        self.process = None
        self.log = (output / 'native.log').open('ab')
        env = dict(os.environ, OMP_NUM_THREADS='1')
        try:
            self.process = subprocess.Popen([str(BINARY), str(config)], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=self.log,
                                             env=env, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            magic = self._read(4, time.monotonic() + 15.)
            if magic != struct.pack('<I', 0x4d335431):
                raise RuntimeError('M3T stream protocol handshake failed')
        except BaseException:
            self.close()
            raise

    def _read(self, length, deadline):
        data = bytearray()
        while len(data) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], max(remaining, 0))[0]:
                raise TimeoutError('M3T response timeout')
            block = os.read(self.process.stdout.fileno(), length - len(data))
            if not block:
                raise RuntimeError('M3T stream stopped; see native.log')
            data.extend(block)
        return bytes(data)

    def _write(self, payload, deadline):
        view = memoryview(payload)
        while len(view):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], max(remaining, 0))[1]:
                raise TimeoutError('M3T request timeout')
            count = os.write(self.process.stdin.fileno(), view)
            if count <= 0:
                raise RuntimeError('M3T stream write failed')
            view = view[count:]

    def step(self, frame, pose=None):
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8:
            raise ValueError('M3T frame differs from calibrated RGB size/type')
        params = np.zeros(6) if pose is None else np.r_[pose[0], pose[1]]
        if params.shape != (6,) or not np.isfinite(params).all():
            raise ValueError('M3T reset pose must be six finite values in radians/mm')
        self.sequence += 1
        header = struct.pack('<II6d', 1 if pose is not None else 2, self.sequence, *params)
        deadline = time.monotonic() + self.timeout_s
        self._write(header + np.ascontiguousarray(frame).tobytes(), deadline)
        response = self._read(64, deadline)
        sequence, ok, *values = struct.unpack('<II7d', response)
        if sequence != self.sequence or not ok or not np.isfinite(values).all():
            raise RuntimeError('M3T returned an invalid or unsynchronized proposal')
        return RegionPose(np.array(values[:3]), np.array(values[3:6]), values[6])

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        self.log.close()


def validate_region_pose(frame, pose, previous, vertices, K, dist, seeder,
                         points, pixels, box=None, min_depth=150., max_depth=2400.):
    from .cad_video_tracker import (project, tracking_support_quality,
                                    detection_matches_pose, pose_delta, projected_coverage)
    from .rgb_pose_recovery import axis_matches_pose, observe_long_axis
    result = dict(accepted=False, reason='geometry', flow_support=0,
                  contour_support=0., contour_median_px=None, yolo_verified=False)
    if not np.isfinite(np.r_[pose.rvec, pose.tvec]).all() or not min_depth <= pose.tvec[2] <= max_depth:
        return result
    rotation = cv2.Rodrigues(pose.rvec)[0]
    if np.min((vertices @ rotation.T + pose.tvec)[:, 2]) <= 0:
        return result
    angle, translation = pose_delta(*previous, pose.rvec, pose.tvec)
    if angle > 45. or translation > 80. or projected_coverage(
            vertices, pose.rvec, pose.tvec, K, dist, frame.shape) < .5:
        return result
    points, pixels = np.asarray(points).reshape(-1,3), np.asarray(pixels).reshape(-1,2)
    flow_ok = False
    if len(points) >= 12 and len(points) == len(pixels):
        residual = np.linalg.norm(project(points, pose.rvec, pose.tvec, K, dist) - pixels, axis=1)
        keep = residual <= 3.5
        result['flow_support'] = int(keep.sum())
        flow_ok = (keep.sum() >= 12 and keep.mean() >= .6 and tracking_support_quality(
            points[keep], pixels[keep], vertices, pose.rvec, pose.tvec, K, dist))
    if box is not None:
        result['yolo_verified'] = bool(detection_matches_pose(
            box, vertices, pose.rvec, pose.tvec, K, dist, image_shape=frame.shape) and
            axis_matches_pose(observe_long_axis(frame, box), pose.rvec, pose.tvec, K, dist))
    if box is not None and not result['yolo_verified']:
        result['reason'] = 'detector_conflict'
        return result
    if not flow_ok and not result['yolo_verified']:
        result['reason'] = 'no_independent_image_support'
        return result
    mask = seeder.mask(frame.shape, vertices, pose.rvec, pose.tvec, erode=False)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return result
    outline = np.concatenate(contours).reshape(-1,2)
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
    if not np.any(edges):
        result['reason'] = 'silhouette_mismatch'
        return result
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    values = distance[outline[:,1], outline[:,0]]
    support, median = float(np.mean(values <= 3.)), float(np.median(values))
    result.update(contour_support=support, contour_median_px=median)
    # A detector box cannot validate orientation alone. Require tighter contour
    # support when image/CAD point correspondences are unavailable.
    result['accepted'] = bool(support >= (.60 if flow_ok else .78) and
                              median <= (3. if flow_ok else 1.8))
    result['reason'] = 'accepted' if result['accepted'] else 'silhouette_mismatch'
    return result


class RegionAssist:
    def __init__(self, native):
        self.native = native
        self.initialized = False
        self.previous = None
        self.rejects = 0
        self.disabled = False
        self.last_quality = None
        self.stats = dict(resets=0, proposals=0, accepted=0, applied=0, rejected=0, errors=0,
                          native_ms=0., reasons={})

    def update(self, frame, rvec, tvec, certified, vertices, K, dist, seeder,
               points, pixels, box, min_depth, max_depth):
        if self.disabled:
            return None
        try:
            if not self.initialized:
                if certified:
                    self.native.step(frame, (rvec, tvec))
                    self.previous = (rvec.copy(), tvec.copy())
                    self.initialized, self.rejects = True, 0
                    self.stats['resets'] += 1
                return None
            pose = self.native.step(frame)
            self.stats['proposals'] += 1
            self.stats['native_ms'] += pose.processing_ms
            quality = validate_region_pose(frame, pose, self.previous, vertices, K, dist,
                                            seeder, points, pixels, box, min_depth, max_depth)
            self.last_quality = quality
            reasons = self.stats['reasons']
            reasons[quality['reason']] = reasons.get(quality['reason'], 0) + 1
            if quality['accepted']:
                self.previous = pose.rvec.copy(), pose.tvec.copy()
                self.rejects = 0
                self.stats['accepted'] += 1
                return pose
            self.stats['rejected'] += 1
            self.rejects += 1
            if self.rejects >= 3:
                self.initialized = False
        except (RuntimeError, OSError, ValueError, cv2.error) as exc:
            self.disabled = True
            self.stats['errors'] += 1
            self.stats['error'] = str(exc)
            self.native.close()
            print(f'M3T assistance unavailable: {exc}; continuing RGB recovery', flush=True)
        return None
