"""Optional YOLO assistance without inference in the camera/pose-search thread.

One load/inference at a time; new requests replace no queue. Detector boxes
are advanced to the requesting frame using checked image correspondences.
They are search hints, never a source of metric 6DoF.
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .frame_trail import FrameTrail


def box_corners(box):
    x1, y1, x2, y2 = np.asarray(box, np.float32)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)


def advance_box(source, current, box, *, corners=None):
    """Advance a search ROI; update optional caller-owned corners on success.

    Keeping the original quadrilateral avoids enlarging an axis-aligned box
    again at every small rotation. The returned ROI remains axis-aligned.
    """
    if source.shape != current.shape:
        return None
    if np.array_equal(source, current):
        return np.asarray(box, np.float32).copy()
    first = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    h, w = first.shape
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    mask = np.zeros_like(first)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(first, 100, .01, 5, mask=mask)
    if points is None or len(points) < 10:
        return None
    forward, valid, _ = cv2.calcOpticalFlowPyrLK(
        first, second, points, None, winSize=(31, 31), maxLevel=3)
    if forward is None or valid is None or not np.isfinite(forward).all():
        return None
    backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(
        second, first, forward, None, winSize=(31, 31), maxLevel=3)
    if backward is None or back_valid is None:
        return None
    good = (valid.ravel().astype(bool) & back_valid.ravel().astype(bool) &
            (np.linalg.norm(backward - points, axis=2).ravel() <= 1.))
    if np.count_nonzero(good) < 10:
        return None
    origin, target = points[good, 0], forward[good, 0]
    matrix, inliers = cv2.estimateAffinePartial2D(
        origin, target, method=cv2.RANSAC, ransacReprojThreshold=2.)
    if matrix is None or inliers is None:
        return None
    keep = inliers.ravel().astype(bool)
    spread = cv2.contourArea(cv2.convexHull(origin[keep])) if keep.sum() >= 3 else 0.
    if (keep.sum() < 10 or keep.mean() < .65 or
            spread < max(100., (x2-x1)*(y2-y1)*.08)):
        return None
    scale = np.linalg.norm(matrix[0, :2])
    if not .65 <= scale <= 1.5:
        return None
    polygon = box_corners([x1, y1, x2, y2]) if corners is None else corners
    moved = cv2.transform(polygon[None], matrix)[0]
    result = np.r_[moved.min(axis=0), moved.max(axis=0)]
    result = np.clip(result, [0,0,0,0], [w,h,w,h])
    if not np.all(result[2:] - result[:2] >= 20):
        return None
    if corners is not None:
        corners[:] = moved
    return result.astype(np.float32)


class BackgroundDetector:
    """Own the detector on one worker, including its slow import/warmup."""

    def __init__(self, loader, detect, max_age_s=.5, multiple=False):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='yolo-assist')
        self.loading = self.executor.submit(loader)
        self.detect = detect
        self.multiple = multiple
        self.max_age_s = max_age_s
        self.pending = None
        self.pending_trail = None
        self.observation = None
        self.observation_corners = None
        self.closed = False
        self.error = None
        self.lock = threading.Lock()
        self.stats = dict(submitted=0, completed=0, expired=0, failures=0, history_lost=0)

    @property
    def ready(self):
        return self.loading.done() and self.loading.exception() is None

    def _infer(self, frame, turn):
        return self.detect(self.loading.result(), frame, turn)

    def get_box(self, frame, **_hints):
        with self.lock:
            return self._get_box(frame)

    def get_boxes(self, frame, **_hints):
        if not self.multiple:
            raise RuntimeError('BackgroundDetector was not configured for multiple targets')
        with self.lock:
            boxes = self._get_box(frame)
            return [] if boxes is None else boxes

    def _get_box(self, frame):
        if self.closed:
            return None
        if self.loading.done() and self.loading.exception() is not None:
            self.error = str(self.loading.exception())
            self.stats['failures'] = 1
            return None
        if self.multiple and self.pending_trail is not None:
            self.pending_trail.append(frame)
        if self.pending is not None and self.pending.done():
            try:
                box = self.pending.result()
                source = self.source
                polygons = [box_corners(candidate) for candidate in box] if self.multiple and box is not None else None
                if self.multiple and box is not None and self.pending_trail is not None:
                    trail = self.pending_trail
                    if not trail.complete or time.monotonic() - self.submitted_at > self.max_age_s:
                        if not trail.complete:
                            self.stats['history_lost'] += 1
                        else:
                            self.stats['expired'] += 1
                        box = None
                    else:
                        for first, second in zip(trail.frames, trail.frames[1:]):
                            advanced = [(advance_box(first, second, candidate, corners=polygon), polygon)
                                        for candidate, polygon in zip(box, polygons)]
                            box = [candidate for candidate, _ in advanced if candidate is not None]
                            polygons = [polygon for candidate, polygon in advanced if candidate is not None]
                            if not box:
                                break
                        source = frame.copy()
                self.observation = (source, box, self.submitted_at) if box is not None else None
                self.observation_corners = polygons if box is not None else None
                self.stats['completed'] += 1
            except Exception as exc:
                self.error = str(exc)
                self.stats['failures'] += 1
                self.observation = None
            self.pending = None
            self.pending_trail = None
        # Readiness is polled, never awaited by the caller. Only copy the
        # newest frame when the worker is available; do not queue old frames.
        if self.ready and self.pending is None:
            self.source = frame.copy()
            if self.multiple:
                self.pending_trail = FrameTrail(self.source)
            self.submitted_at = time.monotonic()
            turn = (1, 3, 2)[self.stats['submitted'] % 3]
            self.pending = self.executor.submit(self._infer, self.source, turn)
            self.stats['submitted'] += 1
        if self.observation is None:
            return None
        source, box, captured_at = self.observation
        if time.monotonic() - captured_at > self.max_age_s:
            self.stats['expired'] += 1
            self.observation = None
            return None
        if self.multiple:
            polygons = self.observation_corners
            if polygons is None or len(polygons) != len(box):
                polygons = [box_corners(candidate) for candidate in box]
            advanced = [(advance_box(source, frame, candidate, corners=polygon), polygon)
                        for candidate, polygon in zip(box, polygons)]
            boxes = [candidate for candidate, _ in advanced if candidate is not None]
            self.observation_corners = [polygon for candidate, polygon in advanced if candidate is not None]
            # Once a delayed observation reaches the UI, advance it through
            # adjacent frames. Keep its ORIGINAL age: flow is not a new detection.
            self.observation = (frame.copy(), boxes, captured_at)
            return boxes
        return advance_box(source, frame, box)

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
            # The live loader owns a subprocess; close it only after in-flight
            # inference finishes, including when startup is cancelled early.
            if not self.loading.cancelled() and self.loading.exception() is None:
                close = getattr(self.loading.result(), 'close', None)
                if callable(close):
                    close()
