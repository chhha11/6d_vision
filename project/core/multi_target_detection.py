"""Collect every energy-unit detection, including rotated views of the frame."""
from __future__ import annotations

import cv2
import numpy as np

from .cad_video_tracker import box_overlap, restore_rotated_box


def diagonal_view(frame, angle):
    """Rotate at the original pixel scale for the detector's trained image size.

    The original/quarter-turn views still cover the corners clipped by this view.
    Expanding then resizing the canvas shrinks small units and loses features.
    """
    height, width = frame.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2., height / 2.), angle, 1.)
    rotated = cv2.warpAffine(frame, matrix, (width, height), borderValue=(114, 114, 114))
    return rotated, cv2.invertAffineTransform(matrix)


def restore_diagonal_box(box, inverse, shape):
    x1, y1, x2, y2 = np.asarray(box, np.float64).reshape(4)
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
    restored = cv2.transform(corners, inverse)[0]
    bounds = np.r_[restored.min(axis=0), restored.max(axis=0)]
    height, width = shape[:2]
    return np.clip(bounds, [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)


def detect_target_boxes(detector, frame, fallback_rotation=2, *, progressive=False):
    """Return confidence-ordered boxes in camera coordinates, with cross-view NMS.

    Always examine the fallback view: one upright object must not hide another
    object which is detectable only when the image is rotated.
    """
    if hasattr(detector, 'get_boxes'):
        if progressive:
            return detector.get_boxes(frame, fallback_rotation=fallback_rotation, progressive=True)
        return detector.get_boxes(frame, fallback_rotation=fallback_rotation)
    from .live_depth_6dof import plausible_target_box

    candidates = []
    for turn in dict.fromkeys((0, fallback_rotation % 4)):
        source = np.ascontiguousarray(np.rot90(frame, turn)) if turn else frame
        result = detector.predict(source, conf=.15, imgsz=640, verbose=False)[0]
        for index in range(len(result.boxes.conf)):
            score = float(result.boxes.conf[index])
            box = restore_rotated_box(result.boxes.xyxy[index].cpu().numpy(),
                                      frame.shape, turn)
            if np.isfinite(score) and plausible_target_box(box, frame.shape):
                candidates.append((score, box))
    # Quarter turns leave a hand-held diagonal object diagonal. Search both
    # diagonal directions as well, even when another upright unit was found.
    # Live requests cycle quarter turns 1, 3, 2. Split the diagonal sweep
    # across those requests so an upright detection need not wait for 8 passes.
    angles = ((-30., 30.) if fallback_rotation % 4 == 1 else
              (-45., 45.) if fallback_rotation % 4 == 3 else (-60., 60.))
    for angle in angles if progressive else (-60., -45., -30., 30., 45., 60.):
        source, inverse = diagonal_view(frame, angle)
        result = detector.predict(source, conf=.15, imgsz=640, verbose=False)[0]
        for index in range(len(result.boxes.conf)):
            score = float(result.boxes.conf[index])
            raw = result.boxes.xyxy[index].cpu().numpy()
            # Reject invalid source boxes before inverse rotation/clipping can
            # turn out-of-image or tiny predictions into plausible rectangles.
            if not plausible_target_box(raw, source.shape):
                continue
            box = restore_diagonal_box(raw, inverse, frame.shape)
            if np.isfinite(score) and plausible_target_box(box, frame.shape):
                candidates.append((score, box))
    kept = []
    for _, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(box_overlap(box, previous)[0] < .5 for previous in kept):
            kept.append(box)
    return kept


def detect_live_target_boxes(detector, frame, fallback_rotation=2):
    """Four inference passes per live round; all diagonal views in three rounds."""
    return detect_target_boxes(detector, frame, fallback_rotation, progressive=True)
