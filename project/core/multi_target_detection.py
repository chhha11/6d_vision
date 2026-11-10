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


def _box_iou(first, second):
    first = np.asarray(first, np.float64).reshape(4)
    second = np.asarray(second, np.float64).reshape(4)
    intersection = np.maximum(
        np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2]), 0.)
    intersection_area = float(np.prod(intersection))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.)))
    return intersection_area / max(first_area + second_area - intersection_area, 1e-6)


def fuse_view_candidates(candidates, *, min_iou=.65, max_area_ratio=1.8):
    """Fuse repeated multi-view boxes before association.

    Rotating the same image changes the detector's axis-aligned rectangle even
    when the physical object is static.  Keeping the highest-confidence box
    makes that rectangle jump whenever the winning view changes.  This small
    weighted box fusion uses only strongly overlapping, similarly sized
    candidates; distinct nearby objects remain separate under the existing NMS
    threshold.

    ``candidates`` contains ``(confidence, xyxy)`` pairs and the result keeps
    the same confidence-ordered list shape.
    """
    clusters = []
    for score, box in sorted(candidates, key=lambda item: float(item[0]), reverse=True):
        box = np.asarray(box, np.float32).reshape(4)
        if not np.isfinite(box).all():
            continue
        area = float(np.prod(np.maximum(box[2:] - box[:2], 0.)))
        selected = None
        for cluster in clusters:
            cluster_box = cluster['box']
            cluster_area = cluster['area']
            ratio = max(area, cluster_area) / max(min(area, cluster_area), 1e-6)
            if ratio <= max_area_ratio and _box_iou(box, cluster_box) >= min_iou:
                selected = cluster
                break
        weight = max(float(score), .05) ** 2
        if selected is None:
            clusters.append(dict(box=box.copy(), area=area, weight=weight,
                                 score=float(score), members=1))
            continue
        if np.max(np.abs(selected['box'] - box)) <= 1e-5:
            selected['score'] = max(selected['score'], float(score))
            selected['members'] += 1
            continue
        selected['box'] = ((selected['box'] * selected['weight'] + box * weight) /
                           (selected['weight'] + weight)).astype(np.float32)
        selected['area'] = float(np.prod(np.maximum(
            selected['box'][2:] - selected['box'][:2], 0.)))
        selected['weight'] += weight
        selected['score'] = max(selected['score'], float(score))
        selected['members'] += 1
    return [(cluster['score'], cluster['box']) for cluster in
            sorted(clusters, key=lambda item: item['score'], reverse=True)]


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
    fused = fuse_view_candidates(candidates)
    for _, box in fused:
        if all(box_overlap(box, previous)[0] < .5 for previous in kept):
            kept.append(box)
    return kept


def detect_live_target_boxes(detector, frame, fallback_rotation=2):
    """Four inference passes per live round; all diagonal views in three rounds."""
    return detect_target_boxes(detector, frame, fallback_rotation, progressive=True)
