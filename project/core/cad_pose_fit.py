"""Fit a CAD pose to RGB edges from a reasonable initial pose.

This is a controlled static experiment. It is not an automatic detector: the
initial pose must put the model roughly over the object. Use the interactive
editor in cad_pose_overlay.py when the optimizer starts in the wrong basin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .cad_pose_overlay import (DEFAULT_INTRINSICS, DEFAULT_MODEL, load_calibration,
                              load_edges, pose_vector, project_overlay, save_result)


def image_points(vertices, edges):
    a, b = vertices[edges[:, 0]], vertices[edges[:, 1]]
    # Four interior points make the objective respond to long edges, while
    # keeping evaluation small enough for CPU-only testing.
    ts = np.array([0.0, .25, .5, .75, 1.0])[:, None, None]
    points = a[None] * (1 - ts) + b[None] * ts
    return points.reshape(-1, 3)


def rotations(args):
    return np.array([args.rx, args.ry, args.rz], np.float64)


def make_args(args, values):
    result = argparse.Namespace(**vars(args))
    result.rx, result.ry, result.rz, result.tx, result.ty, result.tz = values
    return result


def evaluate(values, args, points, K, dist, distance, shape):
    pose = make_args(args, values)
    rvec, tvec = pose_vector(pose)
    projected, _ = cv2.projectPoints(points, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    uv = projected.reshape(-1, 2)
    h, w = shape[:2]
    valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if valid.sum() < len(uv) * .5:
        return 1000.0 + float(len(uv) - valid.sum()), uv
    values_px = distance[np.rint(uv[valid, 1]).astype(int), np.rint(uv[valid, 0]).astype(int)]
    # Penalize projections leaving the image, and use a robust cap for wood
    # grain or reflections that are not part of the object.
    return float(np.mean(np.minimum(values_px, 30.0)) + 0.02 * (len(uv) - valid.sum())), uv


def fit(args):
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {args.image}')
    K, dist, size = load_calibration(args.intrinsics)
    if image.shape[1::-1] != size:
        raise RuntimeError(f'Image is {image.shape[1]}x{image.shape[0]}, calibration is {size[0]}x{size[1]}')
    vertices, edges = load_edges(args.model, args.max_edges)
    points = image_points(vertices, edges)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 45, 120)
    if args.roi:
        x1, y1, x2, y2 = args.roi
        roi = np.zeros_like(edge)
        roi[y1:y2, x1:x2] = edge[y1:y2, x1:x2]
        edge = roi
    # Distance to the closest detected image edge.
    distance = cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 3)
    current = np.array([args.rx, args.ry, args.rz, args.tx, args.ty, args.tz], np.float64)
    steps = np.array([args.angle_step, args.angle_step, args.angle_step, args.xy_step, args.xy_step, args.z_step], np.float64)
    best, _ = evaluate(current, args, points, K, dist, distance, image.shape)
    print(f'Initial edge distance: {best:.3f} px')
    for iteration in range(args.iterations):
        improved = False
        for index in range(6):
            local_best, local_value = best, current[index]
            for direction in (-1, 1):
                candidate = current.copy()
                candidate[index] += direction * steps[index]
                if index == 5 and candidate[index] <= 20:
                    continue
                score, _ = evaluate(candidate, args, points, K, dist, distance, image.shape)
                if score < local_best:
                    local_best, local_value = score, candidate[index]
            if local_best < best:
                current[index] = local_value
                best = local_best
                improved = True
        if not improved:
            steps *= .55
        if iteration % 5 == 0 or not improved:
            print(f'iter {iteration:02d}: {best:.3f} px pose={np.round(current, 2).tolist()} steps={np.round(steps, 2).tolist()}')
        if np.max(steps) < .35:
            break
    fitted = make_args(args, current)
    overlay = project_overlay(image, vertices, edges, K, dist, *pose_vector(fitted))
    output = args.output or args.image.with_name(args.image.stem + '_cad_fit.png')
    save_result(overlay, fitted, output)
    fit_report = output.with_name(output.stem + '_report.json')
    fit_report.write_text(json.dumps(dict(initial=[args.rx, args.ry, args.rz, args.tx, args.ty, args.tz],
                                           fitted=np.round(current, 6).tolist(), edge_distance_px=best,
                                           optimizer='coordinate descent on Canny edge distance',
                                           warning='Static initialization experiment; inspect overlay for physical fit.'), indent=2), encoding='utf-8')
    print(f'Saved {fit_report}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--intrinsics', type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-edges', type=int, default=7000)
    parser.add_argument('--rx', type=float, default=0)
    parser.add_argument('--ry', type=float, default=90)
    parser.add_argument('--rz', type=float, default=0)
    parser.add_argument('--tx', type=float, default=0)
    parser.add_argument('--ty', type=float, default=0)
    parser.add_argument('--tz', type=float, default=260)
    parser.add_argument('--angle-step', type=float, default=8)
    parser.add_argument('--xy-step', type=float, default=12)
    parser.add_argument('--z-step', type=float, default=25)
    parser.add_argument('--iterations', type=int, default=35)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
    args = parser.parse_args()
    try:
        fit(args)
    except (RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
