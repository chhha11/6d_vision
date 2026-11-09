"""Interactive CAD pose overlay for the calibrated RGB camera.

Example:
  python -m core.cad_pose_overlay edit --image data/samples/energy_unit/frame_...png
  python -m core.cad_pose_overlay render --image data/samples/energy_unit/frame_...png \
      --rx 90 --ry 0 --rz 0 --tx 0 --ty 0 --tz 260

The centered CAD model uses millimetres. The camera calibration is for the
original 640x480 images and is never rescaled or cropped here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .paths import CALIBRATION_ROOT, MODEL_ROOT

DEFAULT_MODEL = MODEL_ROOT / 'energy_unit_centered_mm.stl'
DEFAULT_INTRINSICS = CALIBRATION_ROOT / 'camera_intrinsics.npz'
ANGLE_SCALE = 10.0  # 0.1 degree per trackbar count
TRANSLATION_SCALE = 2.0  # 0.5 mm per trackbar count


def load_calibration(path: Path):
    with np.load(path) as data:
        return data['K'].astype(np.float64), data['dist'].astype(np.float64), tuple(int(v) for v in data['image_size'])


def load_edges(path: Path, max_edges: int):
    import trimesh

    mesh = trimesh.load(path, force='mesh', process=False)
    if mesh.vertices.shape[1] != 3 or mesh.edges_unique.size == 0:
        raise RuntimeError(f'Cannot load a usable triangle mesh: {path}')
    edges = mesh.edges_unique
    if len(edges) > max_edges:
        ids = np.linspace(0, len(edges) - 1, max_edges, dtype=np.int64)
        edges = edges[ids]
    # Add a coarse set of vertices to make the overlay readable even when
    # the projected CAD edges fall between image pixels.
    return mesh.vertices.astype(np.float64), edges.astype(np.int32)


def pose_vector(args):
    rx, ry, rz = np.deg2rad([args.rx, args.ry, args.rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rvec, _ = cv2.Rodrigues(Rz @ Ry @ Rx)
    return rvec.reshape(3), np.array([args.tx, args.ty, args.tz], np.float64)


def project_overlay(image, vertices, edges, K, dist, rvec, tvec):
    projected, _ = cv2.projectPoints(vertices, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    projected = projected.reshape(-1, 2)
    canvas = image.copy()
    # Semi-transparent line layer keeps the original image visible.
    lines = np.zeros_like(canvas)
    h, w = canvas.shape[:2]
    for a, b in edges:
        p1, p2 = projected[a], projected[b]
        if not np.all(np.isfinite([*p1, *p2])):
            continue
        q1 = tuple(np.rint(p1).astype(int))
        q2 = tuple(np.rint(p2).astype(int))
        ok, q1, q2 = cv2.clipLine((0, 0, w, h), q1, q2)
        if ok:
            cv2.line(lines, q1, q2, (40, 220, 255), 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(canvas, 1.0, lines, 0.72, 0)
    # Object center and its long axis (+Z in the centered CAD frame).
    axis3d = np.array([[0, 0, 0], [0, 0, 75], [0, 0, -75]], np.float64)
    axis2d, _ = cv2.projectPoints(axis3d, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    axis2d = axis2d.reshape(-1, 2)
    center = tuple(np.rint(axis2d[0]).astype(int))
    cv2.circle(canvas, center, 5, (0, 0, 255), -1, cv2.LINE_AA)
    for end in axis2d[1:]:
        cv2.line(canvas, center, tuple(np.rint(end).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, 'CAD edges / center / long axis', (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def save_result(image, args, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise RuntimeError(f'Cannot write {output}')
    encoded.tofile(output)
    pose_file = output.with_suffix('.json')
    pose_file.write_text(json.dumps(dict(image=str(args.image.resolve()), model=str(args.model.resolve()),
                                         intrinsics=str(args.intrinsics.resolve()), units='mm',
                                         rvec_deg=[args.rx, args.ry, args.rz],
                                         tvec_mm=[args.tx, args.ty, args.tz]), indent=2), encoding='utf-8')
    print(f'Saved {output}')
    print(f'Saved {pose_file}')


def render(args):
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {args.image}')
    K, dist, size = load_calibration(args.intrinsics)
    if image.shape[1::-1] != size:
        raise RuntimeError(f'Image is {image.shape[1]}x{image.shape[0]}, calibration is {size[0]}x{size[1]}')
    vertices, edges = load_edges(args.model, args.max_edges)
    rvec, tvec = pose_vector(args)
    output = args.output or args.image.with_name(args.image.stem + '_cad_overlay.png')
    save_result(project_overlay(image, vertices, edges, K, dist, rvec, tvec), args, output)


def edit(args):
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot read image: {args.image}')
    K, dist, size = load_calibration(args.intrinsics)
    if image.shape[1::-1] != size:
        raise RuntimeError(f'Image is {image.shape[1]}x{image.shape[0]}, calibration is {size[0]}x{size[1]}')
    vertices, edges = load_edges(args.model, args.max_edges)
    window = 'CAD pose editor | S save | Q quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    names = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz']
    values = []
    for name in names:
        if name in {'rx', 'ry', 'rz'}:
            values.append(int(np.clip(round((getattr(args, name) + 180) * ANGLE_SCALE), 0, int(360 * ANGLE_SCALE))))
        elif name == 'tz':
            values.append(int(np.clip(round((getattr(args, name) - 100) * TRANSLATION_SCALE),
                                      0, int(1100 * TRANSLATION_SCALE))))
        else:
            values.append(int(np.clip(round((getattr(args, name) + 300) * TRANSLATION_SCALE),
                                      0, int(600 * TRANSLATION_SCALE))))
    # Trackbars use angle [0,3600] -> [-180,180] in 0.1 degree steps;
    # translation [0,1200] -> [-300,300] and depth [0,2200] -> [100,1200] mm.
    for name, value in zip(names, values):
        limit = int(360 * ANGLE_SCALE) if name in {'rx', 'ry', 'rz'} else (
            int(1100 * TRANSLATION_SCALE) if name == 'tz' else int(600 * TRANSLATION_SCALE))
        cv2.createTrackbar(name, window, value, limit, lambda _: None)
    print('Adjust sliders. Angles move in 0.1 degree steps; translations move in 0.5 mm steps.')
    print('S saves the visible overlay and JSON pose. Q/Esc exits.')
    while True:
        raw = {name: cv2.getTrackbarPos(name, window) for name in names}
        for name in ('rx', 'ry', 'rz'):
            raw[name] = raw[name] / ANGLE_SCALE - 180
        raw['tx'] = raw['tx'] / TRANSLATION_SCALE - 300
        raw['ty'] = raw['ty'] / TRANSLATION_SCALE - 300
        raw['tz'] = raw['tz'] / TRANSLATION_SCALE + 100
        current = argparse.Namespace(**vars(args))
        for name in names:
            setattr(current, name, raw[name])
        overlay = project_overlay(image, vertices, edges, K, dist, *pose_vector(current))
        cv2.putText(overlay,
                    f'R=({current.rx:.1f},{current.ry:.1f},{current.rz:.1f}) deg  '
                    f't=({current.tx:.1f},{current.ty:.1f},{current.tz:.1f}) mm',
                    (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(window, overlay)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if key in (ord('s'), ord('S')):
            output = args.output or args.image.with_name(args.image.stem + '_cad_overlay.png')
            save_result(overlay, current, output)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('render', 'edit'):
        p = sub.add_parser(command)
        p.add_argument('--image', type=Path, required=True)
        p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
        p.add_argument('--intrinsics', type=Path, default=DEFAULT_INTRINSICS)
        p.add_argument('--output', type=Path)
        p.add_argument('--max-edges', type=int, default=3500)
        p.add_argument('--rx', type=float, default=90)
        p.add_argument('--ry', type=float, default=0)
        p.add_argument('--rz', type=float, default=0)
        p.add_argument('--tx', type=float, default=0)
        p.add_argument('--ty', type=float, default=0)
        p.add_argument('--tz', type=float, default=260)
        p.set_defaults(run=render if command == 'render' else edit)
    args = parser.parse_args()
    try:
        args.run(args)
    except (RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
