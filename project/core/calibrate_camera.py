"""Astra RGB calibration. Run `python -m core.calibrate_camera --help`."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .paths import CALIBRATION_ROOT

BOARD = (9, 6)  # INNER corners, not square count
CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
CAMERA_BACKEND_NAME = 'DSHOW' if sys.platform == 'win32' else 'ANY'


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, frame):
    ok, encoded = cv2.imencode('.png', frame)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(path)


def corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCornersSB(gray, BOARD, cv2.CALIB_CB_NORMALIZE_IMAGE)


def object_points(square_mm):
    obj = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:BOARD[0], :BOARD[1]].T.reshape(-1, 2) * square_mm
    return obj


def open_camera(index, width, height, camera_fourcc=None, camera_fps=None):
    cap = cv2.VideoCapture(index, CAMERA_BACKEND)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open camera {index}. Close other camera applications.')
    if camera_fourcc:
        if len(camera_fourcc) != 4:
            cap.release()
            raise ValueError('camera_fourcc must contain exactly four characters')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_fourcc))
    # The Astra DirectShow driver restarts the stream even when a property is
    # assigned its current value. Avoid paying that restart on every handoff.
    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) != width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if camera_fps is not None and camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(camera_fps))
    # Keep the live view close to the newest frame. Drivers may ignore this,
    # but it is harmless for backends that do not expose a configurable queue.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def probe(args):
    for index in range(args.count):
        cap = None
        try:
            cap = open_camera(index, args.width, args.height)
            ok, frame = cap.read()
            if ok:
                print(f'Camera {index}: available, actual {frame.shape[1]}x{frame.shape[0]}')
            else:
                print(f'Camera {index}: opened but no frame')
        except RuntimeError as exc:
            print(exc)
        finally:
            if cap is not None:
                cap.release()
    print('Index does not identify the device. Confirm the Astra view in capture preview.')


def capture(args):
    object_mode = args.no_checkerboard
    folder = args.images.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / 'capture_settings.json'
    settings = dict(camera_index=args.camera, backend=CAMERA_BACKEND_NAME,
                    board_size=list(BOARD),
                    square_size_mm=args.square_mm, image_size=[args.width, args.height],
                    subject='object' if object_mode else 'checkerboard')
    has_images = any(p.suffix.lower() in {'.png', '.jpg', '.jpeg'} for p in folder.iterdir())
    if meta_path.exists() and has_images:
        previous = json.loads(meta_path.read_text(encoding='utf-8'))
        if previous.get('subject', 'checkerboard') != settings['subject']:
            raise RuntimeError('Different image subject; use a new --images directory.')
        for key in ['camera_index', 'backend', 'board_size', 'square_size_mm', 'image_size']:
            if previous.get(key) != settings[key]:
                raise RuntimeError(f'Existing session has different {key}; use a new --images directory.')
    cap = open_camera(args.camera, args.width, args.height)
    window = 'Astra RGB - confirm correct camera | S save | Q quit'
    print('Confirm this is the external Astra COLOR image, not the built-in or IR camera.')
    print('S: save RAW object image. Q / Esc: quit.' if object_mode else
          'S: detect all 9x6 inner corners and save RAW image. Q / Esc: quit.')
    print('Keep the entire object sharp and visible.' if object_mode else
          'Move the board across the frame; vary distance and tilt. Keep all corners visible.')
    print(f'Images: {folder}')
    found, pts, tick = False, None, 0
    note = 'Object photo mode - press S when still' if object_mode else 'Move board; press S when still'
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera frame read failed.')
            actual = [frame.shape[1], frame.shape[0]]
            if actual != settings['image_size']:
                raise RuntimeError(f'Requested {settings["image_size"]}, actual {actual}. '
                                   'Select supported --width/--height; frames are never resized.')
            if not object_mode and tick % 5 == 0:
                found, pts = corners(frame)
            tick += 1
            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, BOARD, pts, found)
            color = (0, 210, 0) if found else (0, 160, 255)
            count = len(list(folder.glob('frame_*.png')))
            cv2.putText(preview, f'Camera {args.camera} | {actual[0]}x{actual[1]} | saved {count}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
            status = 'Object photo mode' if object_mode else f'9x6 corners: {"YES" if found else "NO"}'
            cv2.putText(preview, f'{status} | S save / Q quit',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
            cv2.putText(preview, note, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (0, 160, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')):
                valid = True if object_mode else corners(frame)[0]  # Check the SAME frame that will be saved.
                if valid:
                    path = folder / f'frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png'
                    meta_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
                    write_image(path, frame)
                    note = 'Saved raw image - change viewpoint'
                    print(f'Saved {path.name}')
                else:
                    note = 'Not saved: all 9x6 corners must be visible'
                    print(note)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calibrate(args):
    folder = args.images.resolve()
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files:
        raise RuntimeError(f'No calibration images in {folder}')
    metadata = folder / 'capture_settings.json'
    saved = None
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('subject', 'checkerboard') != 'checkerboard':
            raise RuntimeError('Calibration images must be checkerboard captures.')
        if tuple(saved.get('board_size', ())) != BOARD:
            raise RuntimeError('Checkerboard dimensions differ from capture metadata.')
        if not np.isclose(saved['square_size_mm'], args.square_mm):
            raise RuntimeError('Square size differs from capture metadata.')
    obj = object_points(args.square_mm)
    objects, points, accepted, rejected = [], [], [], []
    size = None
    sample = None
    for path in files:
        frame = read_image(path)
        if frame is None:
            raise RuntimeError(f'Unreadable image: {path}')
        this_size = (frame.shape[1], frame.shape[0])
        if saved is not None and tuple(saved['image_size']) != this_size:
            raise RuntimeError(f'Image size differs from capture metadata: {path}')
        if size is None:
            size = this_size
        if this_size != size:
            raise RuntimeError('Mixed image sizes; use separate calibration sessions.')
        found, pts = corners(frame)
        if not found:
            rejected.append(path.name)
            continue
        objects.append(obj.copy())
        points.append(pts)
        accepted.append(path.name)
        if sample is None:
            sample = frame
    if len(accepted) < 12:
        raise RuntimeError(f'Only {len(accepted)} valid images; need at least 12, preferably 25-35.')
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, points, size, None, None)
    errors = []
    for ob, pt, rv, tv in zip(objects, points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(ob, rv, tv, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((pt - projected) ** 2, axis=2)))))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = dict(board_size=list(BOARD), square_size_mm=args.square_mm,
                  image_size=list(size), camera_matrix=K.tolist(), dist_coeffs=dist.tolist(),
                  rms_px=float(rms), valid_images=len(accepted), rejected_images=rejected,
                  per_image_rms_px=dict(zip(accepted, errors)),
                  pose_translation_unit='mm', source_images=str(folder),
                  capture_settings=json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else None,
                  image_processing='Original frames, no resizing, no crop, no mirror')
    np.savez(output / 'camera_intrinsics.npz', K=K, dist=dist, camera_matrix=K, dist_coeffs=dist,
             image_size=np.array(size), board_size=np.array(BOARD), square_size_mm=args.square_mm,
             rms_px=rms, per_view_errors_px=np.array(errors), rvecs=np.array(rvecs), tvecs=np.array(tvecs))
    (output / 'calibration_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_image(output / 'undistorted_example.png', cv2.undistort(sample, K, dist, None, K))
    print(f'Valid images: {len(accepted)}; rejected: {len(rejected)}; image size: {size}')
    print(f'Overall RMS reprojection error: {rms:.4f} px')
    print('Worst views (review for blur, flat poses, or poor corner detection):')
    for name, error in sorted(zip(accepted, errors), key=lambda p: p[1], reverse=True)[:5]:
        print(f'  {name}: {error:.4f} px')
    print(f'Saved {output / "camera_intrinsics.npz"}')
    if rms > .5:
        print('RMS exceeds the 0.5 px target. Review images and report; do not accept by RMS alone.')
    if len(accepted) < 20:
        print('Fewer than 20 valid images; collect more diverse views.')
    print('Check independent images and edge straightness before accepting calibration.')


def validate(args):
    """Keep K/dist fixed; estimate only each validation board's pose."""
    with np.load(args.intrinsics) as data:
        K, dist = data['K'].copy(), data['dist'].copy()
        size = tuple(int(v) for v in data['image_size'])
        board_size = tuple(int(v) for v in data['board_size'])
        square_mm = float(data['square_size_mm'])
    if board_size != BOARD:
        raise RuntimeError('Intrinsics use a different checkerboard.')
    meta = args.images / 'capture_settings.json'
    if meta.exists():
        settings = json.loads(meta.read_text(encoding='utf-8'))
        if (settings.get('subject', 'checkerboard') != 'checkerboard'
                or tuple(settings.get('board_size', ())) != BOARD
                or tuple(settings['image_size']) != size
                or not np.isclose(settings['square_size_mm'], square_mm)):
            raise RuntimeError('Validation settings differ from calibration.')
    obj = object_points(square_mm)
    # Interleaved points span the board. Fit pose to one half, score the other half.
    grid = np.indices((BOARD[1], BOARD[0]))
    fit_mask = ((grid[0] + grid[1]).ravel() % 2 == 0)
    results, panels = [], []
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not files:
        raise RuntimeError('No validation images found.')
    for path in files:
        frame = read_image(path)
        if frame is None or (frame.shape[1], frame.shape[0]) != size:
            raise RuntimeError(f'Unreadable image or wrong image size: {path}')
        found, pts = corners(frame)
        if not found:
            results.append(dict(file=path.name, detected=False))
            continue
        ok, rv, tv = cv2.solvePnP(obj, pts, K, dist)
        if not ok:
            raise RuntimeError(f'Pose fitting failed: {path}')
        projected, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((pts - projected) ** 2, axis=2))))
        ok, sub_rv, sub_tv = cv2.solvePnP(obj[fit_mask], pts[fit_mask], K, dist)
        if not ok:
            raise RuntimeError(f'Split pose fitting failed: {path}')
        held, _ = cv2.projectPoints(obj[~fit_mask], sub_rv, sub_tv, K, dist)
        held_rms = float(np.sqrt(np.mean(np.sum((pts[~fit_mask] - held) ** 2, axis=2))))
        results.append(dict(file=path.name, detected=True, pose_fit_rms_px=rms,
                            held_out_corners_rms_px=held_rms))
        print(f'{path.name}: RMS {rms:.4f} px; held-out corners {held_rms:.4f} px')
        overlay = frame.copy()
        for measured, predicted in zip(pts.reshape(-1, 2), projected.reshape(-1, 2)):
            cv2.circle(overlay, tuple(np.rint(measured).astype(int)), 3, (0, 255, 0), 1)
            cv2.drawMarker(overlay, tuple(np.rint(predicted).astype(int)), (0, 0, 255),
                           cv2.MARKER_CROSS, 5, 1)
        undistorted = cv2.undistort(frame, K, dist, None, K)
        for image, title in [(overlay, f'Original: RMS {rms:.3f}px'),
                             (undistorted, 'Undistorted (same K, no crop)')]:
            cv2.rectangle(image, (0, 0), (size[0], 30), (255, 255, 255), -1)
            cv2.putText(image, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(cv2.resize(np.hstack([overlay, undistorted]), (960, 360)))
    accepted = [r for r in results if r['detected']]
    if not accepted:
        raise RuntimeError('No complete checkerboards detected in validation images.')
    report = dict(intrinsics=str(args.intrinsics.resolve()), intrinsics_refitted=False,
                  method='Fixed K/dist. Per-image pose fit; interleaved half-corner pose fit and remaining-corner score.',
                  detected_images=len(accepted), total_images=len(files),
                  aggregate_rms_px=float(np.sqrt(np.mean([r['pose_fit_rms_px']**2 for r in accepted]))),
                  held_out_corners_rms_px=float(np.sqrt(np.mean([r['held_out_corners_rms_px']**2 for r in accepted]))),
                  limitations='Image reprojection consistency only; not an independent metric position accuracy measurement.',
                  views=results)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Limit each contact sheet to five rows.
    for start in range(0, len(panels), 5):
        write_image(args.output / f'validation_preview_{start // 5 + 1}.png', np.vstack(panels[start:start+5]))
    print(f'Validation: {len(accepted)}/{len(files)} images; RMS {report["aggregate_rms_px"]:.4f} px; '
          f'held-out corners {report["held_out_corners_rms_px"]:.4f} px')
    print(f'Report: {args.output / "validation_report.json"}; intrinsics unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('probe', help='Check camera indices without opening preview windows')
    p.add_argument('--count', type=int, default=3)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.set_defaults(run=probe)
    p = commands.add_parser('capture', help='Preview and save checkerboard photographs')
    p.add_argument('--camera', type=int, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--images', type=Path, default=CALIBRATION_ROOT / 'images')
    p.add_argument('--square-mm', type=float, default=25.0)
    p.add_argument('--no-checkerboard', action='store_true', help='Save object photographs without corner detection')
    p.set_defaults(run=capture)
    p = commands.add_parser('calibrate', help='Estimate intrinsics from saved images')
    p.add_argument('--images', type=Path, default=CALIBRATION_ROOT / 'images')
    p.add_argument('--output', type=Path, default=CALIBRATION_ROOT)
    p.add_argument('--square-mm', type=float, default=25.0)
    p.set_defaults(run=calibrate)
    p = commands.add_parser('validate', help='Check independent photos using fixed intrinsics')
    p.add_argument('--images', type=Path, default=CALIBRATION_ROOT / 'validation_images')
    p.add_argument('--intrinsics', type=Path, default=CALIBRATION_ROOT / 'camera_intrinsics.npz')
    p.add_argument('--output', type=Path, default=CALIBRATION_ROOT / 'validation')
    p.set_defaults(run=validate)
    args = parser.parse_args()
    if hasattr(args, 'square_mm') and args.square_mm <= 0:
        parser.error('--square-mm must be positive')
    try:
        args.run(args)
    except (RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
