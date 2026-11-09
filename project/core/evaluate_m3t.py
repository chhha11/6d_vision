"""Offline M3T region-tracking experiment; never changes the live backend.

The first valid recorded pose initializes each run. Later recorded poses are used only
for comparison. M3T API success is NOT a validated pose or a recovered target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .paths import PROJECT_ROOT, TRACKING_ROOT

WORKSPACE = PROJECT_ROOT.parent
DEFAULT_SESSIONS = ('20261103_232806', '20261103_233236')
UPSTREAM_COMMIT = 'f02106186a1e756450226b3b793e309506817c58'


def pose_meters(row):
    """Convert one recorded model-to-camera pose, without display offsets."""
    if not row.get('valid', False):
        raise ValueError('The initial recorded pose is invalid')
    rvec = np.asarray(row['rvec_rad'], np.float64).reshape(3)
    tvec = np.asarray(row['tvec_mm'], np.float64).reshape(3)
    if not np.isfinite(np.r_[rvec, tvec]).all() or tvec[2] <= 0:
        raise ValueError('The initial pose must be finite and in front of camera')
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec / 1000.
    return transform


def load_recording(session):
    session = Path(session)
    videos = sorted(session.glob('attempt_001/tracking/diagnostics_*/rgb.avi'))
    if len(videos) != 1:
        raise ValueError(f'Expected one attempt_001 video in {session}, found {len(videos)}')
    video = videos[0]
    metadata = json.loads((video.parent / 'metadata.json').read_text())
    rows = [json.loads(line) for line in (video.parent / 'poses.jsonl').read_text().splitlines()]
    if not rows or rows[0]['video_frame_index'] != 0:
        raise ValueError('A valid pose synchronized with video frame zero is required')
    first_valid = next((row for row in rows if row.get('valid')), None)
    if first_valid is None:
        raise ValueError('Recording has no valid initialization pose')
    pose_meters(first_valid)
    if len(rows) != len(metadata['frame_timestamps_s']):
        raise ValueError('Timestamp / pose counts do not match')
    for index, row in enumerate(rows):
        if row['video_frame_index'] != index:
            raise ValueError('Non-contiguous pose/video correspondence')
    return video, metadata, rows


def prepare_mesh(source, cache):
    """M3T's OBJ loader scales these original millimeter vertices by .001."""
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    output = cache / f'energy_unit_mm_{digest[:12]}.obj'
    if not output.exists():
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(str(source))
        if mesh.is_empty():
            raise ValueError(f'Empty CAD: {source}')
        if not o3d.io.write_triangle_mesh(str(output), mesh, write_triangle_uvs=False):
            raise RuntimeError('Could not convert CAD to OBJ')
    return output, digest


def write_config(path, *, video, mesh, metadata, initial_row, output,
                 cache, divides, image_size, max_frames):
    K = np.asarray(metadata['camera_matrix'], np.float64)
    dist = np.asarray(metadata['distortion'], np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics')
    if not np.isfinite(dist).all():
        raise ValueError('Invalid distortion coefficients')
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f'Cannot create {path}')
    try:
        for key, value in {'video': str(video), 'mesh': str(mesh),
                           'model_cache': str(cache), 'poses_csv': str(output / 'poses.csv'),
                           'native_summary': str(output / 'native_summary.yml')}.items():
            fs.write(key, value)
        fs.write('camera_matrix', K)
        fs.write('distortion', dist)
        fs.write('initial_pose_m', pose_meters(initial_row))
        fs.write('n_divides', divides)
        fs.write('model_image_size', image_size)
        fs.write('max_frames', max_frames)
        fs.write('start_frame', int(initial_row['video_frame_index']))
    finally:
        fs.release()


def read_native_poses(path, start_frame=0):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    poses = []
    for index, row in enumerate(rows):
        if int(row['frame_index']) != index + start_frame or int(row['api_ok']) != 1:
            raise ValueError('M3T output is incomplete or an API call failed')
        vector = np.array([float(row[k]) for k in ('rx', 'ry', 'rz', 'tx_mm', 'ty_mm', 'tz_mm')])
        if not np.isfinite(vector).all():
            raise ValueError(f'Nonfinite M3T pose at frame {index}')
        poses.append({'rvec_rad': vector[:3].tolist(), 'tvec_mm': vector[3:].tolist(),
                      'processing_ms': float(row['processing_ms'])})
    return poses


def lost_spans(rows, timestamps, start_frame=0):
    spans, start = [], None
    for i in range(len(rows) + 1):
        lost = i < len(rows) and rows[i].get('status') == 'TARGET LOST'
        if lost and start is None:
            start = i
        if not lost and start is not None:
            end = min(i, len(rows) - 1)
            spans.append({'start_frame': start + start_frame, 'end_frame_exclusive': i + start_frame,
                          'duration_s': round(timestamps[end] - timestamps[start], 4),
                          'ended_at_recording_end': i == len(rows)})
            start = None
    return spans


def edge_alignment(frame, mask):
    """Diagnostic only: scene edges can match the wrong object/background."""
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 140)
    distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {'contour_edge_median_px': None, 'contour_edge_support_3px': 0.}
    pixels = np.concatenate(contours).reshape(-1, 2)
    values = distance[pixels[:, 1], pixels[:, 0]]
    return {'contour_edge_median_px': float(np.median(values)),
            'contour_edge_support_3px': float(np.mean(values <= 3.))}


def comparison_images(video, metadata, old_rows, new_rows, output, start_frame=0):
    from .cad_video_tracker import make_edge_points
    from .pose_visualization import render_cad_pose_layer
    from .surface_seeding import VisibleSurfaceSeeder
    K = np.array(metadata['camera_matrix'], np.float64)
    dist = np.array(metadata['distortion'], np.float64)
    model = Path(metadata['model'])
    vertices, _, _ = make_edge_points(model, 700, 3)
    seeder = VisibleSurfaceSeeder(model, K, dist)
    count = len(new_rows)
    preferred = ([0, 180, 240, 400, 600, 704] if count < 1000
                 else [0, 600, 800, 1600, 2600, 3000])
    indices = sorted(set(min(i, count - 1) for i in preferred))
    cap = cv2.VideoCapture(str(video))
    contact, diagnostics = [], []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index + start_frame)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode sample frame {index}')
            views = []
            sample = {'frame_index': index + start_frame, 'source_time_s': metadata['frame_timestamps_s'][index],
                      'old_status': old_rows[index].get('status')}
            for name, pose in [('old', old_rows[index]), ('m3t', new_rows[index])]:
                r = np.array(pose['rvec_rad']); t = np.array(pose['tvec_mm'])
                # Measure unshifted CAD geometry on original distorted RGB.
                mask = seeder.mask(frame.shape, vertices, r, t, erode=False)
                sample[name] = edge_alignment(frame, mask)
                if name == 'old':
                    sample[name]['recorded_pose_valid'] = bool(pose.get('valid'))
                view = frame.copy()
                if name == 'm3t' or pose.get('valid', False):
                    layer = render_cad_pose_layer(frame.shape, vertices, K, dist, r, t,
                                                  seeder, display_y_offset_px=12.)
                    view = cv2.add(view, layer)
                header = ('OLD: ' + str(pose.get('status')) if name == 'old'
                          else 'M3T: pose UNVERIFIED')
                cv2.rectangle(view, (0, 0), (view.shape[1], 65), (0, 0, 0), -1)
                cv2.putText(view, f'{header} | frame {index + start_frame}', (8, 23),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(view, 'Display +12 px; metrics use unshifted pose', (8, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
                views.append(view)
            contact.append(np.hstack(views))
            diagnostics.append(sample)
    finally:
        cap.release()
    if not cv2.imwrite(str(output / 'comparison.jpg'), np.vstack(contact)):
        raise RuntimeError('Could not save comparison image')
    return diagnostics


def evaluate(session, output_root, binary, cache_root, divides, image_size, max_frames):
    video, metadata, original = load_recording(session)
    start_frame = next(i for i, row in enumerate(original) if row.get('valid'))
    original = original[start_frame:]
    metadata = dict(metadata, frame_timestamps_s=metadata['frame_timestamps_s'][start_frame:])
    output = output_root / Path(session).name
    output.mkdir(parents=True, exist_ok=True)
    mesh, digest = prepare_mesh(metadata['model'], cache_root)
    cache = cache_root / f'region_{digest[:12]}_d{divides}_i{image_size}_p200.bin'
    config = output / 'config.yml'
    write_config(config, video=video, mesh=mesh, metadata=metadata, initial_row=original[0],
                 output=output, cache=cache, divides=divides, image_size=image_size,
                 max_frames=max_frames)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '1'
    started = time.perf_counter()
    with (output / 'native.log').open('w') as log:
        subprocess.run([str(binary), str(config)], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    wall_s = time.perf_counter() - started
    poses = read_native_poses(output / 'poses.csv', start_frame)
    expected = min(len(original), max_frames) if max_frames else len(original)
    if len(poses) != expected:
        raise ValueError(f'Expected {expected} video frames, got {len(poses)}')
    originals = original[:len(poses)]
    timestamps = metadata['frame_timestamps_s'][:len(poses)]
    with (output / 'poses.jsonl').open('w') as stream:
        for index, pose in enumerate(poses):
            stream.write(json.dumps(dict(pose, frame_index=index + start_frame, source_elapsed_s=timestamps[index],
                coordinate_frame='rgb_camera_optical', units='mm', pose_validated=False)) + '\n')
    fs = cv2.FileStorage(str(output / 'native_summary.yml'), cv2.FILE_STORAGE_READ)
    try:
        setup_ms = fs.getNode('setup_ms').real()
    finally:
        fs.release()
    times = [p['processing_ms'] for p in poses[1:]]
    summary = {
        'session': Path(session).name, 'backend': 'M3T RGB region only',
        'upstream_commit': UPSTREAM_COMMIT, 'frames': len(poses),
        'source_video': str(video), 'source_frames_available': len(original),
        'processing_timing_scope': 'Remap and M3T tracking; excludes decode, camera, CSV, drawing and setup',
        'source_duration_s': timestamps[-1] - timestamps[0], 'wall_s': wall_s,
        'setup_ms': setup_ms, 'tracking_mean_ms': float(np.mean(times)) if times else None,
        'tracking_p95_ms': float(np.percentile(times, 95)) if times else None,
        'old_target_lost_frames': sum(r.get('status') == 'TARGET LOST' for r in originals),
        'old_lost_spans': lost_spans(originals, timestamps, start_frame),
        'm3t_validated_lost_frames': None, 'm3t_recovery_time_s': None,
        'caveat': 'API success and scene-edge alignment are not target-validity or pose-accuracy evidence. '
                  'No automatic global relocalization is included; no independent pose truth exists.',
        'initialization': 'First valid recorded pose only; no later pose injection',
        'source_start_frame': start_frame,
        'model_sha256': digest, 'geometry_mm_to_m': .001,
        'input_undistorted': True, 'display_offset_px': 12,
        'diagnostic_metrics_use_display_offset': False,
        'model_divides': divides, 'model_image_size': image_size,
        'sample_diagnostics': comparison_images(video, metadata, original, poses, output, start_frame),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('session', 'frames', 'setup_ms', 'tracking_mean_ms',
                                              'tracking_p95_ms', 'old_target_lost_frames')}, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', action='append', type=Path)
    parser.add_argument('--output', type=Path, default=TRACKING_ROOT / 'evaluations/m3t_region_20261104')
    parser.add_argument('--binary', type=Path, default=WORKSPACE / 'build/m3t/m3t_offline')
    parser.add_argument('--model-divides', type=int, choices=(2, 3, 4), default=3)
    parser.add_argument('--model-image-size', type=int, default=1000)
    parser.add_argument('--max-frames', type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_frames < 0 or args.model_image_size < 200:
        parser.error('max-frames must be >=0 and model-image-size must be >=200')
    args.output = args.output.resolve()
    cache = TRACKING_ROOT / 'm3t_cache'
    cache.mkdir(parents=True, exist_ok=True)
    sessions = args.session or [TRACKING_ROOT / 'live_depth_6dof' / stamp for stamp in DEFAULT_SESSIONS]
    summaries = [evaluate(s.resolve(), args.output, args.binary.resolve(), cache,
                         args.model_divides, args.model_image_size, args.max_frames) for s in sessions]
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
