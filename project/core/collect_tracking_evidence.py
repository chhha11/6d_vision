"""Guided, depth-verified reference collection and annotated live recordings."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import live_depth_6dof as live
from .capture_global_dataset import RoiSelector
from .collection_recording import SessionRecording
from .collection_ui import instructions, progress_preview, rejection_help, chinese_font
from .paths import DATA_ROOT
from .rgb_global_initializer import prepare_initializer, verified_reference_paths

PLAN = [
    ('failure_view', '摆成最容易长时间不识别的朝向，保持静止'),
    ('failure_oblique', '失败朝向稍微转动约 20–30 度，保持静止'),
    ('upright_front', '直立，常见正面朝向相机'),
    ('upright_back', '直立，转到平时较少出现的背面'),
    ('upright_left', '直立，转到左侧面'),
    ('upright_right', '直立，转到右侧面'),
    ('horizontal_left', '横置，长轴朝左，目标完整入镜'),
    ('horizontal_right', '横置，长轴朝右，目标完整入镜'),
    ('diagonal_left', '向左倾斜约 45 度，目标完整入镜'),
    ('diagonal_right', '向右倾斜约 45 度，目标完整入镜'),
]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding='utf-8')


def snapshot_file(path, folder):
    destination = folder / Path(path).name
    shutil.copy2(path, destination)
    return dict(source=str(Path(path).resolve()), snapshot=str(destination.resolve()),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def save_manifest(path, **updates):
    data = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    data.update(updates)
    write_json(path, data)


def validate_reference(attempt, args):
    # Exactly the production admission criteria, including the stricter
    # reference stillness gates (a trackable pose is not necessarily a reference).
    root = attempt.parent.parent
    pose_path = attempt / 'rgb_initial_pose.json'
    if pose_path not in verified_reference_paths(root):
        raise RuntimeError('Reference rejected: depth/stillness gates did not pass; keep still and retry')
    initializer = prepare_initializer(args.model, args.intrinsics, root=root)
    views = [v for v in initializer.views if Path(v.name) == pose_path.resolve()]
    if not views:
        raise RuntimeError('Reference rejected: fewer than 12 usable CAD feature points; change lighting/view')
    return {v.descriptor_kind: len(v.points) for v in views}


def publish_reference(attempt, destination, args):
    # Revalidate just before publication. Incomplete copies never enter the loader.
    feature_counts = validate_reference(attempt, args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / ('.' + destination.name + '_publishing')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Reference destination already exists: {destination}')
    shutil.copytree(attempt, temporary)
    payload = json.loads((temporary / 'rgb_initial_pose.json').read_text())
    for key in ('rgb_image', 'depth_burst', 'extrinsics', 'intrinsics'):
        source = Path(payload[key])
        if source.is_relative_to(attempt.resolve()):
            payload[key] = str(destination.resolve() / source.relative_to(attempt.resolve()))
    payload['collection_review'] = dict(accepted=True, feature_counts=feature_counts,
                                         reviewed_at=datetime.now().astimezone().isoformat())
    write_json(temporary / 'rgb_initial_pose.json', payload)
    temporary.rename(destination)
    return feature_counts


def capture_reference(cap, attempt, roi, args, camera_index, camera_name):
    """Release RGB for native depth; return reopened camera even after rejection."""
    attempt.mkdir(parents=True, exist_ok=False)
    before = live.read_rgb(cap, 3)
    live.write_image(attempt / 'rgb_before.png', before)
    extrinsics = attempt / 'rgb_depth_extrinsics.npz'
    intrinsics = attempt / ('rgb_intrinsics' + args.intrinsics.suffix)
    shutil.copy2(args.extrinsics, extrinsics)
    shutil.copy2(args.intrinsics, intrinsics)
    x, y, w, h = roi
    with np.load(extrinsics) as archive:
        constraint = live.RgbConstraint(np.array([x, y, x + w, y + h], np.float64),
                                        archive['K_rgb'].copy(), archive['dist_rgb'].copy(),
                                        archive['T_rgb_depth'].copy())
    cap.release()
    cap = None
    error = None
    try:
        live.run_with_preview(lambda: live.run_depth_worker(
            attempt / 'depth', args.depth_warmup, args.depth_timeout_s,
            args.depth_frame_timeout_s, calibration_path=extrinsics), before,
            'Capturing reference depth | keep still', preview_renderer=progress_preview)
    except (live.PreviewCancelled, live.DepthWorkerUnreapedError):
        raise  # Never reopen RGB while native depth worker may still own the device.
    except (OSError, RuntimeError, ValueError) as exc:
        error = exc
    cap, after = live.recover_rgb(args, camera_index, before, camera_name=camera_name)
    try:
        if error is not None:
            raise error
        live.write_image(attempt / 'rgb_after.png', after)
        def fit():
            diagnostics = live.fit_depth_capture(attempt / 'depth', attempt, args.model,
                                                  args.settings, constraint)
            live.initialize_capture(attempt / 'rgb_before.png', attempt / 'depth', attempt,
                                    args.model, extrinsics, intrinsics, args.settings,
                                    args.box_display_y_offset_px, attempt / 'rgb_after.png', diagnostics)
            return validate_reference(attempt, args)
        counts = live.run_with_preview(fit, after, 'Checking CAD fit and reference features', cap,
                                       preview_renderer=progress_preview)
        write_json(attempt / 'collection_check.json', dict(accepted=True, features=counts,
                                                          rgb_roi_xywh=list(roi),
                                                          display_y_offset_px=args.box_display_y_offset_px))
        return cap, counts, None
    except live.PreviewCancelled:
        cap.release()
        raise
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        write_json(attempt / 'collection_check.json', dict(accepted=False, reason=str(exc),
                                                          rgb_roi_xywh=list(roi)))
        return cap, None, str(exc)
    except BaseException:
        cap.release()
        raise


def review(attempt, counts):
    # Render actual CAD contour at zero display offset, not just its bounding box.
    data = json.loads((attempt / 'rgb_initial_pose.json').read_text())
    frame = live.read_image(Path(data['rgb_image']))
    from .cad_video_tracker import render_cad_pose_layer
    from .surface_seeding import VisibleSurfaceSeeder
    diagnostics = json.loads((attempt / 'depth_fit/diagnostics.json').read_text())
    model = Path(diagnostics['input']['model'])
    K, dist, _ = live.load_calibration(Path(data['intrinsics']))
    vertices, _, _ = live.make_edge_points(model, 700, 3)
    rvec = np.array(data['rvec_rgb_model_rad'])
    tvec = np.array(data['tvec_rgb_model_mm'])
    seeder = VisibleSurfaceSeeder(model, K, dist)
    check = json.loads((attempt / 'collection_check.json').read_text()) if (attempt / 'collection_check.json').exists() else {}
    offset = float(check.get('display_y_offset_px', live.load_box_display_y_offset()))
    raw_layer = render_cad_pose_layer(frame.shape, vertices, K, dist, rvec, tvec,
                                      seeder, 0., show_axes=True)
    layer = render_cad_pose_layer(frame.shape, vertices, K, dist, rvec, tvec,
                                  seeder, offset, show_axes=True)
    overlay = cv2.addWeighted(frame, 1., layer, .72, 0)
    raw_overlay = cv2.addWeighted(frame, 1., raw_layer, .72, 0)
    live.write_image(attempt / 'review_cad.png', overlay)
    live.write_image(attempt / 'review_cad_unshifted.png', raw_overlay)
    write_json(attempt / 'review_display.json', dict(display_y_offset_px=offset,
               pose_modified=False, compensated_image='review_cad.png',
               unshifted_image='review_cad_unshifted.png'))
    print(f'几何检查通过，特征数量 {counts}。显示下移 {offset:g} 像素；A 接受，R 重拍，Q 退出。', flush=True)
    show_raw = False
    while True:
        shown_offset = 0. if show_raw else offset
        canvas = instructions(raw_overlay if show_raw else overlay, '第 5 步：请检查绿色轮廓',
                              f'当前显示纵向偏移：{shown_offset:+g} 像素（正数向下）。',
                              '观察轮廓的位置、朝向、大小是否贴合实物。',
                              'A：显示贴合，接受样本   R：明显偏离，重新采集',
                              'B：切换未补偿投影   Q：保存退出')
        cv2.imshow(live.WINDOW, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('b'), ord('B')):
            show_raw = not show_raw
        if key in (ord('a'), ord('A')):
            return True
        if key in (ord('r'), ord('R')):
            return False
        if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(live.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            raise live.PreviewCancelled()


def run_references(options, args):
    manifest_path = options.output / 'session.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    completed = manifest.get('references', {})
    published = Path(manifest['published_reference_session'])
    chinese_font()
    args.reference_preview_renderer = progress_preview
    camera_index, camera_name = live.resolve_camera_index(args.camera_name, args.camera)
    cap = None
    cv2.namedWindow(live.WINDOW, cv2.WINDOW_NORMAL)
    selector = RoiSelector(live.IMAGE_SIZE)
    def on_mouse(event, x, y, flags, param):
        # Instruction panel is below the original image; never start a box there.
        if event == cv2.EVENT_LBUTTONDOWN and (y >= live.IMAGE_SIZE[1] or x >= live.IMAGE_SIZE[0]):
            return
        selector.on_mouse(event, x, y, flags, param)
    cv2.setMouseCallback(live.WINDOW, on_mouse)
    serial = len(list((options.output / 'references').glob('*/attempt_*'))) + 1
    try:
        cap = live.open_camera(camera_index, *live.IMAGE_SIZE, args.camera_fourcc, args.camera_fps)
        for step, (name, instruction) in enumerate(PLAN, 1):
            if completed.get(name, {}).get('accepted'):
                continue
            selector.reset()
            print(f'\n[{name}] {instruction}\n鼠标框住整个目标，移开手，静止至少 2 秒后按 S；N 暂跳过，Q 保存退出。', flush=True)
            feedback = '请用鼠标拖框，完整包住目标，尽量少带背景。'
            while True:
                frame = live.read_rgb(cap, 1)
                canvas = frame.copy()
                box = selector.drag_roi or selector.roi
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(canvas, (x, y), (x+w, y+h), (40, 240, 40), 2)
                canvas = instructions(canvas, f'姿态 {step}/{len(PLAN)}：{instruction}',
                                      feedback, '框选后移开手，静止至少 2 秒，再按 S 开始采集。',
                                      '采集和检查期间，相机与目标都要保持静止。',
                                      'S：开始采集   N：暂时跳过   Q：保存退出')
                cv2.imshow(live.WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(live.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return
                if key in (ord('n'), ord('N')):
                    break
                if key not in (ord('s'), ord('S')):
                    continue
                if selector.roi is None:
                    feedback = '还没有框选：请先用鼠标拖框，完整包住目标。'
                    continue
                x, y, w, h = selector.roi
                if min(w, h) < 20 or x <= 3 or y <= 3 or x+w >= 637 or y+h >= 477:
                    feedback = '框太小或靠近边缘：请让目标完整入镜，留出边缘后重框。'
                    continue
                attempt = options.output / 'references' / name / f'attempt_{serial:03d}'
                serial += 1
                cap, counts, error = capture_reference(cap, attempt, selector.roi, args,
                                                       camera_index, camera_name)
                if error:
                    print(f'本次未入库：{error}\n保持静止/调整角度或照明后按 S 重试；N 跳过。', flush=True)
                    feedback = rejection_help(error)
                    continue
                if not review(attempt, counts):
                    write_json(attempt / 'user_review.json', dict(accepted=False))
                    feedback = '已退回重拍：调整目标并重新框选，静止后按 S。'
                    continue
                destination = published / attempt.name
                counts = publish_reference(attempt, destination, args)
                completed[name] = dict(accepted=True, source=str(attempt), published=str(destination),
                                       features=counts)
                write_json(attempt / 'user_review.json', dict(accepted=True, display_y_offset_px=args.box_display_y_offset_px))
                save_manifest(manifest_path, references=completed)
                print(f'已加入参考库：{destination}。下次启动自动加载。', flush=True)
                break
    except live.PreviewCancelled:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        save_manifest(manifest_path, references=completed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('references', 'motion'))
    parser.add_argument('--output', type=Path, help='New session directory, or resume reference collection')
    parser.add_argument('--duration', type=float, default=150., help='Motion session seconds; default 150')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--camera-name', default='USB Camera')
    parser.add_argument('--camera-fps', type=float, default=30.)
    parser.add_argument('--camera-fourcc', default='MJPG')
    parser.add_argument('--model', type=Path, default=live.DEFAULT_MODEL)
    parser.add_argument('--intrinsics', type=Path, default=live.DEFAULT_INTRINSICS)
    parser.add_argument('--extrinsics', type=Path, default=live.DEFAULT_EXTRINSICS)
    options = parser.parse_args(argv)
    if not np.isfinite(options.duration) or options.duration <= 0:
        parser.error('--duration must be finite and positive')
    if options.output is None:
        options.output = DATA_ROOT / 'tracking' / 'targeted_collection' / (
            datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + options.mode)
    options.output = options.output.resolve()
    return options


def main(argv=None):
    options = parse_args(argv)
    live_cli = ['--model', str(options.model.resolve()), '--intrinsics', str(options.intrinsics.resolve()),
                '--extrinsics', str(options.extrinsics.resolve()), '--camera-name', options.camera_name,
                '--camera-fps', str(options.camera_fps), '--camera-fourcc', options.camera_fourcc,
                '--output-dir', str(options.output / 'live'), '--no-region-tracking',
                '--detector-interval', '2', '--diagnostic-duration-s', str(options.duration)]
    if options.camera is not None:
        live_cli += ['--camera', str(options.camera)]
    args = live.parse_args(live_cli)
    manifest_path = options.output / 'session.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if options.mode != 'references' or manifest.get('mode') != options.mode:
            raise ValueError('Only reference sessions can resume; use a new output directory for motion')
        for key in ('model', 'intrinsics', 'extrinsics'):
            if hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest() != manifest['files'][key]['sha256']:
                raise ValueError(f'Cannot resume with changed {key}')
    else:
        options.output.mkdir(parents=True, exist_ok=False)
        config = options.output / 'configuration'
        config.mkdir()
        files = {name: snapshot_file(getattr(args, name), config)
                 for name in ('model', 'intrinsics', 'extrinsics')}
        write_json(manifest_path, dict(mode=options.mode,
            created_at=datetime.now().astimezone().isoformat(), files=files,
            live_arguments=live_cli, plan=[dict(id=k, instruction=v) for k, v in PLAN],
            published_reference_session=str(DATA_ROOT / 'tracking/live_depth_6dof' /
                ('collection_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))),
            note='Estimated poses are not independent 6DoF ground truth.'))
    print(f'采集目录：{options.output}', flush=True)
    result = 0
    try:
        if options.mode == 'references':
            run_references(options, args)
        else:
            print('先摆容易失败的朝向；U 未识别，F 偏离，T 停止，M 开始运动，V 可见，H 不可见，N 下一段，Q 结束。', flush=True)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            recording = SessionRecording(options.output / 'continuous', options.duration, options.camera_fps)
            args.collection_recording = recording
            try:
                result = live.run_live(args)
            finally:
                recording.close()
        save_manifest(manifest_path, status='saved', exit_code=result)
    except KeyboardInterrupt:
        save_manifest(manifest_path, status='interrupted')
        result = 130
    except Exception as exc:
        save_manifest(manifest_path, status='error', error=str(exc))
        raise
    print(f'已保存：{options.output}\n请提供此目录的完整路径；保留整个目录，包括失败样本。', flush=True)
    return result


if __name__ == '__main__':
    sys.exit(main())
