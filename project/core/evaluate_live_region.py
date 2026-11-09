"""Replay the live tracker with background YOLO and optional gated M3T.

Only the first valid synchronized recorded pose is supplied. This measures
behavior on a clip; it does not measure cold start or independent pose accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from . import cad_video_tracker as tracker
from .background_detector import BackgroundDetector
from .cad_pose_overlay import DEFAULT_INTRINSICS, DEFAULT_MODEL
from .evaluate_m3t import load_recording
from .live_depth_6dof import (DEFAULT_LIVE_DETECTOR_INTERVAL, DETECTOR_MODEL,
                              detect_target_box, prepare_live_detector)
from .m3t_live import NativeRegionTracker
from .paths import DATA_ROOT
from .rgb_global_initializer import prepare_initializer


class SlicedCapture:
    """Start exactly at the supplied valid pose and retain timestamp alignment."""
    def __init__(self, capture, offset, timestamps=None):
        self.capture, self.offset = capture, offset
        self.timestamps, self.started, self.index = timestamps, None, 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, offset)

    def read(self):
        if self.timestamps is not None and self.index < len(self.timestamps):
            if self.started is None:
                self.started = time.monotonic()
            due = self.started + float(self.timestamps[self.index] - self.timestamps[0])
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        result = self.capture.read()
        self.index += 1
        return result

    def get(self, key):
        value = self.capture.get(key)
        return value - self.offset if key == cv2.CAP_PROP_FRAME_COUNT else value

    def __getattr__(self, name):
        return getattr(self.capture, name)


def evaluate(session: Path, output: Path, region_tracking=True, realtime=True,
             detector_interval=DEFAULT_LIVE_DETECTOR_INTERVAL):
    cv2.setNumThreads(2)
    video, metadata, rows = load_recording(session)
    first = next(row for row in rows if row['valid'])
    offset = first['video_frame_index']
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, offset)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError('Cannot read the frame synchronized with the initial pose')
    finally:
        capture.release()
    cv2.imwrite(str(output / 'first.png'), image)
    pose = output / 'initial.json'
    pose.write_text(json.dumps(dict(rvec=first['rvec_rad'], tvec_mm=first['tvec_mm'],
                                    rgb_image=str((output/'first.png').resolve()))))
    (output/'replay_metadata.json').write_text(json.dumps(dict(
        source_video=str(video.resolve()), source_start_frame=offset,
        source_frame_offset=offset, pose_source='first_valid_recorded_pose_only',
        independent_pose_ground_truth=False, region_tracking=region_tracking,
        source_timestamp_pacing=realtime, detector_prewarmed=True,
        detector_interval=detector_interval),indent=2))
    args = tracker.parse_args([
        '--video',str(video),'--pose',str(pose),'--model',str(DEFAULT_MODEL),
        '--intrinsics',str(DEFAULT_INTRINSICS),'--headless',
        '--region-tracking' if region_tracking else '--no-region-tracking',
        '--no-static-correction','--no-session-depth-gate','--no-session-relocalization',
        '--surface-seeding','--rgb-pose-recovery','--fast-pose-recovery',
        '--three-d-only-after-init','--detector-interval',str(detector_interval),
        '--record-diagnostics','--diagnostic-duration-s','180',
        '--output',str(output/'last_pose.json'),'--summary',str(output/'summary.json'),
        '--log',str(output/'frames.csv'),'--xyz-jsonl',str(output/'xyz.jsonl')])
    args.rgb_initializer = prepare_initializer(DEFAULT_MODEL, DEFAULT_INTRINSICS)
    native = detector = None
    try:
        if region_tracking:
            native = NativeRegionTracker(DEFAULT_MODEL,np.array(metadata['camera_matrix']),
                np.array(metadata['distortion']),[640,480],output/'native')
            args.region_tracker_provider = lambda: native
        model = prepare_live_detector(DETECTOR_MODEL,{})
        detector = BackgroundDetector(lambda: model, detect_target_box)
        detector.loading.result(timeout=5.)
        original_capture = cv2.VideoCapture
        timestamps = tracker.load_video_timestamps(video)[offset:]
        def sliced_capture(*a, **kw):
            return SlicedCapture(original_capture(*a, **kw),offset,
                                 timestamps if realtime else None)
        with patch.object(cv2,'VideoCapture',side_effect=sliced_capture), \
                patch.object(tracker,'load_video_timestamps',return_value=timestamps):
            return tracker.run(args,detector=detector)
    finally:
        if native is not None:
            native.close()
        if detector is not None:
            detector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--region-tracking',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--realtime',action=argparse.BooleanOptionalAction,default=True,
                        help='Pace frames with original timestamps for async detector comparisons')
    parser.add_argument('--detector-interval', type=int, default=DEFAULT_LIVE_DETECTOR_INTERVAL,
                        help='Stable tracking check interval, matching the live entry point')
    args = parser.parse_args()
    if args.detector_interval < 1:
        parser.error('--detector-interval must be positive')
    raise SystemExit(evaluate(args.session,args.output,args.region_tracking,args.realtime,
                              args.detector_interval))


if __name__ == '__main__':
    main()
